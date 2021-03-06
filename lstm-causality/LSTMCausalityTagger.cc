#include <algorithm>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/combine.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cnn/expr.h"
#include "cnn/model.h"
#include "BecauseData.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"
#include "utilities.h"

using namespace std;
using namespace boost::algorithm;
using namespace cnn;
using namespace cnn::expr;
using namespace lstm_parser;
typedef BecauseRelation::IndexList IndexList;

#undef NDEBUG  // Keep asserts
#include <cassert>


const std::vector<unsigned> LSTMCausalityTagger::ALL_INDICES(0);


LSTMCausalityTagger::LSTMCausalityTagger(const string& parser_model_path,
                                         const TaggerOptions& options)
    : options(options), parser(parser_model_path),
      sentence_lstms({L1_lstm, L2_lstm, L3_lstm, L4_lstm, action_history_lstm,
                      connective_lstm, cause_lstm, effect_lstm, means_lstm}),
      persistent_lstms({L1_lstm, L2_lstm, L3_lstm, L4_lstm,
                        action_history_lstm}) {
  vocab = *parser.GetVocab();  // now that parser is initialized, copy vocab
  // Reset actions
  vocab.action_names.clear();
  vocab.actions_to_arc_labels.clear();
  // We don't care about characters
  vocab.chars_to_int.clear();
  vocab.int_to_chars.clear();

  initial_param_pool_state = cnn::ps->get_state();

  InitializeModelAndBuilders();
}


void LSTMCausalityTagger::LoadModel(const string& model_path) {
  cerr << "Loading tagger model from " << model_path << "...";
  model.release();
  cnn::ps->restore_state(initial_param_pool_state);
  auto t_start = chrono::high_resolution_clock::now();
  ifstream model_file(model_path.c_str(), ios::binary);
  if (!model_file) {
    cerr << "Unable to open model file; aborting" << endl;
    abort();
  }
  eos::portable_iarchive archive(model_file);
  archive >> *this;
  auto t_end = chrono::high_resolution_clock::now();
  auto ms_passed = chrono::duration<double, milli>(t_end - t_start).count();
  cerr << "done. (Loading took " << ms_passed << " milliseconds.)" << endl;
}


void LSTMCausalityTagger::InitializeModelAndBuilders() {
  model.reset(new cnn::Model);

  L1_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                        options.lambda_hidden_dim, model.get());
  L2_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                        options.lambda_hidden_dim, model.get());
  L3_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                        options.lambda_hidden_dim, model.get());
  L4_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                        options.lambda_hidden_dim, model.get());
  action_history_lstm = LSTMBuilder(options.lstm_layers, options.action_dim,
                                    options.actions_hidden_dim, model.get());
  if (options.parse_path_hidden_dim > 0) {
    unsigned pp_input_dim =
        options.parse_path_arc_dim > 0 ?
            // If using relations directly, add a bit to rel_dim for edge dir
            options.parse_path_arc_dim : parser.options.rel_dim + 1;
    parse_path_lstm = LSTMBuilder(options.lstm_layers,
                                  pp_input_dim,
                                  options.parse_path_hidden_dim, model.get());
  }
  connective_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                                options.span_hidden_dim, model.get());
  cause_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                           options.span_hidden_dim, model.get());
  effect_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                            options.span_hidden_dim, model.get());
  means_lstm = LSTMBuilder(options.lstm_layers, options.token_dim,
                           options.span_hidden_dim, model.get());
}


void LSTMCausalityTagger::Train(BecauseOracleTransitionCorpus* corpus,
                                vector<unsigned> selections, double dev_pct,
                                bool compare_punct, const string& model_fname,
                                unsigned update_groups_between_evals,
                                double epochs_cutoff,
                                double recent_improvements_cutoff,
                                double improvement_epsilon,
                                const volatile sig_atomic_t* requested_stop) {
  const unsigned num_sentences = selections.size();
  // selections gives us the subcorpus to use for training at all. But we'll
  // still keep shuffling each time we've gone through all the training
  // sentences, so we track the order imposed by that shuffle.
  vector<unsigned> sub_order(num_sentences);
  iota(sub_order.begin(), sub_order.end(), 0);
  const unsigned status_every_i_iterations = min(100u, num_sentences);
  cerr << "NUMBER OF TRAINING SENTENCES: " << num_sentences << endl;
  time_t time_start = chrono::system_clock::to_time_t(
      chrono::system_clock::now());
  cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z")
       << endl;

  SimpleSGDTrainer sgd(model.get());
  //MomentumSGDTrainer sgd(model);
  sgd.eta_decay = 0.08;
  //sgd.eta_decay = 0.05;

  if (options.dropout) {
    for (auto &builder : sentence_lstms) {
      builder.get().set_dropout(options.dropout);
    }
  }

  unsigned num_sentences_dev = round(dev_pct * num_sentences);
  unsigned num_sentences_train = num_sentences - num_sentences_dev;

  unsigned sentences_seen = 0;
  double best_dev_score = -numeric_limits<double>::infinity();
  double last_epoch_saved = nan("");
  double last_dev_score = best_dev_score;  // -inf
  double evaluations_per_epoch = num_sentences_train / static_cast<double>(
      status_every_i_iterations * update_groups_between_evals);
  boost::circular_buffer<bool> recent_evals_improved(
      evaluations_per_epoch * epochs_cutoff);

  unsigned sentence_i = num_sentences_train;
  for (unsigned update_group_num = 0; !requested_stop || !(*requested_stop);
       ++update_group_num) {
    unsigned actions_seen = 0;
    double correct = 0;
    double llh = 0;

    for (unsigned iter_i = 0; iter_i < status_every_i_iterations; ++iter_i) {
      if (sentence_i == num_sentences_train) {
        sentence_i = 0;
        if (sentences_seen > 0) {
          sgd.update_epoch();
        }
        cerr << "**SHUFFLE\n";
        random_shuffle(sub_order.begin(), sub_order.end());
      }

      unsigned sentence_index = selections[sub_order[sentence_i]];
      Sentence* sentence = &corpus->sentences[sentence_index];
      const vector<unsigned>& correct_actions =
          corpus->correct_act_sent[sentence_index];
      // Cache the decoded gold relations, if not already cached, and record
      // the corresponding sets of connective words.
      GetDecodedGoldRelations(*sentence, correct_actions);

      // cerr << "Starting sentence " << *sentence << endl;

      ComputationGraph cg;
      vector<unsigned> parse_actions = parser.LogProbTagger(
          &cg, *sentence, true, GetCachedParserStates());
      // Cache parse if we haven't yet.
      double parser_lp = as_scalar(cg.incremental_forward());
      CacheParse(sentence, corpus, sentence_index, parse_actions, parser_lp);
      LogProbTagger(&cg, *sentence, sentence->words, true, correct_actions,
                    &correct);
      double lp = as_scalar(cg.incremental_forward());
      if (lp < 0) {
        cerr << "Log prob " << lp << " < 0 on sentence "
             << corpus->sentences[sentence_index]<< endl;
        abort();
      }
      cg.backward();
      sgd.update(1.0);

      llh += lp;
      ++sentence_i;
      actions_seen += correct_actions.size();
      // With oracle connectives, all NEW-CONN, CONN-FRAG, and SPLIT actions are
      // not actually being predicted. So subtract them from any error metrics.
      if (options.oracle_connectives) {
        actions_seen -= oracle_actions_taken;
      }
      sentences_seen += 1;
    }

    sgd.status();
    time_t time_now = chrono::system_clock::to_time_t(
        chrono::system_clock::now());
    double epoch = sentences_seen / static_cast<double>(num_sentences_train);
    cerr << "update #" << update_group_num << " (epoch " << epoch
         << " | time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "
         << llh << " ppl: " << exp(llh / actions_seen) << " err: "
         << (actions_seen - correct) / actions_seen << endl;

    if (update_group_num % update_groups_between_evals == 0) {
      double dev_score = DoDevEvaluation(corpus, selections, compare_punct,
                                         num_sentences_train, update_group_num,
                                         sentences_seen);
      if (dev_score > best_dev_score) {
        best_dev_score = dev_score;
        SaveModel(model_fname, !isnan(last_epoch_saved));
        last_epoch_saved = epoch;
      }

      recent_evals_improved.push_back(
          dev_score - last_dev_score > improvement_epsilon);
      last_dev_score = dev_score;

      if (epoch - last_epoch_saved > epochs_cutoff && best_dev_score > 0) {
        unsigned num_recent_evals_improved = accumulate(
            recent_evals_improved.begin(), recent_evals_improved.end(), 0);
        double pct_recent_increases = num_recent_evals_improved
            / static_cast<double>(recent_evals_improved.size());
        if (pct_recent_increases < recent_improvements_cutoff) {
          cerr << "Reached cutoff for epochs with no increase in max dev score:"
               << ' ' << epoch - last_epoch_saved << " > " << epochs_cutoff
               << "; terminating training" << endl;
          break;
        }
      } else if (best_dev_score >= 0.999) {
        cerr << "Reached maximal dev performance; terminating training" << endl;
        break;
      }
    }
  }

  training_decoded_cache.clear();
  if (options.dropout) {
    for (auto &builder : sentence_lstms) {
      builder.get().disable_dropout();
    }
  }
}


double LSTMCausalityTagger::DoDevEvaluation(
    BecauseOracleTransitionCorpus* corpus,
    const vector<unsigned>& selections, bool compare_punct,
    unsigned num_sentences_train, unsigned iteration, unsigned sentences_seen) {
  // report on dev set
  const unsigned num_sentences = selections.size();
  double llh_dev = 0;
  double num_actions = 0;
  double correct_dev = 0;
  CausalityMetrics evaluation;
  const auto t_start = chrono::high_resolution_clock::now();
  for (unsigned sii = num_sentences_train; sii < num_sentences; ++sii) {
    unsigned sentence_index = selections[sii];
    Sentence* sentence = &corpus->sentences[sentence_index];
    const vector<unsigned>& correct_actions =
        corpus->correct_act_sent[sentence_index];

    ComputationGraph cg;
    vector<unsigned> parse_actions = parser.LogProbTagger(
        &cg, *sentence, true, GetCachedParserStates());
    double parse_lp = as_scalar(cg.incremental_forward());
    CacheParse(sentence, corpus, sentence_index, parse_actions, parse_lp);
    vector<unsigned> actions = LogProbTagger(&cg, *sentence, sentence->words,
                                             false, correct_actions,
                                             &correct_dev);
    llh_dev += as_scalar(cg.incremental_forward());
    vector<CausalityRelation> predicted = Decode(
        *sentence, actions, options.new_conn_action, options.shift_action);

    const vector<unsigned>& gold_actions =
        corpus->correct_act_sent[sentence_index];
    // NOTE: this assumes that gold relations from dev eval should be counted
    // for what's been seen. In current setup, that's definitely, true, since
    // they'll eventually come round and be used as training in a future epoch.
    const vector<CausalityRelation>& gold = GetDecodedGoldRelations(
        *sentence, gold_actions);

    num_actions += actions.size();
    if (options.oracle_connectives) {
      num_actions -= oracle_actions_taken;
    }
    const GraphEnhancedParseTree& parse =
        *corpus->sentence_parses[sentence_index];
    evaluation += CausalityMetrics(
        gold, predicted, *corpus, parse,
        SpanTokenFilter{compare_punct, *sentence, corpus->pos_is_punct});
  }

  auto t_end = chrono::high_resolution_clock::now();
  double epoch = sentences_seen / static_cast<double>(num_sentences_train);
  double error = (num_actions - correct_dev) / num_actions;
  cerr << "  **dev (iter=" << iteration << " epoch=" << epoch
       << ") llh=" << llh_dev << " ppl: " << exp(llh_dev / num_actions)
       << "\terr: " << error << " evaluation: \n" << evaluation
       << "\n[" << num_sentences - num_sentences_train << " sentences in "
       << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
       << endl;

  // TODO: use action-level accuracy all the time?
  if (options.oracle_connectives)
    return 1 - error;  // Use action-level accuracy
  else
    return evaluation.connective_metrics->GetF1();
}


vector<CausalityRelation> LSTMCausalityTagger::Decode(
    const Sentence& sentence, const vector<unsigned> actions,
    bool new_conn_is_action, bool shift_is_action) {
  // cerr << "Decoding sentence " << sentence << endl;

  vector<CausalityRelation> relations;
  vector<unsigned> lambda_1;
  deque<unsigned> lambda_2; // we'll be appending to the left end
  vector<unsigned> lambda_3;
  // We'll be moving stuff off of the left end of lambda_4.
  deque<unsigned> lambda_4;
  auto end = --sentence.words.end();  // Skip ROOT
  for (auto tok_iter = sentence.words.begin(); tok_iter != end; ++tok_iter) {
    lambda_4.push_back(tok_iter->first);
  }
  unsigned current_conn_token = sentence.words.begin()->first;
  unsigned current_arg_token = sentence.words.begin()->first;
  CausalityRelation* current_rel = nullptr;

  auto AdvanceArgTokenLeft = [&]() {
    assert(!lambda_1.empty());
    lambda_2.push_front(lambda_1.back());
    lambda_1.pop_back();
    current_arg_token = lambda_1.empty() ? lambda_4.front() : lambda_1.back();
  };

  auto AdvanceArgTokenRight = [&]() {
    assert(!lambda_4.empty());
    lambda_3.push_back(lambda_4.front());
    lambda_4.pop_front();
    if (!lambda_4.empty())
      current_arg_token = lambda_4.front();
  };

  auto EnsureCurrentRelation = [&](bool from_new_conn = false) {
    if (new_conn_is_action && !from_new_conn) {
        assert(current_rel);
    }
    if (!current_rel) {
      relations.emplace_back(
          sentence, CausalityRelation::CONSEQUENCE,
          IndexList({current_conn_token}));
      current_rel = &relations.back();
    }
  };

  auto AddArc = [&](unsigned action) {
    EnsureCurrentRelation();
    const string& arc_type = sentence.vocab->actions_to_arc_labels[action];
    vector<string> arg_names;
    boost::split(arg_names, arc_type, boost::is_any_of(","));
    for (const string& arg_name : arg_names) {
      auto arg_iter = find(CausalityRelation::ARG_NAMES.begin(),
          CausalityRelation::ARG_NAMES.end(), arg_name);
      assert(arg_iter != CausalityRelation::ARG_NAMES.end());
      current_rel->AddToArgument(
          arg_iter - CausalityRelation::ARG_NAMES.begin(), current_arg_token);
    }
  };

  auto FixArgsOrder = [&]() {
    // Transitions can scramble argument order. Make sure tokens are ordered by
    // their appearance in the sentence.
    for (unsigned arg_num = 0; arg_num < CausalityRelation::ARG_NAMES.size();
         ++arg_num) {
      CausalityRelation::IndexList* arg = current_rel->GetArgument(arg_num);
      sort(arg->begin(), arg->end());
    }
  };

  for (auto iter = actions.begin(), actions_end = actions.end();
       iter != actions_end; ++iter) {
    auto CompleteRelation = [&]() {  // defined here to capture iterators
      if (lambda_3.size() > 1) {  // contains more than just current token copy
        assert(current_rel);
        // Complete a relation.
        assert(lambda_4.empty() && lambda_1.empty());  // processed all tokens?
        while (!lambda_2.empty()) {  // move all of L2 back to L1
          lambda_1.push_back(move(lambda_2.front()));
          lambda_2.pop_front();
        }
        lambda_1.push_back(current_conn_token);
        // Move L3 back to L4, skipping current token copy.
        while (lambda_3.size() > 1) {
          lambda_4.push_front(move(lambda_3.back()));
          lambda_3.pop_back();
        }
        lambda_3.pop_back();  // current token copy
        current_conn_token = lambda_4.front();
        current_arg_token = lambda_1.back();
      } else {
        // If we have no more tokens to pull in, don't bother updating lambdas,
        // but we'd better be on the last action.
        assert(iter + 1 == actions_end);
      }

      FixArgsOrder();
      current_rel = nullptr;
    };

    unsigned action = *iter;
    const string& action_name = sentence.vocab->action_names[action];
    /*
    cerr << "Decoding action " << action_name << " on connective word \""
         << sentence.vocab->int_to_words.at(
                sentence.words.at(current_conn_token))
         << "\" and argument word \"" << sentence.vocab->int_to_words.at(
                sentence.words.at(current_arg_token)) << '"' << endl;
    //*/
    if (action_name == "NO-CONN") {
      // At a minimum, L4 should have the current token duplicate
      assert(!lambda_4.empty());
      lambda_1.push_back(current_conn_token);
      lambda_4.pop_front();// remove duplicate of current_conn_token
      if (!lambda_4.empty()) {
        current_conn_token = lambda_4.front();
        current_arg_token = lambda_1.back();
      } else {
        // If we have no more tokens to pull in, don't bother updating lambdas,
        // but we'd better be on the last action.
        assert(iter + 1 == actions_end);
      }
    } else if(action_name == "NEW-CONN") {
      assert(new_conn_is_action);  // otherwise this action shouldn't exist
      assert(!current_rel);
      EnsureCurrentRelation(true);
    } else if (action_name == "NO-ARC-LEFT") {
      EnsureCurrentRelation();
      AdvanceArgTokenLeft();
    } else if (action_name == "NO-ARC-RIGHT") {
      EnsureCurrentRelation();
      AdvanceArgTokenRight();
    } else if (action_name == "CONN-FRAG-RIGHT"  // LEFT not currently possible
               /*|| action_name == "CONN-FRAG-LEFT"*/) {
      assert(current_rel && current_arg_token != current_conn_token);
      current_rel->AddConnectiveToken(current_arg_token);
      // Do NOT advance the argument token. It could still be part of an arg.
    } else if (starts_with(action_name, "RIGHT-ARC")) {
      AddArc(action);
      AdvanceArgTokenRight();
    } else if (::starts_with(action_name, "LEFT-ARC")) {
      AddArc(action);
      AdvanceArgTokenLeft();
    } else if (action_name == "SPLIT") {
      assert(current_rel && current_rel->GetConnectiveIndices()->size() > 1);
      FixArgsOrder();  // Fix order before cloning the causal language instance
      // Make a copy of the current relation.
      relations.push_back(*current_rel);
      current_rel = &relations.back();
      // Find the last connective word that shares the same lemma, if possible.
      // (SPLITs only happen with function words, so using surface forms
      // instead of lemmas is OK. Which is good, b/c we don't have any lemmas.)
      // Default is cut off after initial connective word.
      unsigned connective_repeated_token_index_index = 1;
      IndexList* connective_indices = current_rel->GetConnectiveIndices();
      // (Don't bother checking 1; that's our default.)
      for (size_t i = connective_indices->size() - 1; i > 1; --i) {
        if (sentence.words.at((*connective_indices)[i])
            == sentence.words.at(current_arg_token)) {
          connective_repeated_token_index_index = i;
          break;
        }
      }
      unsigned connective_repeated_token_index =
          (*connective_indices)[connective_repeated_token_index_index];
      /*
      cerr << "Splitting " << *current_rel << " at "
           << sentence.vocab->int_to_words[sentence.words.at(
               connective_repeated_token_index)]
           << " (" << connective_repeated_token_index << ')' << endl;
      //*/

      // Now replace that index and everything after it with the new token.
      ThresholdedCmp<greater_equal<unsigned>> gte_repeated_index(
          connective_repeated_token_index);
      ReallyDeleteIf(connective_indices, gte_repeated_index);
      connective_indices->push_back(current_arg_token);
      // Delete all arg words that came after the previous last connective word.
      for (unsigned arg_num = 0; arg_num < current_rel->ARG_NAMES.size();
          ++arg_num) {
        IndexList* arg_indices = current_rel->GetArgument(arg_num);
        ReallyDeleteIf(arg_indices, gte_repeated_index);
      }
      // cerr << "Post-split instance: " << *current_rel << endl;
      // Don't advance the arg token.
    } else if (action_name == "SHIFT") {
      assert(shift_is_action);
      CompleteRelation();
    }

    // Don't wait for a SHIFT to complete the relation if not using SHIFTs.
    if (!shift_is_action && lambda_4.empty() && lambda_1.empty()) {
      CompleteRelation();
    }
  }

  return relations;
}


vector<CausalityRelation> LSTMCausalityTagger::Tag(
    const Sentence& sentence, GraphEnhancedParseTree* parse,
    const vector<unsigned>& gold_actions) {
  ComputationGraph cg;
  vector<unsigned> parse_actions = parser.LogProbTagger(
      &cg, sentence, true, GetCachedParserStates());
  double parser_lp = as_scalar(cg.incremental_forward());
  auto tree = parser.RecoverParseTree(sentence, parse_actions, parser_lp,
                                      true);
  *parse = GraphEnhancedParseTree(move(tree));
  vector<unsigned> actions = LogProbTagger(&cg, sentence, sentence.words, false,
                                           gold_actions);
  return Decode(sentence, actions, options.new_conn_action,
                options.shift_action);
}


void LSTMCausalityTagger::WriteSentenceResults(
    const vector<CausalityRelation>& predicted,
    const Sentence& sentence,
    const string** last_filename,
    ofstream* ann_out_file,
    string* current_txt_contents,
    unordered_map<IndexList, string, boost::hash<IndexList> >*
        current_doc_span_ids,
    unsigned* next_evt_id, unsigned* next_span_id) const {
  // cerr << "Writing sentence: " << sentence << endl;
  const BecauseSentenceMetadata& metadata =
      static_cast<const BecauseSentenceMetadata&>(*sentence.metadata);
  if (*last_filename != &metadata.ann_file_path) {
    ann_out_file->close();
    ann_out_file->open(metadata.ann_file_path, ios::app);
    if (!ann_out_file) {
      cerr << "Error: Could not open file " << metadata.ann_file_path
          << "; skipping output..." << endl;
      return;
    }

    // Read text file
    string txt_filename =
        metadata.ann_file_path.substr(0, metadata.ann_file_path.size() - 4)
        + ".txt";
    ifstream txt_file(txt_filename, std::ios::in | std::ios::binary);
    if (!txt_file) {
      cerr << "Error: Could not open text file for offset alignment for "
           << metadata.ann_file_path << "; skipping output..." << endl;
      return;
    }
    ostringstream txt_contents;
    txt_contents << txt_file.rdbuf();
    txt_file.close();
    *current_txt_contents = txt_contents.str();

    current_doc_span_ids->clear();
  }

  auto GetSpanID = [&](const CausalityRelation& relation,
                       const IndexList& indices, const string& span_type) {
    string* span_id_ptr = &(*current_doc_span_ids)[indices];
    if (span_id_ptr->empty()) {
      *span_id_ptr = "T" + to_string(*next_span_id);
      ++(*next_span_id);
      *ann_out_file << *span_id_ptr << '\t' << span_type << ' '
                    << IndicesToBratLine(sentence, indices,
                                         *current_txt_contents) << endl;
    }
    return *span_id_ptr;
  };

  try {
    for (const CausalityRelation& relation : predicted) {
      const string& trigger_id = GetSpanID(
          relation, relation.GetConnectiveIndices(), "Consequence");
      vector<string> arg_ids(relation.ARG_NAMES.size());
      for (unsigned arg_num = 0; arg_num < relation.ARG_NAMES.size();
           ++arg_num) {
        auto arg_indices = relation.GetArgument(arg_num);
        if (!arg_indices.empty()) {
          arg_ids[arg_num] = GetSpanID(relation, arg_indices, "Argument");
        }
      }
      *ann_out_file << 'E' << *next_evt_id << "\tConsequence:" << trigger_id;
      for (unsigned arg_num = 0; arg_num < relation.ARG_NAMES.size();
           ++arg_num) {
        if (!arg_ids[arg_num].empty()) {
          *ann_out_file << ' ' << relation.ARG_NAMES[arg_num] << ':'
              << arg_ids[arg_num];
        }
      }
      *ann_out_file << endl;
      //ann_out_file << 'A' << attr_id << "\tDegree " << 'E' << evt_id
      //             << " Facilitate" << endl;
      ++(*next_evt_id);
    }
  } catch(runtime_error& e) {
    cerr << "Failed to match tokens for BRAT in " << metadata.ann_file_path
         << ": " << e.what() << "; skipping remaining output for file" << endl;
    return;
  }
}


regex FRACTION_REGEX("^\\d*_\\d*/\\d*$");  // e.g., 2_1/2
const string LSTMCausalityTagger::IndicesToBratLine(
    const Sentence &sentence, const IndexList &token_indices,
    const string &current_txt) const {
  if (token_indices.empty()) {
    return "";
  }

  const BecauseSentenceMetadata& metadata =
      static_cast<const BecauseSentenceMetadata&>(*sentence.metadata);
  size_t txt_byte_index = metadata.document_byte_offset;
  size_t txt_char_index = multibyte_length(current_txt.c_str(), txt_byte_index);
  vector<pair<unsigned, unsigned>> brat_indices;
  vector<pair<unsigned, unsigned>> byte_indices;
  unsigned subspan_bytes_start;
  unsigned subspan_bytes_end;
  unsigned subspan_chars_start;
  unsigned subspan_chars_end;
  bool last_token_was_in_span = false;
  auto next_span_index_iter = token_indices.begin();
  unsigned token_char_index;

  auto record_next_indices = [&]() {
    byte_indices.push_back({subspan_bytes_start, subspan_bytes_end});
    brat_indices.push_back({subspan_chars_start, subspan_chars_end});
  };
  auto advance_stream = [&](unsigned bytes = 1, unsigned chars = 1,
                            unsigned token_chars = 1) {
    txt_byte_index += bytes;
    txt_char_index += chars;
    token_char_index += token_chars;
  };

  // For each token, match its characters against the characters in the .txt
  // file. Error out on any mismatch. As long as things continue to match,
  // track the txt file character counts for BRAT offsets.
  for (auto sentence_iter = sentence.words.begin();
       next_span_index_iter != token_indices.end()  // stop if no more span
           && sentence_iter->first != Corpus::ROOT_TOKEN_ID; ++sentence_iter) {
    unsigned token_idx = sentence_iter->first;
    const string& token_text = sentence.WordForToken(sentence_iter, token_idx);
    bool part_of_span = (token_idx == *next_span_index_iter);
    if (part_of_span) {
      ++next_span_index_iter;
      if (!last_token_was_in_span) {  // start of a new subspan
        subspan_bytes_start = txt_byte_index;
        subspan_chars_start = txt_char_index;
      }
    } else {  // not part of span, so finish off the previous span (if any).
      if (last_token_was_in_span) {  // previous subspan ended on last token
        record_next_indices();
      }
    }

    // Check all characters in the token.
    for (token_char_index = 0; token_char_index < token_text.size(); ) {
      // First check for multibyte.
      const char* byte_ptr = current_txt.c_str() + txt_byte_index;
      int bytes_len = mblen(byte_ptr, current_txt.size() - txt_byte_index);
      if (bytes_len < 0) {
        throw runtime_error("Invalid multibyte token. Next bytes: "
                            + std::string(current_txt.c_str()
                                          + txt_byte_index, 10));
      } else if (bytes_len > 1) {
        string text_bytes = current_txt.substr(txt_byte_index, bytes_len);
        string token_bytes = token_text.substr(token_char_index, bytes_len);
        if (text_bytes != token_bytes) {
          throw runtime_error(
              token_text + " -> " + token_bytes + " vs. " + text_bytes);
        }
        advance_stream(bytes_len, 1, bytes_len);
      } else {  // single-byte character (whew!)
        char next_txt_char = current_txt[txt_byte_index];
        char next_token_char = token_text[token_char_index];
        advance_stream();

        if (next_token_char != next_txt_char) {
          // Special case: quotes can be split into 2 characters in the original
          if (next_token_char == '"') {
            // Advance the stream: if a quote isn't next, we're giving up anyway
            char lookahead = current_txt[txt_byte_index];
            if ((next_txt_char == '`' && lookahead == '`')
                || (next_txt_char == '\'' && lookahead == '\'')) {
              advance_stream();
              continue;
            }

          // Special case: ellipses have spaces in between sometimes in the text
          } else if (next_token_char == '.') {
            if (token_text == "..." && next_txt_char == ' '
                && current_txt[txt_byte_index] == '.') {
              advance_stream(1, 1, 0);
              continue;
            }

          // Special case: fractions can have weird characters.
          } else if (next_token_char == '_') {
            if (regex_match(token_text, FRACTION_REGEX)) {
              continue;
            }
          }

          throw runtime_error(
              token_text + " -> " + string(1, next_token_char)
                  + " vs. " + string(1, next_txt_char));
        }
      }
    }

    if (part_of_span) {
      subspan_bytes_end = txt_byte_index;
      subspan_chars_end = txt_char_index;
    }

    last_token_was_in_span = part_of_span;

    // Advance stream past whitespace, recording if any newlines are found.
    char next_char;
    while(txt_byte_index < current_txt.size() - 1 &&
          isspace(next_char = current_txt[txt_byte_index])) {
      if (last_token_was_in_span) {
        if (next_char == '\n' || next_char == '\r') {
          record_next_indices();
          last_token_was_in_span = false;
        }
      }
      advance_stream();
    }
  }

  if (last_token_was_in_span) {  // a subspan was in progress at end-of-sentence
    record_next_indices();
  }

  // Reconstruct the full string of indices and the checksum text.
  ostringstream full_brat_string;
  for (auto iter = brat_indices.begin(); iter + 1 != brat_indices.end();
       ++iter) {
    full_brat_string << iter->first << ' ' << iter->second << ';';
  }
  full_brat_string << brat_indices.back().first << ' '
                   << brat_indices.back().second;
  full_brat_string << '\t';
  for (auto iter = byte_indices.begin(); iter + 1 != byte_indices.end();
       ++iter) {
    full_brat_string << current_txt.substr(iter->first,
                                           iter->second - iter->first) << ' ';
  }
  const auto& last_indices = byte_indices.back();
  full_brat_string << current_txt.substr(
      last_indices.first, last_indices.second - last_indices.first);

  return full_brat_string.str();
}


CausalityMetrics LSTMCausalityTagger::DoTest(
    BecauseOracleTransitionCorpus* corpus, bool evaluate, bool write_results,
    const vector<unsigned>& sentence_selections, bool compare_punct,
    bool pairwise, double overlap_threshold) {
  CausalityMetrics evaluation;

  const vector<unsigned>* real_sentence_selections = &sentence_selections;
  vector<unsigned> all_sentence_indices(0);
  if (&sentence_selections == &ALL_INDICES) {
    all_sentence_indices.resize(corpus->sentences.size());
    iota(all_sentence_indices.begin(), all_sentence_indices.end(), 0);
    real_sentence_selections = &all_sentence_indices;
  }

  const string* last_filename = nullptr;
  ofstream ann_out_file;
  string current_txt_contents;
  unsigned evt_id = 1000;  // BRAT IDs in the 1000s should be safe from clashes
  unsigned span_id = 1000;
  // unsigned attr_id = 1000;
  unordered_map<IndexList, string, boost::hash<IndexList>> current_doc_span_ids(
      10);

  for (unsigned sentence_index : *real_sentence_selections) {
    // Grab selected sentence and associated metadata from corpus.
    Sentence* sentence = &corpus->sentences[sentence_index];
    const vector<unsigned>& gold_actions =
        corpus->correct_act_sent[sentence_index];
    unsigned missing_instances =
        corpus->missing_instance_counts[sentence_index];
    const vector<BecauseOracleTransitionCorpus::ExtrasententialArgCounts>&
        missing_args = corpus->missing_arg_tokens[sentence_index];

    // Set up parse tree.
    auto parse_with_depths = new GraphEnhancedParseTree(*sentence);
    corpus->sentence_parses[sentence_index].reset(parse_with_depths);
    sentence->tree = parse_with_depths;

    // Actually tag and evaluate.
    vector<CausalityRelation> predicted = Tag(*sentence, parse_with_depths,
                                              gold_actions);
    if (pairwise) {
      FilterToPairwise(&predicted);
    }

    if (evaluate) {
      vector<CausalityRelation> gold = Decode(
          *sentence, gold_actions, options.new_conn_action,
          options.shift_action);
      if (pairwise) {
        FilterToPairwise(&gold);
      }
      evaluation += CausalityMetrics(
          gold, predicted, *corpus, *parse_with_depths,
          SpanTokenFilter {compare_punct, *sentence, corpus->pos_is_punct},
          missing_instances, missing_args, options.save_differences,
          options.log_differences, overlap_threshold);
    }

    if (write_results) {
      WriteSentenceResults(predicted, *sentence, &last_filename, &ann_out_file,
                           &current_txt_contents, &current_doc_span_ids,
                           &evt_id, &span_id);
    }
  }

  return evaluation;
}


vector<Parameters*> LSTMCausalityTagger::GetParameters() {
  vector<Parameters*> params = {
      p_sbias, p_L1toS, p_L2toS, p_L3toS, p_L4toS,
      p_s2a, p_abias,
      p_connective2S, p_cause2S, p_effect2S, p_means2S,
      p_p2t, p_v2t, p_tbias,
      p_L1_guard, p_L2_guard, p_L3_guard, p_L4_guard,
      p_connective_guard, p_cause_guard, p_effect_guard, p_means_guard};
  if (options.conn_in_state)
    params.push_back(p_curconn2S);
  if (options.word_dim > 0)
    params.push_back(p_w2t);
  if (options.gated_parse) {
    auto to_add = {p_parse_sel_bias, p_state_to_parse_sel, p_parse2sel,
                   p_full_state_bias, p_parse2pstate, p_state2pstate};
    params.insert(params.end(), to_add.begin(), to_add.end());
  }
  if (options.action_dim > 0) {
    params.push_back(p_actions2S);
    params.push_back(p_action_start);
  }
  if (options.subtrees)
    params.push_back(p_subtree2t);
  if (options.parse_path_hidden_dim > 0) {
    params.push_back(p_parse_path_start);
    params.push_back(p_parsepath2S);
    if (options.parse_path_arc_dim > 0) {
      params.push_back(p_pp_bias);
      params.push_back(p_parse2pp);
      params.push_back(p_token2pp);
    }
  }
  return params;
}


void LSTMCausalityTagger::InitializeNetworkParameters() {
  const unsigned action_size = vocab.CountActions() + 1;
  const unsigned vocab_size = vocab.CountWords() + 1;
  const unsigned pos_dim = parser.p_p->dim.size(0);
  const unsigned pretrained_dim = parser.pretrained.begin()->second.size();
  const unsigned parser_state_dim = parser.options.hidden_dim;

  assert(!parser.pretrained.empty());
  assert(parser.options.use_pos);

  // Parameters for token representation
  if (options.word_dim > 0) {
    p_w = model->add_lookup_parameters(vocab_size, {options.word_dim});
    p_w2t = model->add_parameters({options.token_dim, options.word_dim});
  }
  p_tbias = model->add_parameters({options.token_dim});
  p_v2t = model->add_parameters({options.token_dim, pretrained_dim});
  p_p2t = model->add_parameters({options.token_dim, pos_dim});
  if (options.subtrees) {
    // Parameters for incorporating parse info into token representation
    p_subtree2t = model->add_parameters({options.token_dim, parser_state_dim});
  }

  // Parameters for overall state representation
  p_sbias = model->add_parameters({options.state_dim});
  p_L1toS = model->add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L2toS = model->add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L3toS = model->add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L4toS = model->add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  if (options.conn_in_state)
    p_curconn2S = model->add_parameters({options.state_dim, options.token_dim});
  if (options.action_dim > 0) {
    p_actions2S = model->add_parameters(
        {options.state_dim, options.actions_hidden_dim});
    p_a = model->add_lookup_parameters(action_size, {options.action_dim});
  }
  p_connective2S = model->add_parameters(
      {options.state_dim, options.span_hidden_dim});
  p_cause2S = model->add_parameters(
      {options.state_dim, options.span_hidden_dim});
  p_effect2S = model->add_parameters(
      {options.state_dim, options.span_hidden_dim});
  p_means2S = model->add_parameters(
      {options.state_dim, options.span_hidden_dim});
  if (options.gated_parse) {
    // Parameters for incorporating parse info into state
    p_parse_sel_bias = model->add_parameters({parser_state_dim});
    p_state_to_parse_sel = model->add_parameters(
        {parser_state_dim, options.state_dim});
    p_parse2sel = model->add_parameters(
        {parser_state_dim, parser_state_dim});
    p_full_state_bias = model->add_parameters({options.state_dim});
    p_parse2pstate = model->add_parameters(
        {options.state_dim, parser_state_dim});
    p_state2pstate = model->add_parameters(
        {options.state_dim, options.state_dim});
  }

  // Parameters for turning states into actions
  p_abias = model->add_parameters({action_size});
  p_s2a = model->add_parameters({action_size, options.state_dim});

  // Parse path parameters
  if (options.parse_path_hidden_dim > 0) {
    const unsigned directed_rel_dim = parser.options.rel_dim + 1;
    p_parsepath2S = model->add_parameters(
        {options.state_dim, options.parse_path_hidden_dim});
    if (options.parse_path_arc_dim > 0) {
      p_pp_bias = model->add_parameters({options.parse_path_arc_dim});
      p_parse2pp = model->add_parameters(
          {options.parse_path_arc_dim, directed_rel_dim});
      p_token2pp = model->add_parameters(
          {options.parse_path_arc_dim, options.token_dim});
      p_parse_path_start = model->add_parameters({options.parse_path_arc_dim});
    } else {
      p_parse_path_start = model->add_parameters({directed_rel_dim});
    }
  }

  // Parameters for guard/start items in empty lists
  if (options.action_dim > 0)
    p_action_start = model->add_parameters({options.action_dim});
  p_L1_guard = model->add_parameters({options.token_dim});
  p_L2_guard = model->add_parameters({options.token_dim});
  p_L3_guard = model->add_parameters({options.token_dim});
  p_L4_guard = model->add_parameters({options.token_dim});
  p_connective_guard = model->add_parameters({options.token_dim});
  p_cause_guard = model->add_parameters({options.token_dim});
  p_effect_guard = model->add_parameters({options.token_dim});
  p_means_guard = model->add_parameters({options.token_dim});
}


LSTMCausalityTagger::TaggerState* LSTMCausalityTagger::InitializeParserState(
    ComputationGraph* cg,
    const Sentence& raw_sent,
    const Sentence::SentenceMap& sentence,  // w/ OOVs replaced
    const vector<unsigned>& correct_actions) {
  CausalityTaggerState* state = new CausalityTaggerState(raw_sent, sentence);

  for (reference_wrapper<LSTMBuilder>& builder : sentence_lstms) {
    builder.get().new_graph(*cg);
  }
  if (options.parse_path_hidden_dim > 0) {
    parse_path_lstm.new_graph(*cg);
  }

  // Non-persistent sentence LSTMs get sequences started in StartNewRelation.
  for (reference_wrapper<LSTMBuilder>& builder : persistent_lstms) {
    builder.get().start_new_sequence();
  }

  // Make sure there's a null relation available to embed.
  StartNewRelation();

  // Initialize the sentence-level LSTMs. All but L4 should start out empty.
  if (options.action_dim > 0)
    action_history_lstm.add_input(GetParamExpr(p_action_start));

  L1_lstm.add_input(GetParamExpr(p_L1_guard));
  state->L1.push_back(GetParamExpr(p_L1_guard));
  state->L1i.push_back(-1);
  L2_lstm.add_input(GetParamExpr(p_L2_guard));
  state->L2.push_back(GetParamExpr(p_L2_guard));
  state->L2i.push_back(-1);
  L3_lstm.add_input(GetParamExpr(p_L3_guard));
  state->L3.push_back(GetParamExpr(p_L3_guard));
  state->L3i.push_back(-1);

  state->L4.push_back(GetParamExpr(p_L4_guard));
  state->L4i.push_back(-1);
  L4_lstm.add_input(GetParamExpr(p_L4_guard));
  // Add words to L4 in reversed order: other than the initial guard entry,
  // the back of the L4 vector is conceptually the front, contiguous with the
  // back of L3.
  // TODO: do we need to do anything special to handle OOV words?
  auto reversed_sentence = sentence | boost::adaptors::reversed;
  reversed_sentence.advance_begin(1);  // Skip ROOT
  for (const auto& index_and_word_id : reversed_sentence) {
    const unsigned token_index = index_and_word_id.first;
    const unsigned word_id = index_and_word_id.second;
    const unsigned pos_id = raw_sent.poses.at(token_index);
    Expression token_repr = GetTokenEmbedding(cg, token_index, word_id, pos_id);

    state->L4.push_back(token_repr);
    state->L4i.push_back(token_index);
    L4_lstm.add_input(token_repr);
    state->all_tokens[token_index] = token_repr;

    // Parse paths may need their own set of subtree-less token embeddings.
    if (options.subtrees && options.parse_path_arc_dim > 0) {
      Expression token_repr_no_subtree = GetTokenEmbedding(
          cg, token_index, word_id, pos_id, true);
      state->all_subtreeless_tokens[token_index] = token_repr_no_subtree;
    }
  }

  state->current_conn_token_i = state->L4i.back();
  state->current_arg_token_i = state->L4i.back();
  state->current_conn_token = state->all_tokens[state->current_conn_token_i];
  state->current_arg_token = state->all_tokens[state->current_arg_token_i];

  if (options.oracle_connectives) {
    const auto oracle_rels = GetDecodedGoldRelations(raw_sent, correct_actions);
    for (const CausalityRelation rel : oracle_rels) {
      const auto connective = rel.GetConnectiveIndices();
      assert(is_sorted(connective.begin(), connective.end()));
      state->oracle_connectives[connective.at(0)].push_back(vector<unsigned>(
          connective.begin() + 1, connective.end()));
    }

    auto conn_frag_action_iter = find(vocab.action_names.begin(),
                                      vocab.action_names.end(),
                                      "CONN-FRAG-RIGHT");
    assert(conn_frag_action_iter != vocab.action_names.end());
    conn_frag_action = conn_frag_action_iter - vocab.action_names.begin();
    auto split_action_iter = find(vocab.action_names.begin(),
                                  vocab.action_names.end(), "SPLIT");
    assert(split_action_iter != vocab.action_names.end());
    split_action = split_action_iter - vocab.action_names.begin();

    oracle_actions_taken = 0;
  }

  return state;
}


Expression LSTMCausalityTagger::GetTokenEmbedding(ComputationGraph* cg,
                                                  unsigned word_index,
                                                  unsigned word_id,
                                                  unsigned pos_id,
                                                  bool no_subtrees) {
  unsigned pretrained_id =
      parser.pretrained.count(word_id) ? word_id : parser.GetVocab()->kUNK;
  Expression pretrained = const_lookup(*cg, parser.p_t, pretrained_id);
  Expression pos = const_lookup(*cg, parser.p_p, pos_id);
  // TODO: add in the token index directly as an input?
  vector<Expression> args = {
      GetParamExpr(p_tbias),
      GetParamExpr(p_p2t), pos,
      GetParamExpr(p_v2t), pretrained};
  if (options.word_dim > 0) {
    Expression word = lookup(*cg, p_w, word_id);
    args.push_back(GetParamExpr(p_w2t));
    args.push_back(word);
  }
  if (options.subtrees && !no_subtrees) {
    Expression subtree_repr = nobackprop(
        parser_states.at(to_string(word_index)));
    args.push_back(GetParamExpr(p_subtree2t));
    args.push_back(subtree_repr);
  }
  Expression token_repr = RectifyWithDropout(affine_transform(args));
  return token_repr;
}


// TODO: forbid punctuation tokens as part of connectives?
bool LSTMCausalityTagger::IsActionForbidden(const unsigned action,
                                            TaggerState* state) const {
  const CausalityTaggerState& real_state =
      static_cast<const CausalityTaggerState&>(*state);
  const string& action_name = vocab.action_names[action];
  bool next_arg_token_is_left = real_state.L1.size() > 1;

  if (real_state.currently_processing_rel) {
    // SHIFT is mandatory if it's available and no tokens are left to compare.
    if (options.shift_action
        && real_state.L1.size() <= 1 && real_state.L4.size() <= 1) {
      return action_name[0] != 'S' || action_name[1] != 'H';
    }

    // Check for unknown connective word set.
    if (!in_training && options.known_conns_only
        && action_name[0] == 'C' && action_name[5] == 'F') {  // CONN-FRAG
      unsigned word_id = real_state.sentence.at(real_state.current_arg_token_i);
      if (word_id != vocab.kUNK) {  // UNK is always OK in a connective
        set<unsigned> conn_words({word_id});
        for (unsigned token_index : real_state.current_rel_conn_tokens) {
          conn_words.insert(real_state.sentence.at(token_index));
        }
        if (!known_connectives.count(conn_words)) {
          return true;
        }
      }
    }

    if (next_arg_token_is_left) {
      // SHIFT, CONN-FRAG, and SPLIT are all forbidden when working to the left,
      // as are rightward-oriented operations.
      return !starts_with(action_name, "LEFT")
          && !ends_with(action_name, "LEFT");
    } else {  // processing right side
      // SHIFT is forbidden if some tokens have not yet been compared.
      if (action_name[0] == 'S' && action_name[1] == 'H') {
        assert(options.shift_action);
        return real_state.L1.size() > 1 || real_state.L4.size() > 1;
      }

      // We should never end up processing a relation after 0 actions.
      assert(real_state.prev_action != UNSIGNED_NEG_1);
      // If there was a last action, check that this one is compatible.
      const string& last_action_name =
          vocab.action_names[real_state.prev_action];
      // SPLIT has unique requirements: can't come after a SPLIT or a
      // CONN-FRAG. Also, we can't have less than two connective words.
      if (action_name[0] == 'S' && action_name[1] == 'P') {
        return real_state.current_rel_conn_tokens.size() < 2
            || starts_with(last_action_name, "SP")
            || last_action_name[0] == 'C';
      }

      // Another special case: forbid CONN-FRAG after SPLIT or CONN-FRAG.
      // Also forbid making the primary connective word (or any word before it)
      // a connective fragment.
      if (action_name[0] == 'C') {
        if (starts_with(last_action_name, "SP") || last_action_name[0] == 'C'
            || (real_state.current_arg_token_i <=
                real_state.current_conn_token_i))
          return true;
      }

      // If it's not a split, a shift, or a forbidden post-split operation,
      // forbid any operation that doesn't act on arg tokens to the right.
      return !starts_with(action_name, "RIGHT")
          && !ends_with(action_name, "RIGHT");
    }
  } else {  // not processing a relation
    if (action_name[0] == 'N' && action_name[3] == 'C') {
      // NO-CONN never forbidden if we're not processing a relation
      return false;
    } else if (action_name[0] == 'N' && action_name[4] == 'C') {  // NEW-CONN
      // Check for forbidden connective word, if needed.
      if (!in_training && options.known_conns_only) {
        unsigned word_id =
            real_state.sentence.at(real_state.current_conn_token_i);
        // Unk is never forbidden. If not UNK, forbidden if it's an unseen word.
        return word_id != vocab.kUNK && !known_connectives.count({word_id});
      } else {         // If we're not checking for forbidden connectives,
        return false;  // NEW-CONN is always OK (i.e., not forbidden).
      }
    }

    if (options.new_conn_action) {
      // If new connectives are separate actions, any relation-dependent action
      // is forbidden if we're not processing a relation yet.
      return true;
    } else {
      // Don't allow starting a relation with a CONN-FRAG. That would mean
      // either that we're adding a left fragment or that we're adding the
      // primary connective word itself as a fragment.
      if (action_name[0] == 'C') {
          return true;
      }

      // Even if we're not processing a relation, we still need to check whether
      // we have any words to compare on the left. If we do, only leftward
      // relation-starting operations are allowed; if not, only rightward ones.
      if (next_arg_token_is_left) {
        return action_name[0] != 'L'              // LEFT-ARC
            && !ends_with(action_name, "FT");     // NO-ARC-LEFT
      } else {
        return action_name[0] != 'R'              // RIGHT-ARC
            && !ends_with(action_name, "HT");     // NO-ARC-RIGHT
      }
    }

    return true;
  }
}


Expression LSTMCausalityTagger::GetParsePathEmbedding(
    CausalityTaggerState* state, unsigned source_token_id,
    unsigned dest_token_id) {
  assert(options.parse_path_hidden_dim > 0);

  parse_path_lstm.start_new_sequence();
  parse_path_lstm.add_input(GetParamExpr(p_parse_path_start));

  if (source_token_id != UNSIGNED_NEG_1 && dest_token_id != UNSIGNED_NEG_1) {
    const GraphEnhancedParseTree* tree = static_cast<GraphEnhancedParseTree*>(
        state->raw_sentence.tree);
    auto parse_path = tree->GetParsePath(source_token_id, dest_token_id);

    for (const GraphEnhancedParseTree::ParsePathLink& arc : parse_path) {
      const vector<string>& action_names = parser.GetVocab()->action_names;
      auto GetParseActionIter = [&arc, &action_names](bool is_left) {
        string action_name = (is_left ? "LEFT-ARC(" : "RIGHT-ARC(")
            + arc.arc_label + ")";
        return find(action_names.begin(), action_names.end(), action_name);
      };

      // Default to LEFT-ARC for parse relation unless it doesn't exist.
      auto action_iter = GetParseActionIter(true);
      if (action_iter == action_names.end()) {
        action_iter = GetParseActionIter(false);
      }
      assert(action_iter != action_names.end());

      int action_id = action_iter - action_names.begin();
      ComputationGraph* cg = state->current_arg_token.pg;
      Expression relation = const_lookup(*cg, parser.p_r, action_id);
      Expression is_back_edge = zeroes(*cg, {1}) + static_cast<cnn::real>(
          arc.reversed);
      Expression directed_rel = concatenate({relation, is_back_edge});

      if (options.parse_path_arc_dim > 0) {
        Expression token_embedding =
            options.subtrees ?
                state->all_subtreeless_tokens.at(arc.end) :
                state->all_tokens.at(arc.end);
        parse_path_lstm.add_input(RectifyWithDropout(affine_transform(
            {GetParamExpr(p_pp_bias), GetParamExpr(p_parse2pp), directed_rel,
             GetParamExpr(p_token2pp), token_embedding})));
      } else {
        parse_path_lstm.add_input(directed_rel);
      }
    }
  }

  return parse_path_lstm.back();
}


bool LSTMCausalityTagger::ShouldUseOracleTransition(
    CausalityTaggerState* state) {
  if (options.oracle_connectives) {
    auto oracle_conn_iter = state->oracle_connectives.find(
        state->current_conn_token_i);
    if (oracle_conn_iter != state->oracle_connectives.end()) {
      if (!state->currently_processing_rel) {
        // The oracle says we ought to start a new instance here, so follow it.
        // cerr << "Should use oracle action NEW-CONN" << endl;
        return true;
      } else {  // we're in the middle of an instance; check for CONN-FRAG/SPLIT
        if (state->current_arg_token_i != state->current_conn_token_i
            // CONN-FRAG and SPLIT doesn't advance the argument token, so we
            // have to make sure we don't repeatedly use the oracle transition.
            && state->prev_action != conn_frag_action
            && state->prev_action != split_action) {
          unsigned next_split = state->splits_seen_for_conn + 1;
          const auto co_starting_conns = oracle_conn_iter->second;

          // Use splits_seen_for_conn to check which instance for this
          // connective anchor we're in.
          if (Contains(co_starting_conns.at(state->splits_seen_for_conn),
                       state->current_arg_token_i)) {
            // cerr << "Should use oracle action CONN-FRAG-RIGHT" << endl;
            return true;
          } else if (  // If there is a next split...
              co_starting_conns.size() > next_split
              // ...and its fragment list starts with our current arg token...
              && co_starting_conns.at(next_split).at(0)
                  == state->current_arg_token_i) {
            // cerr << "Should use oracle action SPLIT" << endl;
            return true;
          }
        }
      }
    }
  }

  return false;
}


Expression LSTMCausalityTagger::GetActionProbabilities(TaggerState* state) {
  CausalityTaggerState* real_state = static_cast<CausalityTaggerState*>(state);
  if (ShouldUseOracleTransition(real_state)) {
    return NeuralTransitionTagger::USE_ORACLE;
  }

  // sbias + ?(actions2S * actions_lstm) + (\sum_i rel_cmpt_i2S * rel_cmpt_i)
  //       + ?(current2S * current_token) + (\sum_i LToS_i * L_i)
  //       + ?(pp2S * parse_path)
  vector<Expression> state_args = {GetParamExpr(p_sbias),
      GetParamExpr(p_connective2S), connective_lstm.back(),
      GetParamExpr(p_cause2S), cause_lstm.back(),
      GetParamExpr(p_effect2S), effect_lstm.back(),
      GetParamExpr(p_means2S), means_lstm.back(),
      GetParamExpr(p_L1toS), L1_lstm.back(),
      GetParamExpr(p_L2toS), L2_lstm.back(),
      GetParamExpr(p_L3toS), L3_lstm.back(),
      GetParamExpr(p_L4toS), L4_lstm.back()};
  if (options.conn_in_state) {
    state_args.push_back(GetParamExpr(p_curconn2S));
    state_args.push_back(real_state->current_conn_token);
  }
  if (options.action_dim > 0) {
    state_args.push_back(GetParamExpr(p_actions2S));
    state_args.push_back(action_history_lstm.back());
  }
  if (options.parse_path_hidden_dim > 0) {
    state_args.push_back(GetParamExpr(p_parsepath2S));
    if (real_state->currently_processing_rel) {
      state_args.push_back(
          GetParsePathEmbedding(real_state, real_state->current_conn_token_i,
                                real_state->current_arg_token_i));
    } else {
      state_args.push_back(
          GetParsePathEmbedding(real_state, UNSIGNED_NEG_1, UNSIGNED_NEG_1));
    }
  }
  Expression state_repr = RectifyWithDropout(affine_transform(state_args));

  Expression full_state_repr;
  if (options.gated_parse) {
    // Mix in parse information from full parse tree.
    Expression parser_tree_embedding = nobackprop(parser_states.at("Tree"));
    Expression parse_state_selections = logistic(affine_transform(
        {GetParamExpr(p_parse_sel_bias),
         GetParamExpr(p_state_to_parse_sel), state_repr,
         GetParamExpr(p_parse2sel), parser_tree_embedding}));
    Expression selected_parse_repr = cwise_multiply(parse_state_selections,
                                                    parser_tree_embedding);
    full_state_repr = RectifyWithDropout(affine_transform(
        {GetParamExpr(p_full_state_bias),
         GetParamExpr(p_parse2pstate), selected_parse_repr,
         GetParamExpr(p_state2pstate), state_repr}));
  } else {
    full_state_repr = state_repr;
  }

  // abias + s2a * full_state_repr
  Expression p_a = affine_transform({GetParamExpr(p_abias), GetParamExpr(p_s2a),
                                     full_state_repr});
  return p_a;
}


void LSTMCausalityTagger::StartNewRelation() {
  connective_lstm.start_new_sequence();
  cause_lstm.start_new_sequence();
  effect_lstm.start_new_sequence();
  means_lstm.start_new_sequence();

  connective_lstm.add_input(GetParamExpr(p_connective_guard));
  cause_lstm.add_input(GetParamExpr(p_cause_guard));
  effect_lstm.add_input(GetParamExpr(p_effect_guard));
  means_lstm.add_input(GetParamExpr(p_means_guard));
}


// Macros are ugly, but this is easier than trying to make sure we remember to
// do all steps for each push/pop. Makes the code below much cleaner.
// TODO: switch to map-based algorithm where we only maintain indices, not
// Expressions?
#define DO_LIST_PUSH(list_name, var_name) \
    list_name.push_back(var_name); \
    list_name##i.push_back(var_name##_i); \
    list_name##_lstm.add_input(var_name);
#define DO_LIST_POP(list_name) \
    list_name.pop_back(); \
    list_name##i.pop_back(); \
    list_name##_lstm.rewind_one_step();
#define SET_LIST_BASED_VARS(var_name, list, expr) \
    /* For the LSTM, don't apply expr to list -- if expr is back(), for example,
       it'll use the encoding of the whole list, not the last element. Instead,
       use the token index from list##i to look up the relevant Expression. */ \
    var_name##_i = list##i.expr; \
    var_name = cst->all_tokens[var_name##_i];
// TODO: redefine to use std::move?
#define MOVE_LIST_ITEM(from_list, to_list, tmp_var) \
    SET_LIST_BASED_VARS(tmp_var, from_list, back()); \
    DO_LIST_PUSH(to_list, tmp_var); \
    DO_LIST_POP(from_list);

void LSTMCausalityTagger::DoAction(unsigned action, TaggerState* state,
                                   ComputationGraph* cg,
                                   map<string, Expression>* states_to_expose) {
  CausalityTaggerState* cst = static_cast<CausalityTaggerState*>(state);
  const string& action_name = vocab.action_names[action];
  if (options.action_dim > 0) {
    Expression action_e = lookup(*cg, p_a, action);
    action_history_lstm.add_input(action_e);
  }

  // Alias key state variables for ease of reference
  auto& L1 = cst->L1;
  auto& L1i = cst->L1i;
  auto& L2 = cst->L2;
  auto& L2i = cst->L2i;
  auto& L3 = cst->L3;
  auto& L3i = cst->L3i;
  auto& L4 = cst->L4;
  auto& L4i = cst->L4i;
  auto& current_conn_token = cst->current_conn_token;
  auto& current_conn_token_i = cst->current_conn_token_i;
  auto& current_arg_token = cst->current_arg_token;
  auto& current_arg_token_i = cst->current_arg_token_i;

  assert(L1.size() == L1i.size() && L2.size() == L2i.size() &&
         L3.size() == L3i.size() && L4.size() == L4i.size());
  assert(!L1i.empty() && !L2i.empty() && !L3i.empty() && !L4i.empty());

  /*
  cerr << "Performing action " << action_name << " on connective word \""
       << vocab.int_to_words.at(
           cst->raw_sentence.words.at(current_conn_token_i)) << '"';
  if (current_arg_token_i != UNSIGNED_NEG_1) {
    cerr << " and argument word \"" << vocab.int_to_words.at(
             cst->raw_sentence.words.at(current_arg_token_i)) << '"';
  }
  cerr << endl;
  //*/

  Expression to_push;  // dummy variables for use below
  unsigned to_push_i;

  auto AdvanceArgTokenLeft = [&]() {
    assert(L1.size() > 1);
    MOVE_LIST_ITEM(L1, L2, to_push);
    if (L1.size() > 1) {
      SET_LIST_BASED_VARS(current_arg_token, L1, back());
    } else {
      SET_LIST_BASED_VARS(current_arg_token, L4, back());
    }
  };

  auto AdvanceArgTokenRight = [&]() {
    assert(L4.size() > 1);
    MOVE_LIST_ITEM(L4, L3, to_push);
    SET_LIST_BASED_VARS(current_arg_token, L4, back());
  };

  auto SetNewConnectiveToken = [&]() {
    if (L4.size() > 1) {
      SET_LIST_BASED_VARS(current_conn_token, L4, back());
      // We're recording the fact that the connective token has changed. That
      // means we have at least one thing in L1: the previous connective token.
      // Set the current arg token to the first available item in L1, excluding
      // the guard.
      SET_LIST_BASED_VARS(current_arg_token, L1, back());
    }
  };

  auto EnsureRelationWithConnective = [&](bool from_new_conn = false) {
    if (options.new_conn_action && !from_new_conn) {
      assert(cst->currently_processing_rel);
    }
    if (!cst->currently_processing_rel) {
      // Don't start a new relation here...we should already have a blank one,
      // either from initialization or from a previous end-of-relation.
      connective_lstm.add_input(current_conn_token);
      cst->current_rel_conn_tokens.push_back(current_conn_token_i);
      cst->currently_processing_rel = true;
    }
  };

  auto CompleteRelation = [&]() {
    // Move L2 back to L1.
    while (L2.size() > 1) {
      MOVE_LIST_ITEM(L2, L1, to_push);
    }
    // Move L3 back to L4 -- except the last item, which we'll move to L1.
    // (There has to be at least one item on L3.)
    while (L3.size() > 2) {
      MOVE_LIST_ITEM(L3, L4, to_push);
    }
    assert(L3.size() == 2);
    MOVE_LIST_ITEM(L3, L1, to_push);
    SetNewConnectiveToken();

    cst->currently_processing_rel = false;
    cst->current_rel_conn_tokens.clear();
    cst->current_rel_cause_tokens.clear();
    cst->current_rel_effect_tokens.clear();
    cst->current_rel_means_tokens.clear();
    cst->splits_seen_for_conn = 0;
    StartNewRelation();  // relation LSTMs should be empty until next relation
  };

  auto AddArc = [&](unsigned action) {
    EnsureRelationWithConnective();
    const string& arc_type = vocab.actions_to_arc_labels[action];
    vector<string> arg_names;
    boost::split(arg_names, arc_type, boost::is_any_of(","));
    for (const string& arg_name : arg_names) {
      LSTMBuilder* arg_builder;
      vector<unsigned>* arg_list;
      if (arg_name == "Cause") {
        arg_builder = &cause_lstm;
        arg_list = &cst->current_rel_cause_tokens;
      } else if (arg_name == "Effect") {
        arg_builder = &effect_lstm;
        arg_list = &cst->current_rel_effect_tokens;
      } else {  // arg_name == "Means"
        arg_builder = &means_lstm;
        arg_list = &cst->current_rel_means_tokens;
      }
      arg_builder->add_input(cst->all_tokens.at(current_arg_token_i));
      arg_list->push_back(current_arg_token_i);
    }
  };

  if (action_name == "NO-CONN") {
    assert(L4.size() > 1);  // L4 should have at least duplicate of current conn
    DO_LIST_PUSH(L1, current_conn_token);
    DO_LIST_POP(L4);  // remove duplicate of current_conn_token
    SetNewConnectiveToken();
  } else if (action_name == "NEW-CONN") {
    assert(options.new_conn_action && !cst->currently_processing_rel);
    EnsureRelationWithConnective(true);
  } else if (action_name == "NO-ARC-LEFT") {
    EnsureRelationWithConnective();
    AdvanceArgTokenLeft();
  } else if (action_name == "NO-ARC-RIGHT") {
    EnsureRelationWithConnective();
    AdvanceArgTokenRight();
  } else if (action_name == "CONN-FRAG-RIGHT"  // LEFT is not currently possible
             /* || action_name == "CONN-FRAG-LEFT"*/) {
    assert(cst->currently_processing_rel);
    assert(cst->current_arg_token_i != cst->current_conn_token_i);
    cst->current_rel_conn_tokens.push_back(current_arg_token_i);
    connective_lstm.add_input(current_arg_token);
    // Do NOT advance the argument token. It could still be part of an arg.
  } else if (starts_with(action_name, "RIGHT-ARC")) {
    AddArc(action);
    AdvanceArgTokenRight();
  } else if (starts_with(action_name, "LEFT-ARC")) {
    AddArc(action);
    AdvanceArgTokenLeft();
  } else if (action_name == "SPLIT") {
    assert(cst->current_rel_conn_tokens.size() > 1
           && cst->currently_processing_rel);
    // Find the last connective word that shares the same lemma, if possible.
    // Default is cut off after initial connective word.
    unsigned connective_repeated_token_index =
        cst->current_rel_conn_tokens.at(1);
    for (auto token_iter = cst->current_rel_conn_tokens.rbegin();
        // Ignore the connective guard.
        token_iter != cst->current_rel_conn_tokens.rend() - 1; ++token_iter) {
      if (cst->sentence.at(*token_iter)
          == cst->sentence.at(current_arg_token_i)) {
        connective_repeated_token_index = *token_iter;
        break;
      }
    }

    // Now copy the current relation over to a new one, keeping only the
    // tokens up to the repeated connective token.
    StartNewRelation();  // reset all the relation-specific LSTMs
    const vector<LSTMBuilder*> builders(
        {&connective_lstm, &cause_lstm, &effect_lstm, &means_lstm});
    const vector<vector<unsigned>*> token_lists(
        {&cst->current_rel_conn_tokens, &cst->current_rel_cause_tokens,
         &cst->current_rel_effect_tokens, &cst->current_rel_means_tokens});
    for (unsigned i = 0; i < builders.size(); ++i) {
      vector<unsigned> new_token_list;
      new_token_list.reserve(token_lists[i]->size());  // can't get any bigger
      for (unsigned token_index : *token_lists[i]) {
        if (token_index < connective_repeated_token_index) {
          builders[i]->add_input(cst->all_tokens.at(token_index));
          new_token_list.push_back(token_index);
        }
      }
      *token_lists[i] = move(new_token_list);
    }

    // Add the repeated token to the relation -- but don't advance the arg
    // token; in theory, this could still be an argument word.
    connective_lstm.add_input(current_arg_token);
    cst->current_rel_conn_tokens.push_back(current_arg_token_i);
    cst->splits_seen_for_conn++;
  } else if (action_name == "SHIFT") {
    assert(options.shift_action);
    assert(L4.size() == 1 && L1.size() == 1);  // processed all tokens?
    CompleteRelation();
  }

  // Don't wait for a SHIFT to complete the relation if we're not using SHIFTs.
  if (!options.shift_action && L4.size() == 1 && L1.size() == 1) {
    CompleteRelation();
  }

  cst->prev_action = action;
  assert(!L1i.empty() && !L2i.empty() && !L3i.empty() && !L4i.empty());
  assert(L1.size() == L1i.size() && L2.size() == L2i.size() &&
         L3.size() == L3i.size() && L4.size() == L4i.size());
}


void LSTMCausalityTagger::RecordKnownConnectives(
    const vector<CausalityRelation>& rels) {
  for (const auto& rel : rels) {
    const auto& conn_indices = rel.GetConnectiveIndices();
    // Record all subsets of the connective words.
    unsigned num_subsets = 1 << conn_indices.size();  // 2^n, 1 bit per word
    for (unsigned i = 1; i < num_subsets; ++i) {  // don't process empty subset
      boost::dynamic_bitset<> conn_subset_mask(conn_indices.size(), i);
      set<unsigned> word_ids;
      for (unsigned index_into_conn = 0;
          index_into_conn < conn_subset_mask.size(); ++index_into_conn) {
        if (conn_subset_mask.test(index_into_conn)) {
          unsigned conn_token_index = conn_indices[index_into_conn];
          word_ids.insert(rel.GetSentence().words.at(conn_token_index));
        }
      }
      known_connectives.insert(word_ids);
    }
  }
}


const vector<CausalityRelation>& LSTMCausalityTagger::GetDecodedGoldRelations(
    const Sentence& sentence, const vector<unsigned>& actions) {
  auto decoded_iter = training_decoded_cache.find(&sentence);
  if (decoded_iter == training_decoded_cache.end()) {
    vector<CausalityRelation> decoded = Decode(
        sentence, actions, options.new_conn_action, options.shift_action);
    if (options.train_pairwise) {
      FilterToPairwise(&decoded);
    }
    RecordKnownConnectives(decoded);
    auto insert_result =
        training_decoded_cache.insert({&sentence, std::move(decoded)});
    return insert_result.first->second;
  } else {
    return decoded_iter->second;
  }
}
