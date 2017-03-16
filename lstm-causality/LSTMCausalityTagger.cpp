#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/combine.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
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


LSTMCausalityTagger::LSTMCausalityTagger(const string& parser_model_path,
                                         const TaggerOptions& options)
    : options(options), parser(parser_model_path),
      L1_lstm(options.lstm_layers, options.token_dim,
              options.lambda_hidden_dim, &model),
      L2_lstm(options.lstm_layers, options.token_dim,
              options.lambda_hidden_dim, &model),
      L3_lstm(options.lstm_layers, options.token_dim,
              options.lambda_hidden_dim, &model),
      L4_lstm(options.lstm_layers, options.token_dim,
              options.lambda_hidden_dim, &model),
      action_history_lstm(options.lstm_layers, options.action_dim,
                          options.actions_hidden_dim, &model),
      // Input to relations LSTM is an embedded relation, whose dimension is
      // already transformed from span_hidden_dim. (In theory, we could control
      // this input dim with another parameter, but it doesn't seem worth it.)
      relations_lstm(options.lstm_layers, options.rels_hidden_dim,
                     options.rels_hidden_dim, &model),
      connective_lstm(options.lstm_layers, options.token_dim,
                      options.span_hidden_dim, &model),
      cause_lstm(options.lstm_layers, options.token_dim,
              options.span_hidden_dim, &model),
      effect_lstm(options.lstm_layers, options.token_dim,
              options.span_hidden_dim, &model),
      means_lstm(options.lstm_layers, options.token_dim,
              options.span_hidden_dim, &model) {
  vocab = *parser.GetVocab();  // now that parser is initialized, copy vocab
  // Reset actions
  vocab.actions.clear();
  vocab.actions_to_arc_labels.clear();
  // We don't care about characters
  vocab.chars_to_int.clear();
  vocab.int_to_chars.clear();
}


void LSTMCausalityTagger::Train(const BecauseOracleTransitionCorpus& corpus,
                                vector<unsigned> selections, double dev_pct,
                                const string& model_fname,
                                double epochs_cutoff,
                                const volatile sig_atomic_t* requested_stop) {
  const unsigned num_sentences = selections.size();
  // selections gives us the subcorpus to use for training at all. But we'll
  // still keep shuffling each time we've gone through all the training
  // sentences, so we track the order imposed by that shuffle.
  vector<unsigned> sub_order(num_sentences);
  iota(sub_order.begin(), sub_order.end(), 0);
  const unsigned status_every_i_iterations = min(100, num_sentences);
  cerr << "NUMBER OF TRAINING SENTENCES: " << num_sentences << endl;
  time_t time_start = chrono::system_clock::to_time_t(
      chrono::system_clock::now());
  cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z")
       << endl;

  SimpleSGDTrainer sgd(&model);
  //MomentumSGDTrainer sgd(model);
  sgd.eta_decay = 0.08;
  //sgd.eta_decay = 0.05;

  double sentences_seen = 0;
  double best_f1 = -numeric_limits<double>::infinity();
  unsigned actions_seen = 0;
  double correct = 0;
  double llh = 0;
  double last_epoch_saved = nan("");

  unsigned num_sentences_dev = round(dev_pct * num_sentences);
  unsigned num_sentences_train = num_sentences - num_sentences_dev;

  unsigned sentence_i = num_sentences_train;
  for (unsigned iteration = 0; !requested_stop || !(*requested_stop);
       ++iteration) {
    for (unsigned iter_i = 0; iter_i < status_every_i_iterations; ++iter_i) {
      if (sentence_i == num_sentences_train) {
        sentence_i = 0;
        if (sentences_seen > 0) {
          sgd.update_epoch();
        }
        cerr << "**SHUFFLE\n";
        random_shuffle(sub_order.begin(), sub_order.end());
      }

      const Sentence& sentence =
          corpus.sentences[selections[sub_order[sentence_i]]];
      const vector<unsigned>& correct_actions =
          corpus.correct_act_sent[selections[sub_order[sentence_i]]];

      // cerr << "Starting sentence " << sentence << endl;

      ComputationGraph cg;
      Expression parser_state;
      parser.LogProbTagger(sentence, *parser.GetVocab(), &cg, true,
                           &parser_state);
      LogProbTagger(&cg, sentence, sentence.words, correct_actions,
                    corpus.vocab->actions, corpus.vocab->int_to_words,
                    &correct, &parser_state);
      double lp = as_scalar(cg.incremental_forward());
      if (lp < 0) {
        cerr << "Log prob " << lp << " < 0 on sentence "
             << corpus.sentences[selections[sub_order[sentence_i]]]<< endl;
        assert(lp >= 0.0);
      }
      cg.backward();
      sgd.update(1.0);

      llh += lp;
      ++sentence_i;
      actions_seen += correct_actions.size();
      sentences_seen += 1;
    }

    sgd.status();
    time_t time_now = chrono::system_clock::to_time_t(
        chrono::system_clock::now());
    double epoch = sentences_seen / num_sentences_train;
    cerr << "update #" << iteration << " (epoch " << epoch
         << " | time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "
         << llh << " ppl: " << exp(llh / actions_seen) << " err: "
         << (actions_seen - correct) / actions_seen << endl;
    // TODO: move declaration to make this unnecessary.
    llh = actions_seen = correct = 0;

    if (iteration % 25 == 0) {
      best_f1 = DoDevEvaluation(corpus, selections, &parser,
                                num_sentences_train, iteration, sentences_seen,
                                best_f1, model_fname, &last_epoch_saved);
      if (epoch - last_epoch_saved > epochs_cutoff) {
        cerr << "Reached cutoff for epochs with no increase in max dev F1: "
             << epoch - last_epoch_saved << " > " << epochs_cutoff
             << "; terminating training" << endl;
        break;
      }
    }
  }
}


double LSTMCausalityTagger::DoDevEvaluation(
    const TrainingCorpus& corpus, const vector<unsigned>& selections,
    LSTMParser* parser, unsigned num_sentences_train, unsigned iteration,
    unsigned sentences_seen, double best_f1, const string& model_fname,
    double* last_epoch_saved) {
  // report on dev set
  const unsigned num_sentences = selections.size();
  double llh_dev = 0;
  double num_actions = 0;
  double correct_dev = 0;
  CausalityMetrics evaluation;
  const auto t_start = chrono::high_resolution_clock::now();
  for (unsigned sii = num_sentences_train; sii < num_sentences; ++sii) {
    const Sentence& sentence = corpus.sentences[selections[sii]];

    ComputationGraph cg;
    Expression parser_state;
    parser->LogProbTagger(sentence, *parser->GetVocab(), &cg, true,
                          &parser_state);
    vector<unsigned> actions = LogProbTagger(sentence, *corpus.vocab, &cg,
                                             false);
    llh_dev += as_scalar(cg.incremental_forward());
    vector<CausalityRelation> predicted = Decode(sentence, actions);

    const vector<unsigned>& gold_actions =
        corpus.correct_act_sent[selections[sii]];
    vector<CausalityRelation> gold = Decode(sentence, gold_actions);

    num_actions += actions.size();
    evaluation += CausalityMetrics(gold, predicted);
  }

  auto t_end = chrono::high_resolution_clock::now();
  double epoch = sentences_seen / static_cast<double>(num_sentences_train);
  cerr << "  **dev (iter=" << iteration << " epoch=" << epoch
       << ")llh=" << llh_dev << " ppl: " << exp(llh_dev / num_actions)
       << "\terr: " << (num_actions - correct_dev) / num_actions
       << " evaluation: \n" << evaluation
       << "\n[" << num_sentences - num_sentences_train << " sentences in "
       << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
       << endl;

  // TODO: should we take into account action-level error, too?
  if (evaluation.connective_metrics->GetF1() > best_f1) {
    best_f1 = evaluation.connective_metrics->GetF1();
    SaveModel(model_fname, !isnan(*last_epoch_saved));
    *last_epoch_saved = epoch;
  }

  return best_f1;
}


vector<CausalityRelation> LSTMCausalityTagger::Decode(
    const Sentence& sentence, const vector<unsigned> actions) {
  // cerr << "Decoding sentence " << sentence << endl;

  vector<CausalityRelation> relations;
  vector<unsigned> lambda_1;
  deque<unsigned> lambda_2; // we'll be appending to the left end
  vector<unsigned> lambda_3;
  // We'll be moving stuff off of the left end of lambda_4.
  deque<unsigned> lambda_4;
  for (auto& token_index_and_id : sentence.words) {
    lambda_4.push_back(token_index_and_id.first);
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

  auto EnsureCurrentRelation = [&]() {
    if (!current_rel) {
      relations.emplace_back(
          sentence, vocab, CausalityRelation::CONSEQUENCE,
          IndexList({current_conn_token}));
      current_rel = &relations.back();
    }
  };

  auto AddArc = [&](unsigned action) {
    EnsureCurrentRelation();
    const string& arc_type = vocab.actions_to_arc_labels[action];
    auto arg_iter = find(CausalityRelation::ARG_NAMES.begin(),
        CausalityRelation::ARG_NAMES.end(), arc_type);
    assert(arg_iter != CausalityRelation::ARG_NAMES.end());
    current_rel->AddToArgument(
        arg_iter - CausalityRelation::ARG_NAMES.begin(), current_arg_token);
  };

  for (auto iter = actions.begin(), end = actions.end(); iter != end; ++iter) {
    unsigned action = *iter;
    const string& action_name = vocab.actions[action];
    /*
    cerr << "Decoding action " << action_name << " on connective word \""
         << vocab.int_to_words.at(sentence.words.at(current_conn_token))
         << "\" and argument word \"" << vocab.int_to_words.at(
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
        // If we have no more tokens to pull in, we'd better be on the last
        // action.
        assert(iter + 1 == end);
      }
    } else if (action_name == "NO-ARC-LEFT") {
      EnsureCurrentRelation();
      AdvanceArgTokenLeft();
    } else if (action_name == "NO-ARC-RIGHT") {
      EnsureCurrentRelation();
      AdvanceArgTokenRight();
    } else if (action_name == "CONN-FRAG-LEFT") {
      EnsureCurrentRelation();
      current_rel->AddConnectiveToken(current_arg_token);
      AdvanceArgTokenLeft();
    } else if (action_name == "CONN-FRAG-RIGHT") {
      EnsureCurrentRelation();
      current_rel->AddConnectiveToken(current_arg_token);
      AdvanceArgTokenRight();
    } else if (starts_with(action_name, "RIGHT-ARC")) {
      AddArc(action);
      AdvanceArgTokenRight();
    } else if (::starts_with(action_name, "LEFT-ARC")) {
      AddArc(action);
      AdvanceArgTokenLeft();
    } else if (action_name == "SPLIT") {
      assert(current_rel && current_rel->GetConnectiveIndices()->size() > 1);
      // Make a copy of the current relation.
      relations.push_back(*current_rel);
      current_rel = &relations.back();
      // Find the last connective word that shares the same lemma, if possible.
      // Default is cut off after initial connective word.
      unsigned connective_repeated_token_index = 1;
      IndexList* connective_indices = current_rel->GetConnectiveIndices();
      for (size_t i = connective_indices->size() - 1; i > 1; --i) {
        if (sentence.words.at((*connective_indices)[i])
            == sentence.words.at(current_arg_token)) {
          connective_repeated_token_index = i;
          break;
        }
      }
      /*
      cerr << "Splitting " << *current_rel << " at "
           << vocab.int_to_words[sentence.words.at(
               connective_repeated_token_index)] << endl;
      //*/

      // Now replace that index and everything after it with the new token.
      ThresholdedCmp<greater_equal<unsigned>> gte_repeated_index(
          (*connective_indices)[connective_repeated_token_index]);
      ReallyDeleteIf(connective_indices, gte_repeated_index);
      connective_indices->push_back(current_arg_token);
      // Delete all arg words that came after the previous last connective word.
      // TODO: make this more efficient?
      for (unsigned arg_num = 0; arg_num < current_rel->ARG_NAMES.size();
          ++arg_num) {
        IndexList* arg_indices = current_rel->GetArgument(arg_num);
        ReallyDeleteIf(arg_indices, gte_repeated_index);
      }
      // Don't advance the arg token.
    } else if (action_name == "SHIFT") {
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
        assert(iter == end - 1);
      }
      current_rel = nullptr;
    }
  }

  return relations;
}


CausalityMetrics LSTMCausalityTagger::Evaluate(
    const BecauseOracleTransitionCorpus& corpus,
    const vector<unsigned>& selections) {
  CausalityMetrics evaluation;
  for (unsigned sentence_index : selections) {
    const Sentence& sentence = corpus.sentences[sentence_index];
    const vector<unsigned>& gold_actions =
        corpus.correct_act_sent[sentence_index];
    vector<CausalityRelation> gold = Decode(sentence, gold_actions);
    vector<CausalityRelation> predicted = Tag(sentence);
    evaluation += CausalityMetrics(gold, predicted);
  }
  return evaluation;
}


void LSTMCausalityTagger::InitializeNetworkParameters() {
  unsigned action_size = vocab.CountActions() + 1;
  unsigned pos_size = vocab.CountPOS() + 10; // bad way of handling new POS
  unsigned vocab_size = vocab.CountWords() + 1;
  unsigned pretrained_dim = parser.pretrained.begin()->second.size();

  assert(!parser.pretrained.empty());
  assert(parser.options.use_pos);

  // Parameters for token representation
  p_w = model.add_lookup_parameters(vocab_size, {options.word_dim});
  p_t = model.add_lookup_parameters(vocab_size, {pretrained_dim});
  for (const auto& it : parser.pretrained)
    p_t->Initialize(it.first, it.second);
  p_a = model.add_lookup_parameters(action_size, {options.action_dim});
  p_pos = model.add_lookup_parameters(pos_size, {options.pos_dim});
  p_tbias = model.add_parameters({options.token_dim});
  p_w2t = model.add_parameters({options.token_dim, options.word_dim});
  p_v2t = model.add_parameters({options.token_dim, pretrained_dim});
  p_p2t = model.add_parameters({options.token_dim, options.pos_dim});

  // Parameters for overall state representation
  p_sbias = model.add_parameters({options.state_dim});
  p_L1toS = model.add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L2toS = model.add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L3toS = model.add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_L4toS = model.add_parameters(
      {options.state_dim, options.lambda_hidden_dim});
  p_actions2S = model.add_parameters(
      {options.state_dim, options.actions_hidden_dim});
  p_rels2S = model.add_parameters({options.state_dim, options.rels_hidden_dim});

  // Parameters for turning states into actions
  p_abias = model.add_parameters({action_size});
  p_s2a = model.add_parameters({action_size, options.state_dim});

  // Parameters for relation list representation
  p_rbias = model.add_parameters({options.rels_hidden_dim});
  p_connective2rel = model.add_parameters(
      {options.rels_hidden_dim, options.span_hidden_dim});
  p_cause2rel = model.add_parameters(
      {options.rels_hidden_dim, options.span_hidden_dim});
  p_effect2rel = model.add_parameters(
      {options.rels_hidden_dim, options.span_hidden_dim});
  p_means2rel = model.add_parameters(
      {options.rels_hidden_dim, options.span_hidden_dim});

  // Parameters for guard/start items in empty lists
  p_action_start = model.add_parameters({options.action_dim});
  p_relations_guard = model.add_parameters({options.rels_hidden_dim});
  p_L1_guard = model.add_parameters({options.token_dim});
  p_L2_guard = model.add_parameters({options.token_dim});
  p_L3_guard = model.add_parameters({options.token_dim});
  p_L4_guard = model.add_parameters({options.token_dim});
  p_connective_guard = model.add_parameters({options.token_dim});
  p_cause_guard = model.add_parameters({options.token_dim});
  p_effect_guard = model.add_parameters({options.token_dim});
  p_means_guard = model.add_parameters({options.token_dim});
}


LSTMCausalityTagger::TaggerState* LSTMCausalityTagger::InitializeParserState(
    ComputationGraph* cg,
    const Sentence& raw_sent,
    const Sentence::SentenceMap& sentence,  // w/ OOVs replaced
    const vector<unsigned>& correct_actions,
    const vector<string>& action_names) {
  CausalityTaggerState* state = new CausalityTaggerState(raw_sent, sentence);

  vector<reference_wrapper<LSTMBuilder>> all_lstms = {
    L1_lstm, L2_lstm, L3_lstm, L4_lstm, action_history_lstm,
    relations_lstm, connective_lstm, cause_lstm, effect_lstm, means_lstm};
  for (reference_wrapper<LSTMBuilder> builder : all_lstms) {
    builder.get().new_graph(*cg);
  }
  // Non-persistent LSTMs get sequences started in StartNewRelation.
  vector<reference_wrapper<LSTMBuilder>> persistent_lstms = {
    L1_lstm, L2_lstm, L3_lstm, L4_lstm, action_history_lstm, relations_lstm};
  for (reference_wrapper<LSTMBuilder> builder : persistent_lstms) {
    builder.get().start_new_sequence();
  }

  // Initialize the sentence-level LSTMs. All but L4 should start out empty.
  action_history_lstm.add_input(GetParamExpr(p_action_start));
  relations_lstm.add_input(GetParamExpr(p_relations_guard));

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
  // TODO: do we need to do anything special to handle OOV words?
  // Add words to L4 in reversed order: other than the initial guard entry,
  // the back of the L4 vector is conceptually the front, contiguous with the
  // back of L3.
  for (const auto& index_and_word_id : sentence | boost::adaptors::reversed) {
    const unsigned token_index = index_and_word_id.first;
    const unsigned word_id = index_and_word_id.second;
    unsigned pos_id = raw_sent.poses.at(token_index);

    Expression word = lookup(*cg, p_w, word_id);
    Expression pretrained = const_lookup(*cg, p_t, word_id);
    Expression pos = lookup(*cg, p_pos, pos_id);
    Expression full_word_repr = rectify(
        affine_transform({GetParamExpr(p_tbias), GetParamExpr(p_w2t), word,
            GetParamExpr(p_p2t), pos, GetParamExpr(p_v2t), pretrained}));

    state->L4.push_back(full_word_repr);
    state->L4i.push_back(token_index);
    L4_lstm.add_input(full_word_repr);
    state->all_tokens[token_index] = full_word_repr;
  }

  state->current_conn_token = state->L4.back();
  state->current_arg_token = state->L4.back();

  return state;
}


bool LSTMCausalityTagger::IsActionForbidden(const unsigned action,
                                            const vector<string>& action_names,
                                            const TaggerState& state) const {
  const CausalityTaggerState& real_state =
      static_cast<const CausalityTaggerState&>(state);
  const string& action_name = action_names[action];
  bool next_arg_token_is_left = real_state.L1.size() > 1;

  if (real_state.currently_processing_rel) {
    // SHIFT is mandatory if there are no more tokens to compare.
    if (real_state.L1.size() <= 1 && real_state.L4.size() <= 1) {
      return action_name[0] != 'S' || action_name[1] != 'H';
    }

    if (next_arg_token_is_left) {
      // SHIFT, CONN-FRAG, and SPLIT are all forbidden when working to the left,
      // as are rightward-oriented operations.
      return !starts_with(action_name, "LEFT")
          && !ends_with(action_name, "LEFT");
    } else {  // processing right side
      // SHIFT is forbidden if some tokens have not yet been compared.
      if (action_name[0] == 'S' && action_name[1] == 'H') {
        return real_state.L1.size() > 1 || real_state.L4.size() > 1;
      }

      // We should never end up processing a relation after 0 actions.
      assert(real_state.prev_action != static_cast<unsigned>(-1));
      // If there was a last action, check that this one is compatible.
      const string& last_action_name = action_names[real_state.prev_action];
      // SPLIT has unique requirements: can't come after a SPLIT or a
      // CONN-FRAG. Also, we can't have less than two connective words.
      if (action_name[0] == 'S' && action_name[1] == 'P') {
        return real_state.current_rel_conn_tokens.size() < 2
            || (last_action_name[0] == 'S' && last_action_name[1] == 'P')
            || last_action_name[0] == 'C';
      }

      // Another special case: forbid CONN-FRAG after SPLIT.
      if (last_action_name[0] == 'S' && last_action_name[1] == 'P'
          && action_name[0] == 'C')
        return false;

      // If it's not a split, a shift, or a forbidden post-split operation,
      // forbid any operation that doesn't act on arg tokens to the right.
      return !starts_with(action_name, "RIGHT")
          && !ends_with(action_name, "RIGHT");
    }
  } else {  // not currently processing a relation
    // NO-CONN is always legal when we're not processing a relation.
    if (action_name[0] == 'N' && action_name[3] == 'C') {
      return false;
    }

    // Even if we're not processing a relation, we still need to check whether
    // we have any words to compare on the left. If we do, only leftward
    // operations are allowed; if not, only rightward ones.
    if (next_arg_token_is_left) {
      return action_name[0] != 'L'              // LEFT-ARC
          && !ends_with(action_name, "LEFT");   // NO-ARC-LEFT
    } else {
      return action_name[0] != 'R'              // RIGHT-ARC
          && !ends_with(action_name, "RIGHT");  // NO-ARC-RIGHT
    }
  }
}


Expression LSTMCausalityTagger::GetActionProbabilities(
    const TaggerState& state) {
  // p_t = sbias + A * actions_lstm + rels2S * rels_lstm + \sum_i LToS_i * L_i
  Expression p_t = affine_transform(
      {GetParamExpr(p_sbias), GetParamExpr(p_actions2S),
          action_history_lstm.back(), GetParamExpr(p_rels2S),
          relations_lstm.back(), GetParamExpr(p_L1toS), L1_lstm.back(),
          GetParamExpr(p_L2toS), L2_lstm.back(), GetParamExpr(p_L3toS),
          L3_lstm.back(), GetParamExpr(p_L4toS), L4_lstm.back()});
  Expression p_t_nonlinear = rectify(p_t);
  // r_t = abias + p2a * nlp
  Expression r_t = affine_transform({GetParamExpr(p_abias), GetParamExpr(p_s2a),
                                     p_t_nonlinear});
  return r_t;
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
    var_name = list.expr; \
    var_name##_i = list##i.expr;
// TODO: redefine to use std::move?
#define MOVE_LIST_ITEM(from_list, to_list, tmp_var) \
    SET_LIST_BASED_VARS(tmp_var, from_list, back()); \
    DO_LIST_PUSH(to_list, tmp_var); \
    DO_LIST_POP(from_list);

void LSTMCausalityTagger::DoAction(unsigned action,
                                   const vector<string>& action_names,
                                   TaggerState* state, ComputationGraph* cg) {
  CausalityTaggerState* cst = static_cast<CausalityTaggerState*>(state);
  const string& action_name = action_names[action];

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
  if (current_arg_token_i != static_cast<unsigned>(-1)) {
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

  auto EmbedCurrentRelation = [&]() {
    Expression current_rel_embedding = rectify(affine_transform(
          {GetParamExpr(p_rbias), GetParamExpr(p_connective2rel),
              connective_lstm.back(), GetParamExpr(p_cause2rel),
              cause_lstm.back(), GetParamExpr(p_effect2rel), effect_lstm.back(),
              GetParamExpr(p_means2rel), means_lstm.back()}));
    relations_lstm.add_input(current_rel_embedding);
  };

  auto UpdateCurrentRelationEmbedding = [&]() {
    relations_lstm.rewind_one_step(); // replace existing relation embedding
    EmbedCurrentRelation();
  };

  auto StartNewRelation = [&]() {
    connective_lstm.start_new_sequence();
    cause_lstm.start_new_sequence();
    effect_lstm.start_new_sequence();
    means_lstm.start_new_sequence();

    connective_lstm.add_input(GetParamExpr(p_connective_guard));
    cause_lstm.add_input(GetParamExpr(p_cause_guard));
    effect_lstm.add_input(GetParamExpr(p_effect_guard));
    means_lstm.add_input(GetParamExpr(p_means_guard));

    cst->currently_processing_rel = true;
  };

  auto EnsureRelationWithConnective = [&](bool embed) {
    if (!cst->currently_processing_rel) {
      StartNewRelation();
      connective_lstm.add_input(current_conn_token);
      cst->current_rel_conn_tokens.push_back(current_conn_token_i);
      if (embed) {
        EmbedCurrentRelation();
      }
    }
  };

  auto AddArc = [&](unsigned action) {
    EnsureRelationWithConnective(false);  // Don't embed yet -- wait for arg

    const string& arc_type = vocab.actions_to_arc_labels[action];
    LSTMBuilder* arg_builder;
    vector<unsigned>* arg_list;
    if (arc_type == "Cause") {
      arg_builder = &cause_lstm;
      arg_list = &cst->current_rel_cause_tokens;
    } else if (arc_type == "Effect") {
      arg_builder = &effect_lstm;
      arg_list = &cst->current_rel_effect_tokens;
    } else {  // arc_type == "Means"
      arg_builder = &means_lstm;
      arg_list = &cst->current_rel_means_tokens;
    }
    arg_builder->add_input(cst->all_tokens.at(current_arg_token_i));
    arg_list->push_back(current_arg_token_i);

    EmbedCurrentRelation();
  };

  if (action_name == "NO-CONN") {
    assert(L4.size() > 1);  // L4 should have at least duplicate of current conn
    DO_LIST_PUSH(L1, current_conn_token);
    DO_LIST_POP(L4);  // remove duplicate of current_conn_token
    SetNewConnectiveToken();
  } else if (action_name == "NO-ARC-LEFT") {
    EnsureRelationWithConnective(true);
    AdvanceArgTokenLeft();
  } else if (action_name == "NO-ARC-RIGHT") {
    EnsureRelationWithConnective(true);
    AdvanceArgTokenRight();
  } else if (action_name == "CONN-FRAG-LEFT") {
    cst->current_rel_conn_tokens.push_back(current_arg_token_i);
    connective_lstm.add_input(current_arg_token);
    UpdateCurrentRelationEmbedding();
    AdvanceArgTokenLeft();
  } else if (action_name == "CONN-FRAG-RIGHT") {
    cst->current_rel_conn_tokens.push_back(current_arg_token_i);
    connective_lstm.add_input(current_arg_token);
    UpdateCurrentRelationEmbedding();
    AdvanceArgTokenRight();
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
    EmbedCurrentRelation();
  } else if (action_name == "SHIFT") {
    assert(L4.size() == 1 && L1.size() == 1);  // processed all tokens?
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
  }

  cst->prev_action = action;
  assert(!L1i.empty() && !L2i.empty() && !L3i.empty() && !L4i.empty());
  assert(L1.size() == L1i.size() && L2.size() == L2i.size() &&
         L3.size() == L3i.size() && L4.size() == L4i.size());
}
