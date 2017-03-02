#include <boost/algorithm/string/predicate.hpp>
#include <chrono>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
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
using namespace cnn;
using namespace cnn::expr;
using namespace lstm_parser;
using namespace boost::algorithm;
typedef BecauseRelation::IndexList IndexList;


void LSTMCausalityTagger::Train(const BecauseOracleTransitionCorpus& corpus,
                                const BecauseOracleTransitionCorpus& dev_corpus,
                                const double unk_prob,
                                const string& model_fname,
                                const volatile bool* requested_stop) {
  const unsigned num_sentences = corpus.sentences.size();
  vector<unsigned> order(corpus.sentences.size());
  for (unsigned i = 0; i < corpus.sentences.size(); ++i)
    order[i] = i;
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

  bool softlink_created = false;
  double sentences_seen = 0;
  double best_f1 = 0;
  unsigned actions_seen = 0;
  double correct = 0;
  double llh = 0;
  bool first = true;
  int iter = -1;

  unsigned sentence_i = num_sentences;
  while (!requested_stop || !(*requested_stop)) {
    ++iter;
    for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
      if (sentence_i == num_sentences) {
        sentence_i = 0;
        if (first) {
          first = false;
        } else {
          sgd.update_epoch();
        }
        cerr << "**SHUFFLE\n";
        random_shuffle(order.begin(), order.end());
      }

      const Sentence& sentence = corpus.sentences[order[sentence_i]];
      const vector<unsigned>& correct_actions = corpus.correct_act_sent[sii];

      ComputationGraph cg;
      Expression parser_state;
      parser.LogProbTagger(sentence, *parser.GetVocab(), &cg, true,
                           &parser_state);
      LogProbTagger(&cg, sentence, sentence.words, correct_actions,
                    corpus.vocab->actions, corpus.vocab->int_to_words,
                    &correct, &parser_state);
      double lp = as_scalar(cg.incremental_forward());
      if (lp < 0) {
        cerr << "Log prob < 0 on sentence " << order[sentence_i] << ": lp="
             << lp << endl;
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
    cerr << "update #" << iter << " (epoch " << (sentences_seen / num_sentences)
         << " |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "
         << llh << " ppl: " << exp(llh / actions_seen) << " err: "
         << (actions_seen - correct) / actions_seen << endl;
    // TODO: move declaration to make this unnecessary.
    llh = actions_seen = correct = 0;

    if (iter % 25 == 1) {
      // report on dev set
      unsigned dev_size = dev_corpus.sentences.size();
      // dev_size = 100;
      double llh = 0;
      double num_actions = 0;
      double correct = 0;
      BecauseRelationMetrics<> evaluation;
      const auto t_start = chrono::high_resolution_clock::now();
      for (unsigned sii = 0; sii < dev_size; ++sii) {
        const Sentence& sentence = dev_corpus.sentences[sii];

        ComputationGraph cg;
        Expression parser_state;
        parser.LogProbTagger(sentence, *parser.GetVocab(), &cg, true,
                             &parser_state);
        vector<unsigned> actions = LogProbTagger(sentence, *dev_corpus.vocab,
                                                 &cg, false);
        llh += as_scalar(cg.incremental_forward());
        vector<CausalityRelation> predicted = Decode(sentence, actions,
                                                     *dev_corpus.vocab);

        const vector<unsigned>& gold_actions = dev_corpus.correct_act_sent[sii];
        vector<CausalityRelation> gold = Decode(sentence, gold_actions,
                                                *dev_corpus.vocab);

        num_actions += actions.size();
        evaluation += BecauseRelationMetrics<>(gold, predicted);
      }

      auto t_end = chrono::high_resolution_clock::now();
      cerr << "  **dev (iter=" << iter << " epoch="
           << (sentences_seen / num_sentences)
           << ")llh=" << llh << " ppl: " << exp(llh / num_actions)
           << "\terr: " << (num_actions - correct) / num_actions
           << " evaluation: \n  " << evaluation
           << "\n[" << dev_size << " sents in "
           << chrono::duration<double, milli>(t_end - t_start).count() << " ms]"
           << endl;

      if (evaluation.connective_metrics.GetF1() > best_f1) {
        best_f1 = evaluation.connective_metrics.GetF1();
        SaveModel(model_fname, softlink_created);
        softlink_created = true;
      }
    }
  }
}


vector<CausalityRelation> LSTMCausalityTagger::Decode(
    const Sentence& sentence, const vector<unsigned> actions,
    const lstm_parser::CorpusVocabulary& vocab) {
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

  auto AdvanceConnToken =
      [&](vector<unsigned>::const_iterator iter,
          vector<unsigned>::const_iterator end) {
        // Move the current token onto lambda_1 and advance to the next one.
        assert(!lambda_4.empty());
        lambda_1.push_back(current_conn_token);
        lambda_4.pop_front();// remove duplicate of current_conn_token
        if (!lambda_4.empty()) {
          current_conn_token = lambda_4.front();
        } else {
          // If we have no more tokens to pull in, we'd better be on the last
          // action.
          assert(iter == end - 1);
        }
      };

  auto AdvanceArgTokenLeft =
      [&]() {
        assert(!lambda_1.empty());
        lambda_2.push_front(lambda_1.back());
        lambda_1.pop_back();
        current_arg_token = lambda_1.back();
      };

  auto AdvanceArgTokenRight =
      [&]() {
        assert(!lambda_4.empty());
        lambda_3.push_back(lambda_4.front());
        lambda_4.pop_front();
        current_arg_token = lambda_4.front();
      };

  auto AddArc =
      [&](unsigned action) {
        const string& arc_type = vocab.actions_to_arc_labels[action];
        auto arg_iter = find(CausalityRelation::ARG_NAMES.begin(),
                             CausalityRelation::ARG_NAMES.end(), arc_type);
        assert(arg_iter != CausalityRelation::ARG_NAMES.end());

        if (!current_rel) {
          relations.emplace_back(
              sentence, vocab, CausalityRelation::CONSEQUENCE,
              IndexList({current_conn_token}));
          current_rel = &relations.back();
        }
        current_rel->AddToArgument(
            arg_iter - CausalityRelation::ARG_NAMES.begin(), current_arg_token);
      };

  for (auto iter = actions.begin(), end = actions.end(); iter != end; ++iter) {
    unsigned action = *iter;
    const string& action_name = vocab.actions[action];
    if (action_name == "NO_CONN") {
      AdvanceConnToken(iter, end);
    } else if (action_name == "NO-ARC-LEFT") {
      AdvanceArgTokenLeft();
    } else if (action_name == "NO-ARC-RIGHT") {
      AdvanceArgTokenRight();
    } else if (action_name == "CONN-FRAG-LEFT") {
      current_rel->AddConnectiveToken(current_arg_token);
      AdvanceArgTokenLeft();
    } else if (action_name == "CONN-FRAG-RIGHT") {
      current_rel->AddConnectiveToken(current_arg_token);
      AdvanceArgTokenRight();
    } else if (boost::algorithm::starts_with(action_name, "RIGHT-ARC")) {
      AddArc(action);
      AdvanceArgTokenRight();
    } else if (boost::algorithm::starts_with(action_name, "LEFT-ARC")) {
      AddArc(action);
      AdvanceArgTokenLeft();
    } else if (action_name == "SPLIT") {
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
      cerr << "Splitting " << *current_rel << " at "
           << vocab.int_to_words[
               sentence.words.at(connective_repeated_token_index)];

      // Now replace that index and everything after it with the new token.
      ThresholdedCmp<greater_equal<unsigned>> gte_repeated_index(
          connective_repeated_token_index);
      ReallyDeleteIf(connective_indices, gte_repeated_index);
      connective_indices->push_back(current_arg_token);
      // Delete all arg words that came after the previous last connective word.
      // TODO: make this more efficient?
      for (unsigned arg_num = 0; arg_num < current_rel->ARG_NAMES.size();
          ++arg_num) {
        IndexList* arg_indices = current_rel->GetArgument(arg_num);
        ReallyDeleteIf(arg_indices, gte_repeated_index);
      }
      // Don't advance the arg token
    } else if (action_name == "SHIFT") {
      // Complete a relation.
      AdvanceConnToken(iter, end);
      current_rel = nullptr;
    }
  }

  return relations;
}


void LSTMCausalityTagger::InitializeNetworkParameters() {
  unsigned action_size = vocab.CountActions() + 1;
  unsigned pos_size = vocab.CountPOS() + 10; // bad way of handling new POS
  unsigned vocab_size = vocab.CountWords() + 1;
  unsigned pretrained_dim = parser.pretrained.begin()->second.size();

  assert(!parser.pretrained.empty());
  assert(parser.options.use_pos);

  p_w = model.add_lookup_parameters(vocab_size, {options.word_dim});
  p_t = model.add_lookup_parameters(vocab_size, {pretrained_dim});
  for (const auto& it : parser.pretrained)
    p_t->Initialize(it.first, it.second);
  p_a = model.add_lookup_parameters(action_size, {options.action_dim});
  p_r = model.add_lookup_parameters(action_size, {options.rel_dim});
  p_pos = model.add_lookup_parameters(pos_size, {options.pos_dim});

  p_sbias = model.add_parameters({options.state_dim});
  p_L1toS = model.add_parameters({options.state_dim, options.lstm_hidden_dim});
  p_L2toS = model.add_parameters({options.state_dim, options.lstm_hidden_dim});
  p_L3toS = model.add_parameters({options.state_dim, options.lstm_hidden_dim});
  p_L4toS = model.add_parameters({options.state_dim, options.lstm_hidden_dim});
  p_actions2S = model.add_parameters({options.state_dim,
                                      options.lstm_hidden_dim});
  p_rels2S = model.add_parameters({options.state_dim, options.rel_dim});
  p_s2a = model.add_parameters({action_size, options.state_dim});
  p_abias = model.add_parameters({action_size});

  p_rbias = model.add_parameters({options.rel_dim});
  p_connective_rel = model.add_parameters({options.rel_dim});
  p_cause_rel = model.add_parameters({options.rel_dim});
  p_effect_rel = model.add_parameters({options.rel_dim});
  p_means_rel = model.add_parameters({options.rel_dim});

  p_w2l = model.add_parameters({options.lstm_input_dim, options.word_dim});
  p_t2l = model.add_parameters({options.lstm_input_dim, pretrained_dim});
  p_p2l = model.add_parameters({options.lstm_input_dim, options.pos_dim});
  p_ib = model.add_parameters({options.lstm_input_dim});
  p_action_start = model.add_parameters({options.action_dim});

  p_relations_guard = model.add_parameters({options.lstm_input_dim});
  p_L1_guard = model.add_parameters({options.lstm_input_dim});
  p_L2_guard = model.add_parameters({options.lstm_input_dim});
  p_L3_guard = model.add_parameters({options.lstm_input_dim});
  p_L4_guard = model.add_parameters({options.lstm_input_dim});
  p_connective_guard = model.add_parameters({options.lstm_input_dim});
  p_cause_guard = model.add_parameters({options.lstm_input_dim});
  p_effect_guard = model.add_parameters({options.lstm_input_dim});
  p_means_guard = model.add_parameters({options.lstm_input_dim});
}


LSTMCausalityTagger::TaggerState* LSTMCausalityTagger::InitializeParserState(
    ComputationGraph* cg,
    const Sentence& raw_sent,
    const Sentence::SentenceMap& sentence,  // w/ OOVs replaced
    const vector<unsigned>& correct_actions,
    const vector<string>& action_names) {
  CausalityTaggerState* state = new CausalityTaggerState;
  state->current_conn_token = sentence.begin()->first;
  state->current_arg_token = state->current_conn_token;
  state->currently_processing_rel = false;

  vector<reference_wrapper<LSTMBuilder>> lstms {
    L1_lstm, L2_lstm, L3_lstm, L4_lstm, action_history_lstm,
    relations_lstm, connective_lstm, cause_lstm, effect_lstm, means_lstm};
  for (reference_wrapper<LSTMBuilder> builder : lstms) {
    builder.get().new_graph(*cg);
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
  state->L2i.push_back(-1);

  state->L4.push_back(GetParamExpr(p_L4_guard));
  state->L4i.push_back(-1);
  // TODO: do we need to do anything special to handle OOV words?
  for (const auto& index_and_word_id : sentence) {
    const unsigned token_index = index_and_word_id.first;
    const unsigned word_id = index_and_word_id.second;
    unsigned pos_id = raw_sent.poses.at(token_index);

    Expression word = lookup(*cg, p_w, word_id);
    Expression pretrained = const_lookup(*cg, p_t, word_id);
    Expression pos = lookup(*cg, p_pos, pos_id);
    Expression full_word_repr = rectify(
        affine_transform( {GetParamExpr(p_ib), GetParamExpr(p_w2l), word,
            GetParamExpr(p_p2l), pos, GetParamExpr(p_t2l), pretrained}));

    state->L4.push_back(full_word_repr);
    state->L4i.push_back(token_index);
  }
  // Add to LSTM in reverse order so that rewind_one_step pulls off the left.
  // (Add guard first; then skip it at the end.)
  L4_lstm.add_input(GetParamExpr(p_L4_guard));
  for (auto iter = state->L4.rbegin(); iter != state->L4.rend() - 1; ++iter) {
    L4_lstm.add_input(*iter);
  }

  return state;
}


bool LSTMCausalityTagger::IsActionForbidden(const unsigned action,
                                            const vector<string>& action_names,
                                            const TaggerState& state) const {
  const CausalityTaggerState& real_state =
      static_cast<const CausalityTaggerState&>(state);
  const string& action_name = action_names[action];
  if (!real_state.currently_processing_rel) {
    return action_name[0] == 'N'  // NO-CONN
        || action_name[0] = 'R'   // RIGHT-ARC
        || action_name[0] = 'L';  // LEFT-ARC
  } else { // When we're processing a relation, everything except NO-CONN is OK.
    return action_name[0] != 'N';
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


void LSTMCausalityTagger::DoAction(unsigned action,
                                   const vector<string>& action_names,
                                   TaggerState* state, ComputationGraph* cg) {
  const CausalityTaggerState* real_state =
      static_cast<const CausalityTaggerState*>(state);

}
