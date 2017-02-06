#include <boost/algorithm/string/predicate.hpp>
#include <chrono>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cnn/model.h"
#include "BecauseData.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"
#include "utilities.h"

using namespace std;
using namespace cnn;
using namespace lstm_parser;
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
      parser.LogProbParser(sentence, parser.vocab, &cg, &parser_state);
      LogProbTagger(&cg, sentence, correct_actions, corpus.vocab->actions,
                    corpus.vocab->int_to_words, &correct);
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

        cnn::ComputationGraph cg;
        vector<unsigned> actions = LogProbTagger(
            &cg, sentence, dev_corpus.correct_act_sent[sii],
            dev_corpus.vocab->actions, dev_corpus.vocab->int_to_words,
            &correct);
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

void LSTMCausalityTagger::SaveModel(const string& model_fname,
                                    bool softlink_created) {
  ofstream out_file(model_fname);
  eos::portable_oarchive archive(out_file);
  archive << *this;
  cerr << "Model saved." << endl;
  // Create a soft link to the most recent model in order to make it
  // easier to refer to it in a shell script.
  if (!softlink_created) {
    string softlink = "latest_model.params";

    if (system((string("rm -f ") + softlink).c_str()) == 0
        && system(("ln -s " + model_fname + " " + softlink).c_str()) == 0) {
      cerr << "Created " << softlink << " as a soft link to " << model_fname
           << " for convenience." << endl;
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
      [&](vector<unsigned>::const_iterator iter) {
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
      AdvanceConnToken(iter);
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

    } else if (action_name == "SHIFT") {
      // Complete a relation.
      AdvanceConnToken(iter);
      current_rel = nullptr;
    }
  }

  return relations;
}


vector<unsigned> LSTMCausalityTagger::LogProbTagger(
    cnn::ComputationGraph* hg, const Sentence& sentence,
    const vector<unsigned>& correct_actions, const vector<string>& action_names,
    const vector<string>& int_to_words, double* correct) {
  // TODO: Fill me in
  return {};
}
