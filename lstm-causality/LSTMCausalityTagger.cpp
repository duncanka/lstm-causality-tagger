#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "cnn/model.h"
#include "BecauseData.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace std;
using namespace cnn;

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

      const map<unsigned, unsigned>& sentence =
          corpus.sentences[order[sentence_i]];
      const map<unsigned, unsigned>& sentence_pos =
          corpus.sentences_pos[order[sentence_i]];
      const vector<unsigned>& correct_actions =
          corpus.correct_act_sent[order[sentence_i]];

      ComputationGraph cg;
      Expression parser_state;
      parser.LogProbParser(sentence, sentence_pos, parser.vocab, &cg,
                           &parser_state);
      LogProbTagger(&cg, sentence, sentence_pos, correct_actions,
                    corpus.vocab->actions, corpus.vocab->int_to_words,
                    &correct);
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
        const map<unsigned, unsigned>& sentence = dev_corpus.sentences[sii];
        const map<unsigned, unsigned>& sentence_pos =
            dev_corpus.sentences_pos[sii];

        cnn::ComputationGraph cg;
        std::vector<unsigned> actions = LogProbTagger(
            &cg, sentence, sentence_pos, dev_corpus.correct_act_sent[sii],
            dev_corpus.vocab->actions, dev_corpus.vocab->int_to_words,
            &correct);
        llh += as_scalar(cg.incremental_forward());
        vector<CausalityRelation> predicted = ReconstructCausations(
            sentence, actions, *dev_corpus.vocab);

        const vector<unsigned>& gold_actions = dev_corpus.correct_act_sent[sii];
        vector<CausalityRelation> gold = ReconstructCausations(
            sentence, gold_actions, *dev_corpus.vocab);

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
