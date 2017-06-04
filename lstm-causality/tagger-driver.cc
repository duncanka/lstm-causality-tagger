#include <algorithm>
#include <boost/range/adaptors.hpp>
#include <boost/range/join.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <sstream>
#include <unistd.h>
#include <vector>

#include "cnn/cnn/init.h"
#include "BecauseOracleTransitionCorpus.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace lstm_parser;
namespace po = boost::program_options;
namespace ad = boost::adaptors;
using namespace std;

volatile sig_atomic_t requested_stop = false;

po::typed_value<bool>* POBooleanFlag(bool default_val) {
  return po::value<bool>()->default_value(default_val)->implicit_value(true);
}

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("parser-model,M",
      po::value<string>()->default_value(
          "lstm-parser/english_pos_2_32_100_20_100_12_20.params"),
     "File from which to load saved syntactic parser model")
    ("training-data,t", po::value<string>(),
     "Directory containing training data")
    ("folds,f", po::value<unsigned>()->default_value(20),
     "How many folds to split the data into for cross-validation")
    ("dev-pct,d", po::value<double>()->default_value(0.2),
     "Percent of training data in each shuffle to use as dev (tuning)")
    ("train", "Whether to train the tagger")
    ("action-dim,a", po::value<unsigned>()->default_value(8),
     "Dimension for vector representation of actions")
    ("word-dim,w", po::value<unsigned>()->default_value(10),
     "Dimension for vector representation of words")
    ("state-dim,s", po::value<unsigned>()->default_value(72),
     "Dimension for overall tagger state")
    ("token-dim,i", po::value<unsigned>()->default_value(48),
     "Dimension for representation of tokens as input to span or lambda LSTMs")
    ("lambda-hidden-dim,h", po::value<unsigned>()->default_value(48),
     "Dimension of the hidden state of each lambda LSTM")
    ("actions-hidden-dim,A", po::value<unsigned>()->default_value(32),
     "Dimension of the hidden state of the action history LSTM")
    ("parse-path-hidden-dim,p", po::value<unsigned>()->default_value(12),
     "Dimension of the hidden state of the parse path embedding LSTM. If 0,"
     " parse paths are not used.")
    ("span-hidden-dim,S", po::value<unsigned>()->default_value(32),
     "Dimension of each connective/argument span LSTM's hidden state")
    ("lstm-layers,l", po::value<unsigned>()->default_value(2),
     "Number of layers for each stack LSTM")
    ("epochs-cutoff,e", po::value<double>()->default_value(5),
     "Number of training epochs without an improvement in the best F1 to allow"
     " before stopping training on that fold (SIGINT always works to stop; see"
     " also --recent-improvements-cutoff)")
    ("compare-punct,c",
     "Whether to count punctuation when comparing argument spans")
    ("eval-pairwise,P", POBooleanFlag(true),
     "Whether to also evaluate on only instances with both cause and effect")
    ("subtrees,u", POBooleanFlag(true),
     "Whether to include embeddings of parse subtrees in token representations")
    ("gated-parse,g", POBooleanFlag(true),
     "Whether to include gated parse tree embedding in the overall state")
    ("dropout,D", po::value<float>()->default_value(0.0),
     "Dropout rate (no dropout is performed for a value of 0)")
    ("new-conn-action,n", POBooleanFlag(true),
     "Whether starting a relation is a separate action (must match data)")
    ("shift-action,H", POBooleanFlag(false),
     "Whether completing a relation is a separate action (must match data)")
    ("known-conns-only,K", POBooleanFlag(false),
     "Whether to restrict the possible transitions at test time to allow only"
     " known connectives")
    ("log-diffs,L", POBooleanFlag(false),
     "Whether to log differences between correct and predicted")
    ("dev-eval-period,E", po::value<unsigned>()->default_value(25),
     "How many training iterations to go between dev evaluations")
    ("recent-improvements-cutoff,I", po::value<double>()->default_value(0.85),
     "Don't stop training yet if this % of evaluations in last --epochs-cutoff"
     " epochs have been an increase from the immediately prior evaluation")
    ("cv-start-at", po::value<unsigned>()->default_value(1),
     "Cross-validation fold to start at (useful for debugging/parallelizing")
    ("cv-end-at", po::value<unsigned>()->default_value(-1),
     "Cross-validation fold to end on (useful for debugging/parallelizing");

  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(0);
  }
}


void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again; quitting.\n";
    exit(1);
  }
  cerr << "\nReceived SIGINT. Terminating optimization early...\n";
  requested_stop = true;
}


int main(int argc, char** argv) {
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

  unsigned random_seed = cnn::Initialize(argc, argv);
  srand(random_seed + 1);

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  LSTMCausalityTagger tagger(
      conf["parser-model"].as<string>(),
      LSTMCausalityTagger::TaggerOptions{
          conf["word-dim"].as<unsigned>(),
          conf["lstm-layers"].as<unsigned>(),
          conf["token-dim"].as<unsigned>(),
          conf["lambda-hidden-dim"].as<unsigned>(),
          conf["actions-hidden-dim"].as<unsigned>(),
          conf["parse-path-hidden-dim"].as<unsigned>(),
          conf["span-hidden-dim"].as<unsigned>(),
          conf["action-dim"].as<unsigned>(),
          conf["state-dim"].as<unsigned>(),
          conf["dropout"].as<float>(),
          conf["subtrees"].as<bool>(),
          conf["gated-parse"].as<bool>(),
          conf["new-conn-action"].as<bool>(),
          conf["shift-action"].as<bool>(),
          conf["known-conns-only"].as<bool>(),
          conf["log-diffs"].as<bool>()});
  if (conf.count("train")) {
    double dev_pct = conf["dev-pct"].as<double>();
    if (dev_pct < 0.0 || dev_pct > 1.0) {
      cerr << "Invalid dev percentage: " << dev_pct << endl;
      abort();
    }
    if (!conf.count("training-data")) {
      cerr << "Can't train without training corpus!"
              " Please provide --training-data." << endl;
      abort();
    }
    if (tagger.options.dropout >= 1.0) {
      cerr << "Invalid dropout rate: " << tagger.options.dropout << endl;
      abort();
    }
    double epochs_cutoff = conf["epochs-cutoff"].as<double>();
    double recent_improvements_cutoff =
        conf["recent-improvements-cutoff"].as<double>();
    bool compare_punct = conf.count("compare-punct");
    bool eval_pairwise = conf.count("eval-pairwise");
    unsigned dev_eval_period = conf["dev-eval-period"].as<unsigned>();
    unsigned folds = conf["folds"].as<unsigned>();

    const string& training_path = conf["training-data"].as<string>();
    BecauseOracleTransitionCorpus full_corpus(tagger.GetVocab(), training_path,
                                              true);
    tagger.FinalizeVocab();
    unsigned num_sentences = full_corpus.sentences.size();
    cerr << "Corpus size: " << num_sentences << " sentences" << endl;

    ostringstream os;
    os << "tagger_" << tagger.options.word_dim
       << '_' << tagger.options.lstm_layers
       << '_' << tagger.options.token_dim
       << '_' << tagger.options.lambda_hidden_dim
       << '_' << tagger.options.actions_hidden_dim
       << '_' << tagger.options.parse_path_hidden_dim
       << '_' << tagger.options.span_hidden_dim
       << '_' << tagger.options.action_dim
       << '_' << tagger.options.state_dim
       << '_' << tagger.options.dropout;
    if (tagger.options.subtrees)
      os << "_subtrees";
    if (tagger.options.gated_parse)
      os << "_gated-parse";
    if (tagger.options.new_conn_action)
      os << "_new-conn";
    if (tagger.options.shift_action)
      os << "_shift";
    if (tagger.options.known_conns_only)
      os << "_known";
    os << "__pid" << getpid() << ".params";
    const string model_fname = os.str();
    cerr << "Writing parameters to file: " << model_fname << endl;

    signal(SIGINT, signal_callback_handler);
    vector<unsigned> all_sentence_indices(num_sentences);
    iota(all_sentence_indices.begin(), all_sentence_indices.end(), 0);
    random_shuffle(all_sentence_indices.begin(), all_sentence_indices.end());
    if (folds <= 1) {
       tagger.Train(&full_corpus, all_sentence_indices, dev_pct, compare_punct,
                    model_fname, dev_eval_period, epochs_cutoff,
                    recent_improvements_cutoff, &requested_stop);
     } else {
      // For cutoffs, we use one *past* the index where the fold should stop.
      vector<unsigned> fold_cutoffs(folds);
      unsigned uneven_sentences_to_distribute = num_sentences % folds;
      unsigned next_cutoff = num_sentences / folds;
      // TODO: merge this loop with the one below?
      for (unsigned i = 0; i < folds;
           ++i, next_cutoff += num_sentences / folds) {
        if (uneven_sentences_to_distribute > 0) {
          ++next_cutoff;
          --uneven_sentences_to_distribute;
        }
        fold_cutoffs[i] = next_cutoff;
      }
      assert(fold_cutoffs.back() == all_sentence_indices.size());

      unsigned previous_cutoff = 0;
      vector<CausalityMetrics> evaluation_results;
      vector<CausalityMetrics> pairwise_eval_results;
      evaluation_results.reserve(folds);
      if (eval_pairwise) {
        pairwise_eval_results.reserve(folds);
      }
      // Folds are 1-indexed, so subtract 1 from CL params to get indices.
      unsigned last_fold = conf["cv-end-at"].as<unsigned>();
      if (last_fold == UNSIGNED_NEG_1)
        last_fold = folds + 1;
      for (unsigned fold = conf["cv-start-at"].as<unsigned>() - 1;
           fold < min(last_fold - 1, folds); ++fold) {
        cerr << "Starting fold " << fold + 1 << " of " << folds << endl;
        size_t current_cutoff = fold_cutoffs[fold];
        auto training_range = join(
            all_sentence_indices | ad::sliced(0u, previous_cutoff),
            all_sentence_indices | ad::sliced(current_cutoff,
                                              all_sentence_indices.size()));
        vector<unsigned> fold_train_order(training_range.begin(),
                                          training_range.end());

        assert(fold_train_order.size()
               == num_sentences - (current_cutoff - previous_cutoff));

        tagger.Train(&full_corpus, fold_train_order, dev_pct, compare_punct,
                     model_fname, dev_eval_period, epochs_cutoff,
                     recent_improvements_cutoff, &requested_stop);

        cerr << "Evaluating..." << endl;
        tagger.LoadModel(model_fname);  // Reset to last saved state
        vector<unsigned> fold_test_order(
            all_sentence_indices.begin() + previous_cutoff,
            all_sentence_indices.begin() + current_cutoff);
        cout << "Evaluation for fold " << fold + 1 << " ("
             << fold_test_order.size() << " test sentences)" << endl;
        CausalityMetrics evaluation = tagger.Evaluate(
            &full_corpus, fold_test_order, compare_punct);
        IndentingOStreambuf indent(cout);
        cout << evaluation << '\n' << endl;
        evaluation_results.push_back(evaluation);

        if (eval_pairwise) {
          CausalityMetrics pairwise_evaluation = tagger.Evaluate(
              &full_corpus, fold_test_order, compare_punct, eval_pairwise);
          cout << "\nPairwise evaluation:";
          IndentingOStreambuf indent(cout);
          cout << '\n' << pairwise_evaluation << '\n' << endl;
          pairwise_eval_results.push_back(pairwise_evaluation);
        }

        requested_stop = false;
        previous_cutoff = current_cutoff;
        tagger.Reset();  // Reset for next fold
      }

      {
        cout << "Average evaluation:\n";
        IndentingOStreambuf indent(cout);
        auto evals_range = boost::make_iterator_range(evaluation_results);
        cout << AveragedCausalityMetrics(evals_range) << endl;
      }

      if (eval_pairwise) {
        cout << "Average pairwise evaluation:" << '\n';
        IndentingOStreambuf indent(cout);
        auto evals_range = boost::make_iterator_range(pairwise_eval_results);
        cout << AveragedCausalityMetrics(evals_range) << endl;
      }
    }
  }


  return 0;
}
