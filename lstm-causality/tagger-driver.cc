#include <algorithm>
#include <boost/range/adaptors.hpp>
#include <boost/range/join.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <csignal>
#include <numeric>
#include <string>
#include <sstream>

#include "BecauseOracleTransitionCorpus.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace lstm_parser;
namespace po = boost::program_options;
namespace ad = boost::adaptors;
using namespace std;

volatile sig_atomic_t requested_stop = false;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("parser-model,p",
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
    ("action-dim,a", po::value<unsigned>()->default_value(20),
     "Dimension for vector representation of actions")
    ("pos-dim,p", po::value<unsigned>()->default_value(20),
     "Dimension for vector representation of parts of speech")
    ("word-dim,w", po::value<unsigned>()->default_value(48),
     "Dimension for vector representation of words")
    ("rels-hidden-dim,r", po::value<unsigned>()->default_value(100),
     "Dimension for vector representation of an entire causal relation")
    ("state-dim,s", po::value<unsigned>()->default_value(100),
     "Dimension for overall tagger state")
    ("token-dim,i", po::value<unsigned>()->default_value(64),
     "Dimension for representation of tokens as input to span or lambda LSTMs")
    ("lambda-hidden-dim,h", po::value<unsigned>()->default_value(64),
     "Dimension of the hidden state of each lambda LSTM")
    ("actions-hidden-dim,c", po::value<unsigned>()->default_value(64),
     "Dimension of the hidden state of the action history LSTM")
    ("span-hidden-dim,n", po::value<unsigned>()->default_value(64),
     "Dimension of each connective/argument span LSTM's hidden state")
    ("lstm-layers,l", po::value<unsigned>()->default_value(2),
     "Number of layers for each stack LSTM");

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
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}


int main(int argc, char** argv) {
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

  cnn::Initialize(argc, argv);

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  LSTMCausalityTagger tagger(
      conf["parser-model"].as<string>(),
      {conf["word-dim"].as<unsigned>(),
       conf["lstm-layers"].as<unsigned>(),
       conf["token-dim"].as<unsigned>(),
       conf["lambda-hidden-dim"].as<unsigned>(),
       conf["actions-hidden-dim"].as<unsigned>(),
       conf["span-hidden-dim"].as<unsigned>(),
       conf["rels-hidden-dim"].as<unsigned>(),
       conf["action-dim"].as<unsigned>(),
       conf["pos-dim"].as<unsigned>(),
       conf["state-dim"].as<unsigned>()});
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

    ostringstream os;
    os << "tagger_" << tagger.options.action_dim
       << '_' << tagger.options.token_dim
       << '_' << tagger.options.lambda_hidden_dim
       << '_' << tagger.options.actions_hidden_dim
       << '_' << tagger.options.span_hidden_dim
       << '_' << tagger.options.lstm_layers
       << '_' << tagger.options.pos_dim
       << '_' << tagger.options.rels_hidden_dim
       << '_' << tagger.options.state_dim
       << '_' << tagger.options.word_dim
       << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Writing parameters to file: " << fname << endl;

    const string& training_path = conf["training-data"].as<string>();
    BecauseOracleTransitionCorpus full_corpus(tagger.GetVocab(), training_path,
                                              true);
    tagger.FinalizeVocab();
    signal(SIGINT, signal_callback_handler);

    unsigned num_sentences = full_corpus.sentences.size();
    cerr << "Corpus size: " << num_sentences << " sentences" << endl;
    vector<unsigned> all_sentence_indices(num_sentences);
    iota(all_sentence_indices.begin(), all_sentence_indices.end(), 0);
    random_shuffle(all_sentence_indices.begin(), all_sentence_indices.end());
    unsigned folds = conf["folds"].as<unsigned>();
    // For cutoffs, we use one *past* the index where the fold should stop.
    vector<unsigned> fold_cutoffs(folds);
    unsigned uneven_sentences_to_distribute = num_sentences % folds;
    unsigned next_cutoff = num_sentences / folds;
    for (unsigned i = 0; i < folds; ++i, next_cutoff += num_sentences / folds) {
      if (uneven_sentences_to_distribute > 0) {
        ++next_cutoff;
        --uneven_sentences_to_distribute;
      }
      fold_cutoffs[i] = next_cutoff;
    }
    assert(fold_cutoffs.back() == all_sentence_indices.size());

    unsigned previous_cutoff = 0;
    for (unsigned fold = 0; fold < folds; ++fold) {
      cerr << "Starting fold " << fold + 1 << " of " << folds << endl;
      unsigned current_cutoff = fold_cutoffs[fold];
      auto training_range = join(
          all_sentence_indices | ad::sliced(0, previous_cutoff),
          all_sentence_indices | ad::sliced(current_cutoff,
                                            all_sentence_indices.size()));
      vector<unsigned> fold_train_order(training_range.begin(),
                                        training_range.end());

      unsigned fold_test_size = current_cutoff - previous_cutoff;
      unsigned fold_training_size = num_sentences - fold_test_size;
      assert(fold_train_order.size() == fold_training_size);

      tagger.Train(full_corpus, fold_train_order, dev_pct, fname,
                   &requested_stop);

      vector<unsigned> fold_test_order(
          all_sentence_indices.begin() + previous_cutoff,
          all_sentence_indices.begin() + current_cutoff);
      CausalityMetrics evaluation = tagger.Evaluate(full_corpus,
                                                    fold_test_order);
      cerr << "Evaluation for fold " << fold << ':' << endl;
      cerr << evaluation << endl << endl;

      requested_stop = false;
      previous_cutoff = current_cutoff;
    }

  }


  return 0;
}
