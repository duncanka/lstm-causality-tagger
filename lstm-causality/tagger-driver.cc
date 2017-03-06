#include <boost/program_options.hpp>
#include <iostream>
#include <signal.h>
#include <string>
#include <sstream>

#include "BecauseOracleTransitionCorpus.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace lstm_parser;
namespace po = boost::program_options;
using namespace std;

volatile bool requested_stop = false;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("parser-model,p",
      po::value<string>()->default_value(
          "lstm-parser/english_pos_2_32_100_20_100_12_20.params"),
     "File from which to load saved syntactic parser model")
    ("training-data,t", po::value<string>(),
     "Directory containing training data")
    ("dev-pct,d", po::value<double>()->default_value(0.2),
     "Percent of data in each training shuffle to use as dev (tuning)")
    ("train", "Whether to train the tagger")
    ("action-dim,a", po::value<unsigned>()->default_value(20),
     "Dimension for vector representation of actions")
    ("pos-dim,p", po::value<unsigned>()->default_value(12),
     "Dimension for vector representation of parts of speech")
    ("word-dim,w", po::value<unsigned>()->default_value(50),
     "Dimension for entire vector representation of words")
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
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

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
    os << "tagger_" << '_' << tagger.options.action_dim
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
    BecauseOracleTransitionCorpus corpus(tagger.GetVocab(), training_path,
                                         true);
    tagger.FinalizeVocab();
    signal(SIGINT, signal_callback_handler);

    tagger.Train(corpus, dev_pct, fname, &requested_stop);
  }


  return 0;
}
