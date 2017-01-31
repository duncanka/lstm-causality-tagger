#include <boost/program_options.hpp>
#include <iostream>
#include <string>

#include "BecauseOracleTransitionCorpus.h"
#include "LSTMCausalityTagger.h"

using namespace lstm_parser;
namespace po = boost::program_options;
using namespace std;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("parser-model,p",
     po::value<string>()->default_value("latest_model.params.gz"),
     "File from which to load saved syntactic parser model")
    ("training-data,t", po::value<string>(),
     "Directory containing training data");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(0);
  }
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  LSTMCausalityTagger tagger(conf["parser-model"].as<string>());
  CorpusVocabulary vocab(tagger.GetParser().vocab);
  const string& training_path = conf["training-data"].as<string>();
  BecauseOracleTransitionCorpus corpus(&vocab, training_path, true);

  return 0;
}
