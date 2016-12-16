#include <boost/program_options.hpp>
#include <iostream>
#include <string>

#include "parser/lstm-parser.h"

using namespace lstm_parser;
namespace po = boost::program_options;
using namespace std;

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("parser-model,p",
     po::value<string>()->default_value("latest_model.params.gz"),
     "File from which to load saved syntactic parser model");
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

  LSTMParser parser(conf["parser-model"].as<string>());

  return 0;
}
