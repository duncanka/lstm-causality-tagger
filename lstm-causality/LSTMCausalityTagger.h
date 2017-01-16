#ifndef LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_
#define LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_

#include <string>

#include "parser/lstm-parser.h"

class LSTMCausalityTagger {
public:
  explicit LSTMCausalityTagger(const std::string& parser_model_path)
    : parser(parser_model_path) {}

  virtual ~LSTMCausalityTagger() {}

  lstm_parser::LSTMParser parser;
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
