#ifndef LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_
#define LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_

#include <string>

#include "BecauseData.h"
#include "BecauseOracleTransitionCorpus.h"
#include "cnn/model.h"
#include "parser/corpus.h"
#include "parser/lstm-parser.h"

class LSTMCausalityTagger {
public:
  explicit LSTMCausalityTagger(const std::string& parser_model_path)
    : parser(parser_model_path) {}

  virtual ~LSTMCausalityTagger() {}

  void Train(const BecauseOracleTransitionCorpus& corpus,
             const BecauseOracleTransitionCorpus& dev_corpus,
             const double unk_prob, const std::string& model_fname,
             const volatile bool* requested_stop = nullptr);

  std::vector<CausalityRelation> Tag(
      const lstm_parser::Sentence& sentence,
      const lstm_parser::CorpusVocabulary& vocab,
      double* correct = nullptr) {
    cnn::ComputationGraph hg;
    std::vector<unsigned> actions = LogProbTagger(
        &hg, sentence, std::vector<unsigned>(), vocab.actions,
        vocab.int_to_words, correct);
    return Decode(sentence, actions, vocab);
  }

  const lstm_parser::LSTMParser& GetParser() const { return parser; }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & model;
  }


protected:
  lstm_parser::LSTMParser parser;
  cnn::Model model;

  std::vector<unsigned> LogProbTagger(
        cnn::ComputationGraph* hg,
        const lstm_parser::Sentence& sentence,
        const std::vector<unsigned>& correct_actions,
        const std::vector<std::string>& action_names,
        const std::vector<std::string>& int_to_words, double* correct);

  std::vector<CausalityRelation> Decode(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned> actions,
      const lstm_parser::CorpusVocabulary& vocab);

  void SaveModel(const std::string& model_fname, bool softlink_created);
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
