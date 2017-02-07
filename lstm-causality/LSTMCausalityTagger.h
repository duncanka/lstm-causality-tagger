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

  std::vector<CausalityRelation> Tag(const lstm_parser::Sentence& sentence,
                                     const lstm_parser::CorpusVocabulary& vocab,
                                     double* correct = nullptr) {
    cnn::ComputationGraph cg;
    cnn::expr::Expression parser_state;
    parser.LogProbParser(sentence, parser.vocab, &cg, &parser_state);
    std::vector<unsigned> actions = LogProbTagger(
        &cg, sentence, std::vector<unsigned>(), vocab.actions,
        vocab.int_to_words, &parser_state, correct);
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
        const std::vector<std::string>& int_to_words,
        cnn::expr::Expression* parser_state, double* correct);

  std::vector<CausalityRelation> Decode(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned> actions,
      const lstm_parser::CorpusVocabulary& vocab);

  void SaveModel(const std::string& model_fname, bool softlink_created);
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
