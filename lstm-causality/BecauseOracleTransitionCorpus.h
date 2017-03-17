#ifndef BECAUSEORACLETRANSITIONCORPUS_H_
#define BECAUSEORACLETRANSITIONCORPUS_H_

#include <map>
#include <memory>
#include <string>

#include "parser/corpus.h"
#include "parser/lstm-parser.h"


class GraphEnhancedParseTree : public lstm_parser::ParseTree {
public:
  GraphEnhancedParseTree(ParseTree&& tree) : ParseTree(std::move(tree)) {
    CalculateDepths();
  }

  GraphEnhancedParseTree(const ParseTree& tree) : ParseTree(tree) {
    CalculateDepths();
  }

  unsigned GetTokenDepth(unsigned token_id) const {
    return token_depths.at(token_id);
  }

protected:
  std::map<unsigned, unsigned> token_depths;

  void CalculateDepths();
};

class BecauseOracleTransitionCorpus: public lstm_parser::TrainingCorpus {
public:
  friend class LSTMCausalityTagger;
  const static std::vector<std::string> PTB_PUNCT_TAGS;

  BecauseOracleTransitionCorpus(lstm_parser::CorpusVocabulary* vocab,
                                const std::string& file, bool is_training)
      : TrainingCorpus(vocab) {
    BecauseTransitionsReader(is_training).ReadSentences(file, this);
    sentence_parses.resize(sentences.size());
  }

  // Cache of sentence parses for evaluation purposes
  std::vector<std::unique_ptr<GraphEnhancedParseTree>> sentence_parses;

private:
  class BecauseTransitionsReader: public OracleTransitionsCorpusReader {
  public:
    static constexpr const char* FILE_EXTENSION = ".trans";
    static constexpr const char POS_SEPARATOR = '/';

    BecauseTransitionsReader(bool is_training)
      : OracleTransitionsCorpusReader(is_training) {}

    virtual void ReadSentences(const std::string& directory,
                               Corpus* corpus) const;

  protected:
    inline void ReadFile(const std::string& file_name,
                         TrainingCorpus* corpus) const;
  };

  std::vector<bool> pos_is_punct;
};

#endif /* BECAUSEORACLETRANSITIONCORPUS_H_ */
