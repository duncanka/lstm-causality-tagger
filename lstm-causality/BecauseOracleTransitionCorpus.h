#ifndef BECAUSEORACLETRANSITIONCORPUS_H_
#define BECAUSEORACLETRANSITIONCORPUS_H_

#include <memory>
#include <string>

#include "parser/corpus.h"
#include "parser/lstm-parser.h"

class BecauseOracleTransitionCorpus: public lstm_parser::TrainingCorpus {
public:
  BecauseOracleTransitionCorpus(lstm_parser::CorpusVocabulary* vocab,
                                const std::string& file, bool is_training)
      : TrainingCorpus(vocab) {
    BecauseTransitionsReader(is_training).ReadSentences(file, this);
    sentence_parses.resize(sentences.size());
  }

  // Cache of sentence parses for evaluation purposes
  std::vector<std::unique_ptr<lstm_parser::ParseTree>> sentence_parses;

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
};

#endif /* BECAUSEORACLETRANSITIONCORPUS_H_ */
