#ifndef BECAUSEORACLETRANSITIONCORPUS_H_
#define BECAUSEORACLETRANSITIONCORPUS_H_

#include <string>

#include "parser/corpus.h"

class BecauseOracleTransitionCorpus: public lstm_parser::TrainingCorpus {
public:
  BecauseOracleTransitionCorpus(lstm_parser::CorpusVocabulary* vocab,
                                const std::string& file, bool is_training)
      : TrainingCorpus(vocab) {
    BecauseTransitionsReader(is_training).ReadSentences(file, this);
  }
  virtual ~BecauseOracleTransitionCorpus() {}

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
