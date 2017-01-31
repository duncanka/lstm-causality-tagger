#ifndef LSTM_CAUSALITY_BECAUSEDATA_H_
#define LSTM_CAUSALITY_BECAUSEDATA_H_

#include <string>
#include <vector>

#include "parser/corpus.h"


class BecauseRelation {
public:
  typedef std::vector<std::reference_wrapper<const std::string>> TokenList;
  typedef std::vector<unsigned> IndexList;
  typedef std::vector<std::string> StringList;

  const TokenList& GetSentenceTokens() const { return sentence_tokens; }

  const IndexList& GetConnectiveIndices() const { return connective_indices; }

  const unsigned GetRelationType() const { return relation_type; }

protected:
  const StringList& arg_names;
  const StringList& type_names;
  std::vector<IndexList> arguments;
  IndexList connective_indices;
  unsigned relation_type;
  TokenList sentence_tokens;

  friend std::ostream& operator<<(std::ostream& os, const BecauseRelation& rel);

  BecauseRelation(const StringList& arg_names, const StringList& type_names,
                    const IndexList& connective_indices,
                    const unsigned relation_type,
                    const std::vector<unsigned> known_word_ids,
                    const StringList& unknown_word_strs,
                    const lstm_parser::CorpusVocabulary& vocab) :
      arg_names(arg_names), type_names(type_names), arguments(arg_names.size()),
      connective_indices(connective_indices), relation_type(relation_type) {
    assert(known_word_ids.size() == unknown_word_strs.size());
    sentence_tokens.reserve(known_word_ids.size());
    unsigned unk = vocab.GetWord(vocab.UNK);
    for (size_t i = 0; i < known_word_ids.size(); ++i) {
      unsigned word_id = known_word_ids[i];
      if (word_id == unk) {
        sentence_tokens.push_back(unknown_word_strs[i]);
      } else {
        sentence_tokens.push_back(vocab.int_to_words[word_id]);
      }
    }
  }

  void SetArgument(unsigned arg_num, const IndexList& indices) {
    for (unsigned i : indices) {
      assert(i < sentence_tokens.size());
    }
    arguments[arg_num] = indices;
  }

  inline TokenList GetTokensForIndices(const IndexList& indices) const {
    TokenList tokens;
    tokens.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      tokens.push_back(sentence_tokens[indices[i]]);
    }
    return tokens;
  }

  inline const std::string& ArgNameForArgIndex(unsigned arg_index) const {
    return arg_names[arg_index];
  }

  inline const std::string& GetRelationName(unsigned type_id = -1) const {
    if (type_id == static_cast<unsigned>(-1))
      type_id = relation_type;
    return type_names[type_id];
  }
};


class CausalityRelation: public BecauseRelation {
public:
  static const StringList ARG_NAMES;
  static const StringList TYPE_NAMES;
  enum ArgumentTypes {CAUSE = 0, EFFECT = 1, MEANS = 2};

  inline void SetCause(const IndexList& indices) {
    SetArgument(CAUSE, indices);
  }

  inline void SetEffect(const IndexList& indices) {
    SetArgument(EFFECT, indices);
  }

  inline void SetMeans(const IndexList& indices) {
    SetArgument(MEANS, indices);
  }

  inline const IndexList& GetCause() { return arguments[CAUSE]; }

  inline const IndexList& GetEffect() { return arguments[EFFECT]; }

  inline const IndexList& GetMeans() { return arguments[MEANS]; }
};


#endif /* LSTM_CAUSALITY_BECAUSEDATA_H_ */
