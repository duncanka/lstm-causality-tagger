#ifndef LSTM_CAUSALITY_BECAUSEDATA_H_
#define LSTM_CAUSALITY_BECAUSEDATA_H_

#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "parser/corpus.h"

class BecauseRelation {
public:
  typedef std::map<unsigned, std::reference_wrapper<const std::string>>
      IndexedTokenList;
  typedef std::vector<std::reference_wrapper<const std::string>> TokenList;
  typedef std::vector<unsigned> IndexList; // TODO: switch to set (no dupes)
  typedef std::vector<std::string> StringList;

  const IndexList& GetConnectiveIndices() const { return connective_indices; }

  IndexList* GetConnectiveIndices() { return &connective_indices; }

  void AddConnectiveToken(unsigned index) {
    connective_indices.push_back(index);
  }

  void AddToArgument(const unsigned arg_num, const unsigned index) {
    assert(index <= sentence.words.rbegin()->first);  // index < max token ID
    arguments[arg_num].push_back(index);
  }

  const unsigned GetRelationType() const { return relation_type; }

  const IndexList& GetArgument(unsigned index) const {
    return arguments[index];
  }

  IndexList* GetArgument(unsigned index) {
    return &arguments[index];
  }

  void SetArgument(unsigned arg_num, const IndexList& indices) {
    for (unsigned i : indices) {
      assert(i < sentence.Size());
    }
    arguments[arg_num] = indices;
  }

protected:
  const StringList& arg_names;
  const StringList& type_names;

  const lstm_parser::Sentence& sentence;
  const lstm_parser::CorpusVocabulary& vocab;

  unsigned relation_type;
  IndexList connective_indices;
  std::vector<IndexList> arguments;

  friend std::ostream& operator<<(std::ostream& os, const BecauseRelation& rel);

  BecauseRelation(const StringList& arg_names, const StringList& type_names,
                  const lstm_parser::Sentence& sentence,
                  const lstm_parser::CorpusVocabulary& vocab,
                  const unsigned relation_type,
                  const IndexList& connective_indices,
                  const std::vector<IndexList>& arguments = {{}})
      : arg_names(arg_names), type_names(type_names), sentence(sentence),
        vocab(vocab), relation_type(relation_type),
        connective_indices(connective_indices), arguments(arguments) {
    assert(arguments.size() < arg_names.size());
    // Make sure all arguments are created, even if they weren't yet specified.
    this->arguments.resize(arg_names.size());
  }

  TokenList GetTokensForIndices(const IndexList& indices) const {
    TokenList tokens;
    tokens.reserve(indices.size());
    unsigned unk = vocab.GetWord(vocab.UNK);
    for (unsigned index : indices) {
      unsigned word_id = sentence.words.at(index);
      tokens.push_back(word_id == unk ? sentence.unk_surface_forms.at(index)
                                      : vocab.int_to_words[word_id]);
    }
    return tokens;
  }

  const std::string& ArgNameForArgIndex(unsigned arg_index) const {
    return arg_names[arg_index];
  }

  const std::string& GetRelationName(unsigned type_id = -1) const {
    if (type_id == static_cast<unsigned>(-1))
      type_id = relation_type;
    return type_names[type_id];
  }
};


class CausalityRelation: public BecauseRelation {
public:
  // TODO: Is there a better way to sync an enum and a string list?
  static const StringList ARG_NAMES;
  enum ArgumentType {CAUSE = 0, EFFECT = 1, MEANS = 2};
  static const StringList TYPE_NAMES;
  enum CausationType {CONSEQUENCE = 0, MOTIVATION = 1, PURPOSE = 3};

  CausalityRelation(const lstm_parser::Sentence& sentence,
                    const lstm_parser::CorpusVocabulary& vocab,
                    const unsigned relation_type = CONSEQUENCE,
                    const IndexList& connective_indices = {},
                    const std::vector<IndexList>& arguments = {{}})
      : BecauseRelation(ARG_NAMES, TYPE_NAMES, sentence, vocab, relation_type,
                        connective_indices, arguments) {}

  void SetCause(const IndexList& indices) {
    SetArgument(CAUSE, indices);
  }

  void SetEffect(const IndexList& indices) {
    SetArgument(EFFECT, indices);
  }

  void SetMeans(const IndexList& indices) {
    SetArgument(MEANS, indices);
  }

  const IndexList& GetCause() const { return GetArgument(CAUSE); }

  const IndexList& GetEffect() const { return GetArgument(EFFECT); }

  const IndexList& GetMeans() const { return GetArgument(MEANS); }

  IndexList* GetCause() { return GetArgument(CAUSE); }

  IndexList* GetEffect() { return GetArgument(EFFECT); }

  const IndexList* GetMeans() { return GetArgument(MEANS); }
};


#endif /* LSTM_CAUSALITY_BECAUSEDATA_H_ */
