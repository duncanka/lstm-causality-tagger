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
  // TODO: switch index list to set (no dupes). That would complicate the
  // algorithm for a SPLIT operation a bit.
  typedef std::vector<unsigned> IndexList;
  typedef std::vector<std::string> StringList;

  BecauseRelation(const BecauseRelation& other)
      : arg_names(other.arg_names), type_names(other.type_names),
        sentence(other.sentence), relation_type(other.relation_type),
        connective_indices(other.connective_indices),
        arguments(other.arguments) {}

  BecauseRelation(BecauseRelation&& other)
      : arg_names(other.arg_names), type_names(other.type_names),
        sentence(other.sentence), relation_type(other.relation_type),
        connective_indices(std::move(other.connective_indices)),
        arguments(std::move(other.arguments)) {}

  const lstm_parser::Sentence& GetSentence() const { return *sentence; }

  const IndexList& GetConnectiveIndices() const { return connective_indices; }

  IndexList* GetConnectiveIndices() { return &connective_indices; }

  void AddConnectiveToken(unsigned index) {
    connective_indices.push_back(index);
  }

  void AddToArgument(const unsigned arg_num, const unsigned index) {
    // Make sure index < max token ID (that isn't ROOT)
    assert(index <= (++sentence->words.rbegin())->first);
    arguments[arg_num].push_back(index);
  }

  const unsigned GetRelationType() const { return relation_type; }

  const IndexList& GetArgument(unsigned index) const {
    return arguments.at(index);
  }

  IndexList* GetArgument(unsigned index) {
    return &arguments.at(index);
  }

  template <typename IndexListType>
  void SetArgument(unsigned arg_num, const IndexListType& indices) {
    for (unsigned i : indices) {
      assert(i < sentence->Size());
    }
    arguments[arg_num] = indices;
  }

  void PrintTokens(std::ostream& os,
                   const BecauseRelation::IndexList& indices) const;

  static TokenList GetTokensForIndices(const lstm_parser::Sentence &sentence,
                                       const IndexList &indices) {
    TokenList tokens;
    tokens.reserve(indices.size());
    unsigned unk = sentence.vocab->GetWord(sentence.vocab->UNK);
    for (unsigned index : indices) {
      unsigned word_id = sentence.words.at(index);
      tokens.push_back(word_id == unk ? sentence.unk_surface_forms.at(index)
                                      : sentence.vocab->int_to_words[word_id]);
    }
    return tokens;
  }


protected:
  friend std::ostream& operator<<(std::ostream& os, const BecauseRelation& rel);

  BecauseRelation(const StringList& arg_names, const StringList& type_names,
                  const lstm_parser::Sentence& sentence,
                  const unsigned relation_type,
                  const IndexList& connective_indices,
                  const std::vector<IndexList>& arguments = {{}})
      : arg_names(arg_names), type_names(type_names), sentence(&sentence),
        relation_type(relation_type), connective_indices(connective_indices),
        arguments(arguments) {
    assert(arguments.size() < arg_names.size());
    // Make sure all arguments are created, even if they weren't yet specified.
    this->arguments.resize(arg_names.size());
  }

  TokenList GetTokensForIndices(const IndexList& indices) const {
    return GetTokensForIndices(*sentence, indices);
  }

  const std::string& ArgNameForArgIndex(unsigned arg_index) const {
    return arg_names[arg_index];
  }

  const std::string& GetRelationName(unsigned type_id = -1) const {
    if (type_id == static_cast<unsigned>(-1))
      type_id = relation_type;
    return type_names[type_id];
  }

  const StringList& arg_names;
  const StringList& type_names;

  const lstm_parser::Sentence* sentence;
  unsigned relation_type;
  IndexList connective_indices;
  std::vector<IndexList> arguments;
};


class CausalityRelation: public BecauseRelation {
public:
  // TODO: Is there a better way to sync an enum and a string list?
  static const StringList ARG_NAMES;
  enum ArgumentType {CAUSE = 0, EFFECT = 1, MEANS = 2};
  static const StringList TYPE_NAMES;
  enum CausationType {CONSEQUENCE = 0, MOTIVATION = 1, PURPOSE = 2};

  CausalityRelation(const lstm_parser::Sentence& sentence,
                    const unsigned relation_type = CONSEQUENCE,
                    const IndexList& connective_indices = {},
                    const std::vector<IndexList>& arguments = {{}})
      : BecauseRelation(ARG_NAMES, TYPE_NAMES, sentence, relation_type,
                        connective_indices, arguments) {}

  CausalityRelation(const CausalityRelation& other) = default;

  CausalityRelation& operator=(CausalityRelation&& other) {
    sentence = other.sentence;
    relation_type = other.relation_type;
    connective_indices = std::move(other.connective_indices);
    arguments = std::move(other.arguments);
    return *this;
  }

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
