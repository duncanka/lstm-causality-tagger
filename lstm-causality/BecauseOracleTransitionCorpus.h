#ifndef BECAUSEORACLETRANSITIONCORPUS_H_
#define BECAUSEORACLETRANSITIONCORPUS_H_

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <map>
#include <memory>
#include <string>

#include "parser/corpus.h"
#include "parser/lstm-parser.h"


class GraphEnhancedParseTree : public lstm_parser::ParseTree {
public:
  GraphEnhancedParseTree(ParseTree&& tree)
      : ParseTree(std::move(tree)), sentence_graph(GetHighestNonRootTokenID()) {
    MakeGraphAndCalculateDepths();
  }

  GraphEnhancedParseTree(const ParseTree& tree)
      : ParseTree(tree), sentence_graph(GetHighestNonRootTokenID()) {
    MakeGraphAndCalculateDepths();
  }

  unsigned GetTokenDepth(unsigned token_id) const {
    return token_depths.at(token_id);
  }

  auto GetChildren(unsigned token_id) const {
    if (token_id == lstm_parser::Corpus::ROOT_TOKEN_ID)
      token_id = 0;
    auto get_child_from_edge = [&](const Graph::edge_descriptor& edge) {
      return boost::target(edge, sentence_graph);
    };
    return boost::out_edges(token_id, sentence_graph)
        | boost::adaptors::transformed(get_child_from_edge);
  }

protected:
  typedef boost::adjacency_list<boost::vecS, boost::vecS,
                                boost::bidirectionalS> Graph;
  std::map<unsigned, unsigned> token_depths;
  Graph sentence_graph;

  void MakeGraphAndCalculateDepths();

  unsigned GetHighestNonRootTokenID() {
    return GetSentence().words.rbegin()->first + 1;
  }
};

class BecauseOracleTransitionCorpus: public lstm_parser::TrainingCorpus {
public:
  friend class LSTMCausalityTagger;
  const static std::vector<std::string> PTB_PUNCT_TAGS;
  const static std::vector<std::string> PTB_VERB_TAGS;
  const static std::vector<std::string> PTB_NOUN_TAGS;
  const static std::vector<std::string> INCOMING_CLAUSE_EDGES;

  BecauseOracleTransitionCorpus(lstm_parser::CorpusVocabulary* vocab,
                                const std::string& file, bool is_training)
      : TrainingCorpus(vocab) {
    BecauseTransitionsReader(is_training).ReadSentences(file, this);
    sentence_parses.resize(sentences.size());
  }

  // Cache of sentence parses for evaluation purposes
  std::vector<std::unique_ptr<GraphEnhancedParseTree>> sentence_parses;
  std::vector<bool> pos_is_punct;
  std::vector<bool> pos_is_non_modal_verb;
  std::vector<bool> pos_is_noun;

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
