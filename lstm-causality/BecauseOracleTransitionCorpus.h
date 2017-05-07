#ifndef BECAUSEORACLETRANSITIONCORPUS_H_
#define BECAUSEORACLETRANSITIONCORPUS_H_

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/multi_array.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <map>
#include <memory>
#include <string>

#include "parser/corpus.h"
#include "parser/lstm-parser.h"


class GraphEnhancedParseTree : public lstm_parser::ParseTree {
public:
  struct ArcInfo {
    std::string dep_label;
    float weight;
  };
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                                boost::no_property, ArcInfo> Graph;
  typedef Graph::vertex_descriptor Vertex;
  typedef boost::multi_array<typename Graph::vertex_descriptor, 2>
      PredecessorMatrix;

  struct ParsePathLink {
    unsigned start;
    unsigned end;
    const std::string& arc_label;
    bool reversed;
  };

  static const std::vector<std::string> SUBJECT_EDGE_LABELS;

  GraphEnhancedParseTree(const lstm_parser::Sentence& sentence)
      : ParseTree(sentence, false), sentence_graph(GetGraphSize()),
        path_predecessors(boost::array<unsigned, 2>{GetGraphSize(),
                                                    GetGraphSize()}) {}

  GraphEnhancedParseTree(ParseTree&& tree)
      : ParseTree(std::move(tree)), sentence_graph(GetGraphSize()),
        path_predecessors(boost::array<unsigned, 2>{GetGraphSize(),
                                                    GetGraphSize()}) {
    BuildAndAnalyzeGraph();
  }

  GraphEnhancedParseTree(const ParseTree& tree)
      : ParseTree(tree), sentence_graph(GetGraphSize()),
        path_predecessors(boost::array<unsigned, 2>{GetGraphSize(),
                                                    GetGraphSize()}) {
    BuildAndAnalyzeGraph();
  }

  unsigned GetTokenDepth(unsigned token_id) const {
    auto depth_iter = token_depths.find(token_id);
    if (depth_iter == token_depths.end())
      return -1;
    return depth_iter->second;
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

  std::vector<ParsePathLink> GetParsePath(unsigned source, unsigned dest) const;

protected:
  Graph sentence_graph;
  std::map<Vertex, unsigned> token_depths;

  // Row i of the predecessor matrix contains information on the shortest paths
  // from point i: each entry predecessors[i][j] gives the index of the previous
  // node in the path from point i to point j. If no path exists between point i
  // and j, then predecessors[i][j] = -1.
  PredecessorMatrix path_predecessors;

  void BuildAndAnalyzeGraph();

  void ComputeDepthsAndShortestPaths();

  unsigned GetGraphSize() const {
    // To determine the size of the graph, we find the index of the last token
    // that isn't ROOT. We still have to add 1, because that will be the token's
    // index in the graph, and the graph is 0-indexed.
    return (++GetSentence().words.rbegin())->first + 1;
  }

  unsigned ConvertRoot(unsigned token_id, bool from_graph = false) const {
    if (from_graph)
      return token_id == 0 ? lstm_parser::Corpus::ROOT_TOKEN_ID : token_id;
    else
      return token_id == lstm_parser::Corpus::ROOT_TOKEN_ID ? 0 : token_id;
  };

};

class BecauseOracleTransitionCorpus: public lstm_parser::TrainingCorpus {
public:
  friend class LSTMCausalityTagger;
  const static std::vector<std::string> PTB_PUNCT_TAGS;
  const static std::vector<std::string> PTB_VERB_TAGS;
  const static std::vector<std::string> PTB_NOUN_TAGS;
  const static std::vector<std::string> INCOMING_CLAUSE_EDGES;

  struct ExtrasententialArgs {
    unsigned cause_tokens;
    unsigned effect_tokens;
    unsigned means_tokens;
  };

  BecauseOracleTransitionCorpus(lstm_parser::CorpusVocabulary* vocab,
                                const std::string& file, bool is_training)
      : TrainingCorpus(vocab) {
    BecauseTransitionsReader(is_training).ReadSentences(file, this);
    sentence_parses.resize(sentences.size());
  }

  std::vector<unsigned> missing_instance_counts;
  std::vector<std::vector<ExtrasententialArgs>> missing_arg_tokens;

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
