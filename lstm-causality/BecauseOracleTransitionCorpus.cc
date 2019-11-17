#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <cstddef>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>

#include "BecauseOracleTransitionCorpus.h"
#include "utilities.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace lstm_parser;

const vector<string> BecauseOracleTransitionCorpus::PTB_PUNCT_TAGS = {
    ".", ",", ":", "``", "''", "-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-",
    "-RSB-"
};

const vector<string> BecauseOracleTransitionCorpus::PTB_VERB_TAGS = {
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"
};

const vector<string> BecauseOracleTransitionCorpus::PTB_NOUN_TAGS = {
    "NN", "NP", "NNS", "NNP", "NNPS", "PRP", "WP", "WDT"
};

const vector<string> BecauseOracleTransitionCorpus::INCOMING_CLAUSE_EDGES = {
    "ccomp", "xcomp", "csubj", "csubjpass", "advcl", "acl", "acl:relcl"
};

const vector<string> GraphEnhancedParseTree::SUBJECT_EDGE_LABELS = {
  "nsubj", "csubj", "nsubjpass", "csubjpass"
};


void GraphEnhancedParseTree::BuildAndAnalyzeGraph() {
  using namespace boost;
  // ROOT is (unsigned)-1, so it shows up last. We definitely don't want such
  // a big graph, though, so we just store it as 0 in the graph, CoNLL-style,
  // and skip it below.
  const string& root_arc_label = arc_labels->at(root_child);
  add_edge(0, root_child, {root_arc_label, 1.0}, sentence_graph);

  // Adjust edge weights to make better paths preferred.
  vector<unsigned> xcomp_children;
  vector<unsigned> subjects;
  for (const auto& child_and_parent : parents) {
    unsigned child, parent;
    boost::tie(child, parent) = child_and_parent;
    if (child != root_child) {
      float weight = 1.0;
      const string& arc_label = arc_labels->at(child);
      if (arc_label == "xcomp") {
        weight = 0.98;
        xcomp_children.push_back(child);
      } else if (arc_label == "expl" || boost::starts_with(arc_label, "acl")) {
        weight = 1.01;
      } else if (Contains(SUBJECT_EDGE_LABELS, arc_label)) {
        subjects.push_back(child);
      }
      add_edge(parent, child, {arc_label, weight}, sentence_graph);
    }
  }

  // We still need to adjust the weights on subject children of xcomps.
  for (const auto& child_and_parent : parents) {
    unsigned child, parent;
    boost::tie(child, parent) = child_and_parent;
    if (Contains(xcomp_children, parent) && Contains(subjects, child)) {
      Graph::edge_descriptor e = edge(parent, child, sentence_graph).first;
      sentence_graph[e].weight = 0.985;
    }
  }

  ComputeDepthsAndShortestPaths();
}


void GraphEnhancedParseTree::ComputeDepthsAndShortestPaths() {
  // First compute unweighted depths from ROOT using BFS.
  boost::associative_property_map<map<Vertex, unsigned>> token_depths_map(
      token_depths);
  auto depth_visitor = boost::make_bfs_visitor(boost::record_distances(
      token_depths_map, boost::on_tree_edge()));
  breadth_first_search(sentence_graph, vertex(0, sentence_graph),
                       boost::visitor(depth_visitor));

  // For shortest paths, we want to allow following arcs in both directions, so
  // we create artificial back-edges for each one-way arc.
  Graph pseudo_unweighted_graph = sentence_graph;
  vector<Graph::edge_descriptor> edges_to_replicate;
  for (const auto& e : boost::make_iterator_range(
      edges(pseudo_unweighted_graph))) {
    bool reverse_exists = edge(boost::target(e, pseudo_unweighted_graph),
                               boost::source(e, pseudo_unweighted_graph),
                               pseudo_unweighted_graph).second;
    if (!reverse_exists) {
      edges_to_replicate.push_back(e);
    }
  }
  // Do the actual adding after to avoid iterator stability issues.
  for (const auto& e : edges_to_replicate) {
    boost::add_edge(boost::target(e, pseudo_unweighted_graph),
                    boost::source(e, pseudo_unweighted_graph),
                    {"", pseudo_unweighted_graph[e].weight},
                    pseudo_unweighted_graph);
  }

  // Fill predecessors with -1's, since there will be some tokens that are not
  // part of the graph at all.
  std::fill(path_predecessors.origin(),
            path_predecessors.origin() + path_predecessors.num_elements(), -1);
  /*
  const lstm_parser::Sentence& s = sentence.get();
  cerr << "Sentence: " << s << endl;
  //*/
  // Unfortunately, the Boost all-pairs shortest path algorithms don't allow
  // saving the predecessors, so we'll just do iterated Dijkstra (which anyway
  // is just as fast for this graph, probably).
  // TODO: for ROOT, don't bother with Dijkstra -- just use this->parents.
  auto all_vertices = boost::vertices(pseudo_unweighted_graph);
  for (const Vertex& start : boost::make_iterator_range(all_vertices)) {
    auto predecessor_map = boost::make_iterator_property_map(
        path_predecessors[start].begin(), boost::identity_property_map());
    auto weight_map = get(&ArcInfo::weight, pseudo_unweighted_graph);
    boost::dijkstra_shortest_paths(
        pseudo_unweighted_graph, start,
        boost::weight_map(weight_map).predecessor_map(predecessor_map));
    /*
    for (unsigned child = 0; child < GetGraphSize(); ++child) {
      unsigned predecessor = path_predecessors[start][child];
      cerr << "On path from "
           << s.vocab.int_to_words.at(s.words.at(ConvertRoot(start, true)))
           << ": set parent of "
           << s.vocab.int_to_words.at(s.words.at(ConvertRoot(child, true)))
           << " to " << s.vocab.int_to_words.at(s.words.at(
                            ConvertRoot(predecessor, true))) << endl;
    }
    //*/
  }
}


vector<GraphEnhancedParseTree::ParsePathLink>
    GraphEnhancedParseTree::GetParsePath(unsigned source, unsigned dest) const {
  source = ConvertRoot(source);
  dest = ConvertRoot(dest);
  vector<ParsePathLink> path;
  Graph::edge_descriptor edge;
  bool is_forward_edge;
  for (unsigned predecessor = dest; predecessor != source;
       dest = predecessor) {
    predecessor = ConvertRoot(path_predecessors[source][dest]);
    if (predecessor == static_cast<unsigned>(-1)) {
      assert(path.empty());
      return path;
    }
    tie(edge, is_forward_edge) =
        boost::edge(predecessor, dest, sentence_graph);
    if (!is_forward_edge) {
      edge = boost::edge(dest, predecessor, sentence_graph).first;
    }
    path.push_back({predecessor, dest, sentence_graph[edge].dep_label,
                    is_forward_edge});
  }
  return path;
}


BecauseOracleTransitionCorpus::BecauseOracleTransitionCorpus(
    CorpusVocabulary* vocab, const string& file, bool is_training,
    bool sort_by_sentence)
    : TrainingCorpus(vocab) {
  BecauseTransitionsReader(is_training, &filenames).ReadSentences(file, this);
  sentence_parses.resize(sentences.size());

  if (sort_by_sentence) {
    // Reconstruct the sentence strings.
    vector<string> sentence_texts;
    sentence_texts.reserve(sentences.size());
    for (const Sentence& sentence : sentences) {
      ostringstream os;
      for (const auto& index_and_word : sentence.words) {
        unsigned word = index_and_word.second;
        if (word == vocab->kUNK) {
          cerr << "Can't sort on unknown word" << endl;
          abort();
        }
        os << vocab->int_to_words.at(word) << ' ';
      }
      // Normalize quotes
      string sentence_text = boost::replace_all_copy(os.str(), "''", "\"");
      boost::replace_all(sentence_text, "``", "\"");
      // TODO: deal with fractions and ellipses, which could also affect sort
      // order but don't in practice?
      sentence_texts.push_back(sentence_text);
    }
    // Now get the sort order of those strings and apply it to the sentences.
    vector<size_t> sort_order = SortIndices(sentence_texts);
    Reorder(&sentences, sort_order);
    Reorder(&correct_act_sent, sort_order);
  }
}



void BecauseOracleTransitionCorpus::BecauseTransitionsReader::ReadSentences(
    const string& directory_path, Corpus* corpus) const {
  BecauseOracleTransitionCorpus* training_corpus =
      static_cast<BecauseOracleTransitionCorpus*>(corpus);

  cerr << "Loading " << (is_training ? "training" : "dev")
       << " corpus from " << directory_path << "..." << flush;
  for (fs::recursive_directory_iterator file_iter(directory_path), end;
       file_iter != end; ++file_iter) {
    const fs::path& file_path = file_iter->path();
    if (fs::is_directory(file_path) || file_path.extension() != FILE_EXTENSION)
      continue;
    // cerr << "Loading from file " << file_path << endl;
    ReadFile(file_path.string(), training_corpus);
  }

  cerr << "done." << "\n";
  if (is_training) {
    for (auto a : training_corpus->vocab->action_names) {
      cerr << a << "\n";
    }
    cerr << "# of actions: " << training_corpus->vocab->CountActions() << "\n";
  }

  training_corpus->sentences.shrink_to_fit();
  training_corpus->correct_act_sent.shrink_to_fit();

  // Remember which POS tag encodings have what statuses (mainly used in
  // calculation of metrics)
  training_corpus->pos_is_punct.resize(training_corpus->vocab->CountPOS());
  training_corpus->pos_is_non_modal_verb.resize(
      training_corpus->vocab->CountPOS());
  training_corpus->pos_is_noun.resize(training_corpus->vocab->CountPOS());
  for (const auto& pos_entry : training_corpus->vocab->pos_to_int) {
    training_corpus->pos_is_punct[pos_entry.second] =
        boost::algorithm::any_of_equal(PTB_PUNCT_TAGS, pos_entry.first);
    training_corpus->pos_is_non_modal_verb[pos_entry.second] =
        pos_entry.first == "MD"
            ? false
            : boost::algorithm::any_of_equal(PTB_VERB_TAGS, pos_entry.first);
    training_corpus->pos_is_noun[pos_entry.second] =
        boost::algorithm::any_of_equal(PTB_NOUN_TAGS, pos_entry.first);
  }
}


void BecauseOracleTransitionCorpus::BecauseTransitionsReader::ReadFile(
    const string& file_name, TrainingCorpus* corpus) const {
  enum LineType {
    SENTENCE_START_LINE, STATE_LINE, RELS_LINE, ARC_LINE
  };
  BecauseOracleTransitionCorpus* training_corpus =
      static_cast<BecauseOracleTransitionCorpus*>(corpus);

  ifstream actions_file(file_name);
  filenames->emplace_front(
      fs::path(file_name).replace_extension("ann").string());
  const string& ann_file_name = filenames->front();
  string line;

  LineType next_line_type = SENTENCE_START_LINE;
  bool first = true;

  map<unsigned, unsigned> sentence;
  map<unsigned, unsigned> sentence_pos;
  map<unsigned, string> sentence_unk_surface_forms;
  vector<unsigned> correct_actions;
  unsigned document_byte_offset;

  unsigned root_symbol = corpus->vocab->GetOrAddWord(CorpusVocabulary::ROOT);
  unsigned root_pos_symbol = corpus->vocab->GetOrAddEntry(
      CorpusVocabulary::ROOT, &corpus->vocab->pos_to_int,
      &corpus->vocab->int_to_pos);

  while (getline(actions_file, line)) {
    if (boost::starts_with(line, ">>>")) { // extrasententials line
      istringstream line_stream(line);
      line_stream.get(); line_stream.get(); // skip the >>>
      unsigned missing_instances;
      line_stream >> missing_instances;
      training_corpus->missing_instance_counts.push_back(missing_instances);

      string missing_args_str;
      vector<ExtrasententialArgCounts> sentence_missing_args;
      while(getline(line_stream, missing_args_str, ' ')) {
        if (!missing_args_str.empty()) {
          size_t slash_pos_1 = missing_args_str.find('/');
          size_t slash_pos_2 = missing_args_str.find('/', slash_pos_1 + 1);
          assert(missing_args_str.find('/', slash_pos_2 + 1) == string::npos);
          sentence_missing_args.push_back({
            boost::lexical_cast<unsigned>(missing_args_str.data(), slash_pos_1),
            boost::lexical_cast<unsigned>(
                missing_args_str.data() + slash_pos_1 + 1,
                slash_pos_2 - (slash_pos_1 + 1)),
            boost::lexical_cast<unsigned>(
                missing_args_str.c_str() + slash_pos_2 + 1)
          });
        }
      }
      training_corpus->missing_arg_tokens.push_back(sentence_missing_args);

      getline(actions_file, line); // line break between sentences
      assert(line.empty());

      if (!first) { // if first, first line is blank, but no sentence yet
        sentence[Corpus::ROOT_TOKEN_ID] = root_symbol;
        sentence_pos[Corpus::ROOT_TOKEN_ID] = root_pos_symbol;
        sentence_unk_surface_forms[Corpus::ROOT_TOKEN_ID] = "";
        RecordSentence(
            training_corpus, &sentence, &sentence_pos,
            &sentence_unk_surface_forms, &correct_actions,
            new BecauseSentenceMetadata(ann_file_name, document_byte_offset));
      }
      next_line_type = SENTENCE_START_LINE;
      continue; // don't update next_line_type again
    }
    first = false;

    if (next_line_type == SENTENCE_START_LINE) {
      // the initial line in each sentence should look like:
      // 0 the/DT, cat/NN, is/VB, on/PRP, the/DT, mat/NN, .-.
      istringstream iss(line);
      iss >> document_byte_offset;
      do {
        string word;
        iss >> word;
        if (word.size() == 0)
          continue;
        // Remove the trailing comma, if need be.
        if (*word.rbegin() == ',')
          word = word.substr(0, word.size() - 1);
        // Split the string into word and POS tag.
        size_t pos_index = word.rfind(POS_SEPARATOR);
        assert(pos_index != string::npos);
        string pos = word.substr(pos_index + 1);
        word = word.substr(0, pos_index);

        // Use 1-indexed token IDs to leave room for ROOT in position 0. (We
        // don't use ROOT, but this makes it easier to match up to the syntactic
        // parse.)
        unsigned next_token_index = sentence.size() + 1;
        RecordWord(word, pos, next_token_index, training_corpus,
                   &sentence, &sentence_pos, &sentence_unk_surface_forms);
      } while(iss);
      next_line_type = STATE_LINE;
    } else if (next_line_type == STATE_LINE) {
      // Ignore state line. TODO: check that our internal state matches?
      next_line_type = RELS_LINE;
    } else if (next_line_type == RELS_LINE) {
      // Ignore relations line. TODO: check that our internal state matches?
      next_line_type = ARC_LINE;
    } else { // next_line_type == ARC_LINE
      RecordAction(line, training_corpus, &correct_actions);
      next_line_type = STATE_LINE;
    }
  }

  if (!sentence.empty()) {
    sentence[Corpus::ROOT_TOKEN_ID] = root_symbol;
    sentence_pos[Corpus::ROOT_TOKEN_ID] = root_pos_symbol;
    sentence_unk_surface_forms[Corpus::ROOT_TOKEN_ID] = "";
    RecordSentence(
        training_corpus, &sentence, &sentence_pos, &sentence_unk_surface_forms,
        &correct_actions,
        new BecauseSentenceMetadata(ann_file_name, document_byte_offset));
  }
}
