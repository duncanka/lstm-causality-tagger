#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <cstddef>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <sstream>
#include <string>

#include "BecauseOracleTransitionCorpus.h"

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



class DepthRecorder : public boost::default_bfs_visitor {
public:
  DepthRecorder(map<unsigned, unsigned>* depths) : depths(depths) {}

  template<typename Edge, typename Graph>
  void tree_edge(const Edge& e, const Graph& g) const {
    unsigned parent = boost::source(e, g);
    unsigned child = boost::target(e, g);
    // On the first access of the source, the map access should initialize its
    // distance to be 0.
    (*depths)[child] = (*depths)[parent] + 1;
  }

  map<unsigned, unsigned>* depths;
};


void GraphEnhancedParseTree::MakeGraphAndCalculateDepths() {
  using namespace boost;
  // ROOT is (unsigned)-1, so it shows up last. We definitely don't want such
  // a big graph, though, so we just store it as 0 in the graph, CoNLL-style,
  // and skip it below.
  const string& root_arc_label = arc_labels->at(root_child);
  add_edge(0, root_child, root_arc_label, sentence_graph);
  for (const auto& child_and_parent : parents) {
    unsigned child, parent;
    boost::tie(child, parent) = child_and_parent;
    if (child != root_child) {
      const string& arc_label = arc_labels->at(child);
      add_edge(parent, child, arc_label, sentence_graph);
    }
  }
  DepthRecorder depths_visitor(&token_depths);
  breadth_first_search(sentence_graph,
                       vertex(0, sentence_graph),
                       visitor(depths_visitor));
}


void BecauseOracleTransitionCorpus::BecauseTransitionsReader::ReadSentences(
    const std::string& directory_path, Corpus* corpus) const {
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
  TrainingCorpus* training_corpus = static_cast<TrainingCorpus*>(corpus);

  ifstream actions_file(file_name);
  string line;

  LineType next_line_type = SENTENCE_START_LINE;
  bool first = true;

  map<unsigned, unsigned> sentence;
  map<unsigned, unsigned> sentence_pos;
  map<unsigned, string> sentence_unk_surface_forms;
  vector<unsigned> correct_actions;

  unsigned root_symbol = corpus->vocab->GetOrAddWord(CorpusVocabulary::ROOT);
  unsigned root_pos_symbol = corpus->vocab->GetOrAddEntry(
      CorpusVocabulary::ROOT, &corpus->vocab->pos_to_int,
      &corpus->vocab->int_to_pos);

  while (getline(actions_file, line)) {
    if (line.empty()) { // An empty line marks the end of a sentence.
      if (!first) { // if first, first line is blank, but no sentence yet
        sentence[Corpus::ROOT_TOKEN_ID] = root_symbol;
        sentence_pos[Corpus::ROOT_TOKEN_ID] = root_pos_symbol;
        sentence_unk_surface_forms[Corpus::ROOT_TOKEN_ID] = "";
        RecordSentence(training_corpus, &sentence, &sentence_pos,
                       &sentence_unk_surface_forms, &correct_actions);
      }
      next_line_type = SENTENCE_START_LINE;
      continue; // don't update next_line_type again
    }
    first = false;

    if (next_line_type == SENTENCE_START_LINE) {
      // the initial line in each sentence should look like:
      // the/DT, cat/NN, is/VB, on/PRP, the/DT, mat/NN, .-.
      istringstream iss(line);
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
    RecordSentence(training_corpus, &sentence, &sentence_pos,
                   &sentence_unk_surface_forms, &correct_actions);
  }
}
