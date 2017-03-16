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

void BecauseOracleTransitionCorpus::BecauseTransitionsReader::ReadSentences(
    const std::string& directory_path, Corpus* corpus) const {
  TrainingCorpus* training_corpus = static_cast<TrainingCorpus*>(corpus);

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
    for (auto a : training_corpus->vocab->actions) {
      cerr << a << "\n";
    }
    cerr << "# of actions: " << training_corpus->vocab->CountActions() << "\n";
  }

  training_corpus->sentences.shrink_to_fit();
  training_corpus->correct_act_sent.shrink_to_fit();
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

  while (getline(actions_file, line)) {
    if (line.empty()) { // An empty line marks the end of a sentence.
      if (!first) { // if first, first line is blank, but no sentence yet
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
    RecordSentence(training_corpus, &sentence, &sentence_pos,
                   &sentence_unk_surface_forms, &correct_actions);
  }
}
