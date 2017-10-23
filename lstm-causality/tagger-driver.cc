#include <algorithm>
#include <array>
#include <boost/range/adaptors.hpp>
#include <boost/range/join.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sstream>
#include <unistd.h>
#include <vector>

#include "cnn/cnn/init.h"
#include "BecauseOracleTransitionCorpus.h"
#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace lstm_parser;
namespace ad = boost::adaptors;
namespace po = boost::program_options;
using namespace std;

volatile sig_atomic_t requested_stop = false;

po::typed_value<bool>* POBooleanFlag(bool default_val) {
  return po::value<bool>()->default_value(default_val)->implicit_value(true);
}

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    // Basic operation options
    ("help", "Print these usage instructions and exit")
    ("parser-model,M",
      po::value<string>()->default_value(
          "lstm-parser/english_pos_2_32_100_20_100_12_20.params"),
     "File from which to load saved syntactic parser model")
    ("model-dir", po::value<string>()->default_value("models"),
     "Directory in which to store saved model")

    // Data options
    ("training-data,t", po::value<string>(),
     "Directory containing training data")
    ("new-conn-action,n", POBooleanFlag(true),
     "Whether starting a relation is a separate action (must match data)")
    ("shift-action,H", POBooleanFlag(false),
     "Whether completing a relation is a separate action (must match data)")

    // Testing/evaluation options
    ("folds,f", po::value<unsigned>()->default_value(20),
     "How many folds to split the data into for cross-validation")
    ("eval-pairwise,P", POBooleanFlag(true),
     "Whether to also evaluate on just instances with both cause and effect")
    ("compare-punct,c", POBooleanFlag(false),
     "Whether to count punctuation when comparing argument spans")
    ("log-diffs,L", POBooleanFlag(false),
     "Whether to log differences between correct and predicted")
    ("for-comparison,C", POBooleanFlag(false),
     "Whether we'll be comparing outputs against a system with different"
     " randomization (prints order of folds and test sentences, logs"
     " differences as TSV format, and forces the corpus to sort its sentences"
     " alphabetically for easier sentence index comparison)")
    ("known-conns-only,K", POBooleanFlag(false),
     "Whether to restrict the possible transitions at test time to allow only"
     " known connectives")
    ("oracle-conns,o", POBooleanFlag(false),
     "Whether to use oracle NEW-CONN, CONN-FRAG, and SPLIT transitions at test"
     " time. Valid only in conjunction with --new-conn-action.")
    ("cv-start-at", po::value<unsigned>()->default_value(1),
     "Cross-validation fold to start at (useful for debugging/parallelizing")
    ("cv-end-at", po::value<unsigned>()->default_value(-1),
     "Cross-validation fold to end on (useful for debugging/parallelizing")

    // Training options
    ("train", "Whether to train the tagger")
    ("dev-pct,d", po::value<double>()->default_value(0.2),
     "Percent of training data in each shuffle to use as dev (tuning)")
    ("train-pairwise,r", POBooleanFlag(false),
     "Whether to train on just instances with both cause and effect")
    ("dropout,D", po::value<float>()->default_value(0.0),
     "Dropout rate (no dropout is performed for a value of 0)")
    ("dev-eval-period,E", po::value<unsigned>()->default_value(25),
     "How many training iterations to go between dev evaluations")
    ("epochs-cutoff,e", po::value<double>()->default_value(5),
     "Number of training epochs without an improvement in the best F1 to allow"
     " before stopping training (SIGINT always works to stop; see also"
     " --recent-improvements-cutoff)")
    ("recent-improvements-cutoff,I", po::value<double>()->default_value(0.85),
     "Don't stop training yet if this % of evaluations in last --epochs-cutoff"
     " epochs have been an increase from the immediately prior evaluation")
    ("recent-improvements-epsilon,N", po::value<double>()->default_value(0.005),
     "How much better a dev evaluation must be than the previous one to be"
     " considered a score increase for termination purposes")

    // Network dimensionality/structure options
    ("action-dim,a", po::value<unsigned>()->default_value(8),
     "Dimension for vector representation of actions. If 0, action history is"
     " not used.")
    ("word-dim,w", po::value<unsigned>()->default_value(10),
     "Dimension for task-specific vector representation of words. If 0, no"
     " task-specific word vectors are created.")
    ("state-dim,s", po::value<unsigned>()->default_value(72),
     "Dimension for overall tagger state")
    ("conn-in-state,G", POBooleanFlag(false),
     "Whether to include the current connective token as its own state input")
    ("token-dim,i", po::value<unsigned>()->default_value(48),
     "Dimension for representation of tokens, e.g., as span/lambda LSTM inputs")
    ("lambda-hidden-dim,h", po::value<unsigned>()->default_value(48),
     "Dimension of the hidden state of each lambda LSTM")
    ("actions-hidden-dim,A", po::value<unsigned>()->default_value(32),
     "Dimension of the hidden state of the action history LSTM")
    ("parse-path-arc-dim,R", po::value<unsigned>()->default_value(0),
     "Dimension for embedding parse relation + POS + word vector for input to"
     " parse path. If 0, the raw parse relation embedding is used.")
    ("parse-path-hidden-dim,p", po::value<unsigned>()->default_value(20),
     "Dimension of the hidden state of the parse path embedding LSTM. If 0,"
     " parse paths are not used.")
    ("span-hidden-dim,S", po::value<unsigned>()->default_value(32),
     "Dimension of each connective/argument span LSTM's hidden state")
    ("lstm-layers,l", po::value<unsigned>()->default_value(2),
     "Number of layers for each stack LSTM")
    ("subtrees,u", POBooleanFlag(false),
     "Whether to include embeddings of parse subtrees in token representations")
    ("gated-parse,g", POBooleanFlag(false),
     "Whether to include gated parse tree embedding in the overall state");

  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    std::exit(0);
  }
}


void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again; quitting.\n";
    std::exit(1);
  }
  cerr << "\nReceived SIGINT. Terminating optimization early...\n";
  requested_stop = true;
}

const string GetModelFileName(const LSTMCausalityTagger& tagger,
                              const string& model_dir) {
  ostringstream os;
  if (!model_dir.empty()) {
    os << model_dir;
    os << '/';
  }
  // TODO: update with latest options?
  os << "tagger_" << tagger.options.word_dim
     << '_' << tagger.options.lstm_layers
     << '_' << tagger.options.token_dim
     << '_' << tagger.options.lambda_hidden_dim
     << '_' << tagger.options.actions_hidden_dim
     << '_' << tagger.options.parse_path_hidden_dim
     << '_' << tagger.options.span_hidden_dim
     << '_' << tagger.options.action_dim
     << '_' << tagger.options.state_dim
     << '_' << tagger.options.dropout;

  if (tagger.options.subtrees)
    os << "_subtrees";
  if (tagger.options.gated_parse)
    os << "_gated-parse";
  if (tagger.options.new_conn_action)
    os << "_new-conn";
  if (tagger.options.shift_action)
    os << "_shift";
  if (tagger.options.known_conns_only)
    os << "_known";
  if (tagger.options.train_pairwise)
    os << "_pairwise";
  os << "__pid" << getpid() << ".params";

  return os.str();
}


void OutputComparison(const CausalityMetrics& metrics,
                      const std::vector<bool>& pos_is_punct) {
  auto log_instances = [&](
      const vector<CausalityRelation>& gold_instances,
      const vector<CausalityRelation>& predicted_instances,
      const string& connective_status,
      const vector<array<bool, 3>>& argument_matches = {}) {
    if (gold_instances.size() != predicted_instances.size()
        && !gold_instances.empty() && !predicted_instances.empty()) {
      cerr << "Invalid comparison instances!" << endl;
      abort();
    }
    if (!argument_matches.empty() &&
        (argument_matches.size() != predicted_instances.size()
            || gold_instances.empty() || predicted_instances.empty())) {
      cerr << "Invalid argument matches!" << endl;
      abort();
    }

    // cout << "Sentence\tConnective\tConnective indices\tGold cause\tGold effect\t"
    //      << "Gold means\tLSTM status\tLSTM cause\tLSTM effect\tLSTM means\t"
    //      << "LSTM cause matches\tLSTM effect matches\tLSTM means matches\n";
    for (unsigned i = 0; i < max(gold_instances.size(),
                                 predicted_instances.size()); ++i) {
      // Always print sentence and connective.
      const CausalityRelation& non_null_instance =
          (gold_instances.empty() ? predicted_instances : gold_instances).at(i);
      cout << non_null_instance.GetSentence() << '\t';
      non_null_instance.PrintTokens(cout, non_null_instance.GetConnectiveIndices());
      cout << '\t';
      cout << non_null_instance.GetConnectiveIndices() << '\t';

      // Print gold arguments, if available.
      if (!gold_instances.empty()) {
        const CausalityRelation& gold_instance = gold_instances.at(i);
        for (unsigned j = 0; j < CausalityMetrics::NumArgs(); ++j) {
          gold_instance.PrintTokens(cout, gold_instance.GetArgument(j));
          cout << '\t';
        }
      } else {
        cout << "\t\t\t";
      }

      cout << connective_status << '\t';

      // Print predicted arguments, if available.
      if (!predicted_instances.empty()) {
        const CausalityRelation& predicted_instance = predicted_instances.at(i);
        for (unsigned j = 0; j < CausalityMetrics::NumArgs(); ++j) {
          predicted_instance.PrintTokens(cout, predicted_instance.GetArgument(j));
          cout << '\t';
        }
      } else {
        cout << "\t\t\t";
      }

      // If we're comparing TPs, print argument match statuses.
      if (!gold_instances.empty() && !predicted_instances.empty()) {
        for (unsigned j = 0; j < CausalityMetrics::NumArgs(); ++j) {
          unsigned args_match = // assume a match unless we have detailed info
              argument_matches.empty() ? 1 : argument_matches.at(i).at(j);
          cout << args_match;
          if (j + 1 < CausalityMetrics::NumArgs()) {
            cout << '\t';
          }
        }
      }
      cout << endl;
    }
  };

  log_instances(metrics.GetArgumentMatches(), metrics.GetArgumentMatches(),
                "TP");
  log_instances(metrics.GetFNs(), {}, "FN");
  log_instances({}, metrics.GetFPs(), "FP");

  vector<CausalityRelation> arg_mismatch_gold;
  vector<CausalityRelation> arg_mismatch_predicted;
  vector<array<bool, 3>> arg_matches;
  const lstm_parser::Sentence* previous_sentence = nullptr;
  const BecauseRelation::IndexList* previous_connective_indices = nullptr;

  for (const auto& match_tuple : metrics.GetArgumentMismatches()) {
    const CausalityRelation& gold_instance = std::get<0>(match_tuple);
    const CausalityRelation& predicted_instance = std::get<1>(match_tuple);
    unsigned mismatched_arg = std::get<2>(match_tuple);

    // Add a new entry to output if we've moved on to a different instance.
    // (Argument mismatches for the same instance will be adjacent in the list.)
    if (&gold_instance.GetSentence() != previous_sentence
        || gold_instance.GetConnectiveIndices()
            != *previous_connective_indices) {
      arg_mismatch_gold.push_back(gold_instance);
      arg_mismatch_predicted.push_back(predicted_instance);
      arg_matches.push_back({1, 1, 1});  // If we find no mismatch, it's a match
    }
    // The reason we're here is that evaluation found a mismatch.
    arg_matches.back().at(mismatched_arg) = 0;

    previous_sentence = &gold_instance.GetSentence();
    previous_connective_indices = &gold_instance.GetConnectiveIndices();
  }
  log_instances(arg_mismatch_gold, arg_mismatch_predicted, "TP", arg_matches);
}


void DoTrain(LSTMCausalityTagger* tagger,
             BecauseOracleTransitionCorpus* full_corpus, unsigned folds,
             double dev_pct, bool compare_punct, const string& model_fname,
             unsigned dev_eval_period, double epochs_cutoff,
             double recent_improvements_cutoff,
             double recent_improvements_epsilon, bool eval_pairwise,
             unsigned cv_start_at, unsigned cv_end_at,
             bool for_comparison) {
  unsigned num_sentences = full_corpus->sentences.size();
  vector<unsigned> all_sentence_indices(num_sentences);
  iota(all_sentence_indices.begin(), all_sentence_indices.end(), 0);
  random_shuffle(all_sentence_indices.begin(), all_sentence_indices.end());
  if (folds <= 1) {
    tagger->Train(full_corpus, all_sentence_indices, dev_pct, compare_punct,
                  model_fname, dev_eval_period, epochs_cutoff,
                  recent_improvements_cutoff, recent_improvements_epsilon,
                  &requested_stop);
  } else {
    cout << setprecision(4);

    // For cutoffs, we use one *past* the index where the fold should stop.
    vector<unsigned> fold_cutoffs(folds);
    unsigned uneven_sentences_to_distribute = num_sentences % folds;
    unsigned next_cutoff = num_sentences / folds;
    // TODO: merge this loop with the one below?
    for (unsigned i = 0; i < folds; ++i, next_cutoff += num_sentences / folds) {
      if (uneven_sentences_to_distribute > 0) {
        ++next_cutoff;
        --uneven_sentences_to_distribute;
      }
      fold_cutoffs[i] = next_cutoff;
    }
    assert(fold_cutoffs.back() == all_sentence_indices.size());
    unsigned previous_cutoff = 0;
    vector<CausalityMetrics> evaluation_results;
    vector<CausalityMetrics> pairwise_eval_results;
    evaluation_results.reserve(folds);
    if (eval_pairwise) {
      pairwise_eval_results.reserve(folds);
    }

    // Folds are 1-indexed, so subtract 1 from CL params to get indices.
    unsigned last_fold = cv_end_at;
    if (last_fold == UNSIGNED_NEG_1)
      last_fold = folds + 1;
    for (unsigned fold = cv_start_at - 1;
        fold < min(last_fold - 1, folds); ++fold) {
      cerr << "Starting fold " << fold + 1 << " of " << folds << endl;
      size_t current_cutoff = fold_cutoffs[fold];
      auto training_range = join(
          all_sentence_indices | ad::sliced(0u, previous_cutoff),
          all_sentence_indices
              | ad::sliced(current_cutoff, all_sentence_indices.size()));
      vector<unsigned> fold_train_order(training_range.begin(),
                                        training_range.end());
      assert(
          fold_train_order.size()
              == num_sentences - (current_cutoff - previous_cutoff));
      vector<unsigned> fold_test_order(
          all_sentence_indices.begin() + previous_cutoff,
          all_sentence_indices.begin() + current_cutoff);
      if (for_comparison) {
        cerr << "Testing order: " << fold_test_order << endl;
      }

      tagger->Train(full_corpus, fold_train_order, dev_pct, compare_punct,
                    model_fname, dev_eval_period, epochs_cutoff,
                    recent_improvements_cutoff, recent_improvements_epsilon,
                    &requested_stop);
      cerr << "Evaluating..." << endl;
      if (for_comparison) {
        cout << "Fold " << fold + 1 << " test sentences:" << endl;
        IndentingOStreambuf indent(cout);
        for (unsigned sentence_index : fold_test_order) {
          cout << full_corpus->sentences[sentence_index] << '\n';
        }
        cout << endl;
      }
      tagger->LoadModel(model_fname);  // Reset to last saved state
      cout << "Evaluation for fold " << fold + 1 << " ("
           << fold_test_order.size() << " test sentences)" << endl;
      CausalityMetrics evaluation = tagger->Evaluate(
          full_corpus, fold_test_order, compare_punct);

      if (for_comparison) {
        OutputComparison(
            evaluation,
            compare_punct ? vector<bool>() : full_corpus->pos_is_punct);
        cout << "\n\n";
      }

      IndentingOStreambuf indent(cout);
      cout << evaluation << '\n' << endl;
      evaluation_results.push_back(evaluation);
      if (eval_pairwise) {
        CausalityMetrics pairwise_evaluation = tagger->Evaluate(
            full_corpus, fold_test_order, compare_punct, eval_pairwise);
        cout << "Pairwise evaluation:";
        IndentingOStreambuf indent(cout);
        if (for_comparison) {
          cout << '\n';
          OutputComparison(
              pairwise_evaluation,
              compare_punct ? vector<bool>() : full_corpus->pos_is_punct);
          cout << '\n';
        }

        cout << '\n' << pairwise_evaluation << '\n' << endl;
        pairwise_eval_results.push_back(pairwise_evaluation);
      }

      requested_stop = false;
      previous_cutoff = current_cutoff;
      tagger->Reset();  // Reset for next fold
    }

    {
      cout << "Average evaluation:\n";
      IndentingOStreambuf indent(cout);
      auto evals_range = boost::make_iterator_range(evaluation_results);
      cout << AveragedCausalityMetrics(evals_range) << endl;
    }
    if (eval_pairwise) {
      cout << "Average pairwise evaluation:" << '\n';
      IndentingOStreambuf indent(cout);
      auto evals_range = boost::make_iterator_range(pairwise_eval_results);
      cout << AveragedCausalityMetrics(evals_range) << endl;
    }
  }
}


void RunCommandToStream(const char* cmd, ostream& s) {
  array<char, 128> buffer;
  unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe)
    throw runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
      s << buffer.data();
  }
}


int main(int argc, char** argv) {
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i)
    cerr << ' ' << argv[i];
  cerr << endl;

  // Print Git status.
  {
    cerr << "Git status:";
    IndentingOStreambuf indent(cerr);
    cerr << '\n';
    RunCommandToStream("git rev-parse HEAD", cerr);
    cerr << "Modified:";
    IndentingOStreambuf indent2(cerr);
    cerr << '\n';
    RunCommandToStream("git ls-files -m", cerr);
  }

  unsigned random_seed = cnn::Initialize(argc, argv);
  srand(random_seed + 1);

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);

  bool for_comparison = conf["for-comparison"].as<bool>();
  bool log_diffs = conf["log-diffs"].as<bool>();

  LSTMCausalityTagger tagger(
      conf["parser-model"].as<string>(),
      LSTMCausalityTagger::TaggerOptions{
          conf["word-dim"].as<unsigned>(),
          conf["lstm-layers"].as<unsigned>(),
          conf["token-dim"].as<unsigned>(),
          conf["lambda-hidden-dim"].as<unsigned>(),
          conf["actions-hidden-dim"].as<unsigned>(),
          conf["parse-path-arc-dim"].as<unsigned>(),
          conf["parse-path-hidden-dim"].as<unsigned>(),
          conf["span-hidden-dim"].as<unsigned>(),
          conf["action-dim"].as<unsigned>(),
          conf["state-dim"].as<unsigned>(),
          conf["dropout"].as<float>(),
          conf["conn-in-state"].as<bool>(),
          conf["subtrees"].as<bool>(),
          conf["gated-parse"].as<bool>(),
          conf["new-conn-action"].as<bool>(),
          conf["shift-action"].as<bool>(),
          conf["known-conns-only"].as<bool>(),
          conf["train-pairwise"].as<bool>(),
          log_diffs || for_comparison,
          log_diffs,
          conf["oracle-conns"].as<bool>()});

  if (conf.count("train")) {
    double dev_pct = conf["dev-pct"].as<double>();
    if (dev_pct < 0.0 || dev_pct > 1.0) {
      cerr << "Invalid dev percentage: " << dev_pct << endl;
      abort();
    }
    if (!conf.count("training-data")) {
      cerr << "Can't train without training corpus!"
              " Please provide --training-data." << endl;
      abort();
    }
    if (tagger.options.dropout >= 1.0) {
      cerr << "Invalid dropout rate: " << tagger.options.dropout << endl;
      abort();
    }
    if (tagger.options.oracle_connectives && !tagger.options.new_conn_action) {
      cerr << "Oracle connectives are available only with NEW-CONN actions"
           << endl;
      abort();
    }

    double epochs_cutoff = conf["epochs-cutoff"].as<double>();
    double recent_improvements_cutoff =
        conf["recent-improvements-cutoff"].as<double>();
    double recent_improvements_epsilon =
        conf["recent-improvements-epsilon"].as<double>();
    bool compare_punct = conf.count("compare-punct");
    bool eval_pairwise = conf.count("eval-pairwise");
    unsigned dev_eval_period = conf["dev-eval-period"].as<unsigned>();
    unsigned folds = conf["folds"].as<unsigned>();

    const string& training_path = conf["training-data"].as<string>();
    BecauseOracleTransitionCorpus full_corpus(
        tagger.GetVocab(), training_path, true, for_comparison);
    tagger.FinalizeVocab();
    cerr << "Corpus size: " << full_corpus.sentences.size() << " sentences"
         << endl;

    const string& model_dir = conf["model-dir"].as<string>();
    const string model_fname = GetModelFileName(tagger, model_dir);
    cerr << "Writing parameters to file: " << model_fname << endl;

    signal(SIGINT, signal_callback_handler);
    DoTrain(&tagger, &full_corpus, folds, dev_pct, compare_punct, model_fname,
            dev_eval_period, epochs_cutoff, recent_improvements_cutoff,
            recent_improvements_epsilon, eval_pairwise,
            conf["cv-start-at"].as<unsigned>(),
            conf["cv-end-at"].as<unsigned>(), for_comparison);
  }


  return 0;
}
