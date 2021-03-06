#ifndef LSTM_CAUSALITY_METRICS_H_
#define LSTM_CAUSALITY_METRICS_H_

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/equal.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/algorithm/set_algorithm.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#ifndef NDEBUG // In debug mode, include FRIEND_TEST code.
  #include <gtest/gtest_prod.h>
#endif
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "BecauseData.h"
#include "BecauseOracleTransitionCorpus.h"
#include "diff-cpp/lcs.h"
#include "diff-cpp/RandomAccessSequence.h"
#include "utilities.h"


constexpr unsigned UNSIGNED_NEG_1 = static_cast<unsigned>(-1);


template <class RangeType>
std::pair<double, double> GetMeanAndStdDev(const RangeType& range) {
  using namespace boost::accumulators;
  accumulator_set<double, stats<tag::lazy_variance>> acc_variance;
  for (const auto &elt : range) {
    acc_variance(elt);
  }
  return {mean(acc_variance), std::sqrt(variance(acc_variance))};
}


class ClassificationMetrics {
public:
  // Often there is no well-defined concept of a true negative, so it
  // defaults to undefined.
  ClassificationMetrics(unsigned tp = 0, unsigned fp = 0, unsigned fn = 0,
                        unsigned tn = -1) :
      tp(tp), fp(fp), fn(fn), tn(tn) {}

  void operator+=(const ClassificationMetrics& other) {
    tp += other.tp;
    fp += other.fp;
    fn += other.fn;
    tn = (tn == UNSIGNED_NEG_1 || other.tn == UNSIGNED_NEG_1 ?
            UNSIGNED_NEG_1 : tn + other.tn);
  }

  ClassificationMetrics operator+(const ClassificationMetrics& other) const {
    ClassificationMetrics sum = *this;
    sum += other;
    return sum;
  }

  bool operator==(const ClassificationMetrics& other) const {
    return tp == other.tp && fp == other.fp && fn == other.fn && tn == other.tn
        // In case this is an averaged metrics, check the derived values, too.
        && FloatsAlmostSame(GetAccuracy(), other.GetAccuracy())
        && FloatsAlmostSame(GetPrecision(), other.GetPrecision())
        && FloatsAlmostSame(GetRecall(), other.GetRecall())
        && FloatsAlmostSame(GetF1(), other.GetF1());
  }

  virtual double GetAccuracy() const {
    return CalculateAccuracy(tp, fp, fn, tn);
  }
  virtual double GetPrecision() const { return CalculatePrecision(tp, fp); }
  virtual double GetRecall() const { return CalculateRecall(tp, fn); }
  virtual double GetF1() const {
    return CalculateF1(ClassificationMetrics::GetPrecision(),
                       ClassificationMetrics::GetRecall());
  }

  virtual ~ClassificationMetrics() {}

  static double CalculateAccuracy(unsigned tp, unsigned fp, unsigned fn,
                                  unsigned tn = -1) {
    if (tn != static_cast<unsigned>(-1)) {
      return static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    } else {
      return std::nan("");
    }
  }

  static double CalculatePrecision(unsigned tp, unsigned fp) {
    if (tp + fp == 0)
      return 0;
    return static_cast<double>(tp) / (tp + fp);
  }

  static double CalculateRecall(unsigned tp, unsigned fn) {
    if (tp + fn == 0)
      return 0;
    return static_cast<double>(tp) / (tp + fn);
  }

  static double CalculateF1(double precision, double recall) {
    if (precision + recall == 0)
      return 0;
    return (2 * precision * recall) / (precision + recall);
  }

  // Store as doubles to allow averaging.
  double tp;
  double fp;
  double fn;
  double tn;

  virtual void Print(std::ostream& s) const {
#ifdef METRICS_PRINT_RAW_COUNTS
    s << "TP: " << tp << "\nFP:" << fp
    << "\nFN:" << fn << "\nTN: " << tn << "\n";
#endif
    s << "Accuracy: " << GetAccuracy()
        << "\nPrecision: " << GetPrecision()
        << "\nRecall: " << GetRecall()
        << "\nF1: " << GetF1();
  }
};

class AveragedClassificationMetrics : public ClassificationMetrics {
#ifndef NDEBUG
  FRIEND_TEST(DataMetricsTest, AveragingMetricsWorks);
#endif
public:
  template <class IteratorType>
  AveragedClassificationMetrics(
      const boost::iterator_range<IteratorType> all_metrics)
      : // Start by initializing our counts to the sums of the individual counts
        ClassificationMetrics(
          boost::accumulate(all_metrics, ClassificationMetrics())),
        avg_f1(0), avg_recall(0), avg_precision(0), avg_accuracy(0) {
    // Now divide to turn those into averages.
    tp = tp / all_metrics.size();
    fp = fp / all_metrics.size();
    fn = fn / all_metrics.size();
    if (tn != UNSIGNED_NEG_1)
      tn = tn / all_metrics.size();

    // Compute averages and derived metrics.
    std::tie(avg_f1, f1_stddev) = GetMeanAndStdDev(
        BOOST_TRANSFORMED_RANGE(all_metrics, m, m.GetF1()));
    std::tie(avg_recall, recall_stddev) = GetMeanAndStdDev(
        BOOST_TRANSFORMED_RANGE(all_metrics, m, m.GetRecall()));
    std::tie(avg_precision, precision_stddev) = GetMeanAndStdDev(
        BOOST_TRANSFORMED_RANGE(all_metrics, m, m.GetPrecision()));
    std::tie(avg_accuracy, accuracy_stddev) = GetMeanAndStdDev(
        BOOST_TRANSFORMED_RANGE(all_metrics, m, m.GetAccuracy()));
  }

  virtual double GetAccuracy() const { return avg_accuracy; }
  virtual double GetPrecision() const { return avg_precision; }
  virtual double GetRecall() const { return avg_recall; }
  virtual double GetF1() const { return avg_f1; }

  virtual void Print(std::ostream& s) const {
#ifdef METRICS_PRINT_RAW_COUNTS
    s << "TP: " << tp << "\nFP:" << fp
    << "\nFN:" << fn << "\nTN: " << tn << "\n";
#endif
    s << "Accuracy: " << GetAccuracy() << "±" << accuracy_stddev
        << "\nPrecision: " << GetPrecision() << "±" << precision_stddev
        << "\nRecall: " << GetRecall() << "±" << recall_stddev
        << "\nF1: " << GetF1() << "±" << f1_stddev;
  }

protected:
  double avg_f1;
  double avg_recall;
  double avg_precision;
  double avg_accuracy;

  double f1_stddev;
  double recall_stddev;
  double precision_stddev;
  double accuracy_stddev;
};

inline std::ostream& operator<<(std::ostream& s,
                         const ClassificationMetrics& metrics) {
  metrics.Print(s);
  return s;
}


struct ArgumentMetrics {
  ArgumentMetrics()
      : spans(new ClassificationMetrics(0, 0, 0)), jaccard_index(0),
        instance_count(0) {}

  ArgumentMetrics(unsigned tps, unsigned fps, unsigned fns,
                  double jaccard_index, unsigned instance_count = 0)
      : spans(new ClassificationMetrics(tps, fps, fns)),
        jaccard_index(jaccard_index), instance_count(instance_count) {}

  virtual ~ArgumentMetrics() {}

  ArgumentMetrics(const ArgumentMetrics& other)
      : spans(new ClassificationMetrics(*other.spans)),
        jaccard_index(other.jaccard_index),
        instance_count(other.instance_count) {}

  void operator+=(const ArgumentMetrics& other) {
    *spans += *other.spans;
    // Jaccard indices were already probably averages. So we have to preserve
    // the weighting for the new average.
    // If either Jaccard index is nan, ignore it and use the other one (and its
    // instance count). A nan just means there were no span scores to average.
    if (std::isnan(jaccard_index)) {
      jaccard_index = other.jaccard_index;
      instance_count = other.instance_count;
    } else if (!std::isnan(other.jaccard_index)) {
      unsigned combined_instance_count = instance_count + other.instance_count;
      jaccard_index = (jaccard_index * instance_count
                       + other.jaccard_index * other.instance_count)
          / combined_instance_count;
      instance_count = combined_instance_count;
    }
  }

  ArgumentMetrics operator+(const ArgumentMetrics& other) const {
    ArgumentMetrics sum = *this;
    sum += other;
    return sum;
  }

  bool operator==(const ArgumentMetrics& other) const {
    // instance_count is irrelevant; we only care about it for tracking future
    // changes.
    return *spans == *other.spans
        && FloatsAlmostSame(jaccard_index, other.jaccard_index);
  }

  virtual void Print(std::ostream& s) const {
    s << "Spans:";
    {
      IndentingOStreambuf indent(s);
      s << '\n' << *spans;
    }
    s << "\nJaccard index (conditional): " << jaccard_index;
  }

  std::unique_ptr<ClassificationMetrics> spans;
  double jaccard_index;
  unsigned instance_count;
};


class AveragedArgumentMetrics : public ArgumentMetrics {
#ifndef NDEBUG
  FRIEND_TEST(DataMetricsTest, AveragingMetricsWorks);
#endif
public:
  template <class IteratorType>
  AveragedArgumentMetrics(boost::iterator_range<IteratorType> all_metrics) {
    auto all_spans = BOOST_TRANSFORMED_RANGE(all_metrics, m, *m.spans);
    auto all_jaccards = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                m.jaccard_index);
    auto all_instance_counts = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                       m.instance_count);
    spans.reset(new AveragedClassificationMetrics(all_spans));
    // We're assuming an unweighted average here, so don't worry too much about
    // what happens to the instance counts. Those only matter for adding.
    instance_count = std::round(boost::accumulate(all_instance_counts, 0) /
                                static_cast<double>(all_metrics.size()));
    std::tie(jaccard_index, jaccard_stddev) = GetMeanAndStdDev(all_jaccards);
  }

  virtual void Print(std::ostream& s) const {
    ArgumentMetrics::Print(s);
    // Last thing output is Jaccard index, which won't have stddev attached yet.
    s << "±" << jaccard_stddev;
  }

protected:
  double jaccard_stddev;
};

inline std::ostream& operator<<(std::ostream& s,
                                const ArgumentMetrics& metrics) {
  metrics.Print(s);
  return s;
}

// Specialization so GTest can find it.
inline std::ostream& operator<<(std::ostream& s,
                                const AveragedArgumentMetrics& metrics) {
  return operator<<(s, static_cast<const ArgumentMetrics&>(metrics));
}


// Represents a filter that returns true for span tokens that should be filtered
// out of comparisons.
struct SpanTokenFilter {
  bool operator()(unsigned token_id) const {
    if (compare_punct) {
      return false;
    } else {
      return pos_is_punct[sentence.poses.at(token_id)];
    }
  }

  bool compare_punct;
  const lstm_parser::Sentence& sentence;
  const std::vector<bool>& pos_is_punct;
};


template <typename RelationType>  // forward declaration
class BecauseRelationMetrics;

template <typename RelationType>
inline std::ostream& operator<<(
    std::ostream& s, const BecauseRelationMetrics<RelationType>& metrics) {
  s << "Connectives:";
  {
    IndentingOStreambuf indent(s);
    s << '\n' << *metrics.connective_metrics;
  }
  s << "\nArguments:";
  {
    IndentingOStreambuf indent(s);
    for (unsigned argi :
         boost::irange(0u, BecauseRelationMetrics<RelationType>::NumArgs())) {
      s << '\n' << RelationType::ARG_NAMES[argi] << ':';
      IndentingOStreambuf indent2(s);
      s << '\n' << *metrics.argument_metrics[argi];
    }
  }

  if (metrics.log_differences) {
    auto log_instances = [&s](
        const std::vector<CausalityRelation>& instances,
        const std::string& label) {
      if (!instances.empty()) {
        s << label << ":\n";
        IndentingOStreambuf indent(s);
        for (const CausalityRelation& instance : instances) {
          s << instance << std::endl;
        }
      }
    };

    s << "\n\n";
    log_instances(metrics.argument_matches, "Correct instances");
    s << '\n';
    log_instances(metrics.fps, "False positives");
    s << '\n';
    log_instances(metrics.fns, "False negatives");

    if (!metrics.argument_mismatches.empty()) {
      s << "\nArgument mismatches:\n";
      IndentingOStreambuf indent(s);
      for (const auto& instance_tuple : metrics.argument_mismatches) {
        const unsigned arg_type = std::get<0>(instance_tuple);
        s << RelationType::ARG_NAMES[arg_type] << " mismatch (J="
          << std::get<3>(instance_tuple) << "):\n"
          << "  " << std::get<1>(instance_tuple)
          << "\n  vs.\n  " << std::get<2>(instance_tuple) << std::endl;
      }
    }
  }

  return s;
}


template <class RelationType>
class BecauseRelationMetrics {
public:
  // Comparison machinery that allows partial overlap.
  template <class IndexContainer>
  struct SortedSpansMatch {
    SortedSpansMatch(double ovl_threshold) : ovl_threshold(ovl_threshold) {}

    bool operator()(const IndexContainer& i1, const IndexContainer& i2) const {
      if (ovl_threshold >= 1.0) {
        return boost::equal(i1, i2);
      } else {
        if (i1.empty() && i2.empty()) {
          return true;
        }

        // assert(std::is_sorted(begin(i1), end(i1))
        //       && std::is_sorted(begin(i2), end(i2)));
        Counter intersection_counter;
        boost::set_intersection(i1, i2,
                                std::back_inserter(intersection_counter));
        double longer_length = static_cast<double>(
            i2.size() > i1.size() ? i2.size() : i1.size());
        return intersection_counter.count / longer_length >= ovl_threshold;
      }
    }

    double ovl_threshold;
  };

  struct ConnectivesMatch {
    ConnectivesMatch(double ovl_threshold) : ovl_threshold(ovl_threshold) {}

    bool operator()(const RelationType& i1, const RelationType& i2) const {
      return SortedSpansMatch<typename RelationType::IndexList>(ovl_threshold)(
          i1.GetConnectiveIndices(), i2.GetConnectiveIndices());
    }

    double ovl_threshold;
  };

  typedef Diff<RandomAccessSequence<
      typename std::vector<RelationType>::const_iterator>, ConnectivesMatch>
      ConnectiveDiff;
  typedef Diff<RandomAccessSequence<
      typename BecauseRelation::IndexList::const_iterator>> IndexDiff;

  // Default constructor: initialize metrics with zero instances, and prepare
  // argument metrics to contain the correct number of entries for the relation
  // type.
  BecauseRelationMetrics()
      : connective_metrics(new ClassificationMetrics),
        argument_metrics(RelationType::ARG_NAMES.size()),
        log_differences(false) {
    for (unsigned i = 0; i < argument_metrics.size(); ++i) {
      argument_metrics[i].reset(new ArgumentMetrics);
    }
  }

  BecauseRelationMetrics(const BecauseRelationMetrics& other)
      : connective_metrics(
          new ClassificationMetrics(*other.connective_metrics)),
        argument_metrics(other.argument_metrics.size()),
        log_differences(other.log_differences) {
    for (unsigned i = 0; i < argument_metrics.size(); ++i) {
      argument_metrics[i].reset(
          new ArgumentMetrics(*other.argument_metrics[i]));
    }
  }

  BecauseRelationMetrics(
      const std::vector<RelationType>& sentence_gold,
      const std::vector<RelationType>& sentence_predicted,
      const BecauseOracleTransitionCorpus& source_corpus,
      const GraphEnhancedParseTree& parse, const SpanTokenFilter& filter,
      const unsigned missing_instances = 0,
      const std::vector<
          BecauseOracleTransitionCorpus::ExtrasententialArgCounts>&
          missing_args = {},
      bool save_differences = false, bool log_differences = false,
      double overlap_threshold = 1.0)
        : argument_metrics(NumArgs()), log_differences(log_differences) {
    ConnectiveDiff diff(sentence_gold, sentence_predicted,
                        ConnectivesMatch(overlap_threshold));
    unsigned tp = diff.LCS().size();
    unsigned connective_fps = sentence_predicted.size() - tp;
    unsigned connective_fns = sentence_gold.size() - tp + missing_instances;

    const typename ConnectiveDiff::IndexList matching_gold =
        diff.OrigLCSIndices();
    const typename ConnectiveDiff::IndexList matching_pred =
        diff.NewLCSIndices();
    assert(matching_pred.size() == matching_gold.size());
    unsigned num_matching_connectives = matching_gold.size();
    connective_metrics.reset(
        new ClassificationMetrics(tp, connective_fps, connective_fns));

    // Entries for non-TP will all be true.
    std::vector<bool> gold_args_match_if_tp(sentence_gold.size(), true);
    for (unsigned arg_num : boost::irange(0u, NumArgs())) {
      unsigned span_tps = 0;
      unsigned span_fps = 0;
      unsigned span_fns = 0;
      unsigned span_mismatches = 0;
      unsigned gold_index;
      unsigned pred_index;
      double jaccard_sum = 0.0;
      for (auto both_indices : boost::combine(matching_gold, matching_pred)) {
        boost::tie(gold_index, pred_index) = both_indices;
        const CausalityRelation& gold_instance = sentence_gold.at(gold_index);
        const auto& gold_span = gold_instance.GetArgument(arg_num);
        const CausalityRelation& pred_instance =
            sentence_predicted.at(pred_index);
        const auto& pred_span = pred_instance.GetArgument(arg_num);
        // We're going to need to copy over the spans anyway for Jaccard index
        // calculations (diff requires random access). So we'll just copy and
        // filter the copy.
        BecauseRelation::IndexList filtered_gold(gold_span);
        ReallyDeleteIf(&filtered_gold, filter);
        BecauseRelation::IndexList filtered_pred(pred_span);
        ReallyDeleteIf(&filtered_pred, filter);

        unsigned gold_arg_missing_tokens =
            missing_args.empty() ? 0 : missing_args[gold_index][arg_num];
        // If we have any missing tokens for the argument, we definitely got the
        // span wrong.
        bool args_match = true;
        if (gold_arg_missing_tokens == 0) {
          SortedSpansMatch<decltype(filtered_gold)> arg_cmp(overlap_threshold);
          if (arg_cmp(filtered_gold, filtered_pred)) {
            if (!filtered_gold.empty())
              ++span_tps;
          } else {
            args_match = false;
            if (filtered_gold.empty()) {
              ++span_fps;
            } else if (filtered_pred.empty()) {
              ++span_fns;
            } else {
              ++span_mismatches;
            }
          }
        } else {  // We're missing some argument tokens (cross-sentence span)
          args_match = false;
          if (filtered_pred.empty()) {
            ++span_fns;
          } else {
            std::cerr << "WARNING: assuming mismatch for argument "
                      << RelationType::ARG_NAMES[arg_num]
                      << " of instance with missing argument tokens: "
                      << sentence_gold[gold_index] << std::endl;
          }
        }

        double jaccard = CalculateJaccard(filtered_gold, filtered_pred,
                                          gold_arg_missing_tokens);
        jaccard_sum += jaccard;
        if (save_differences && !args_match) {
          argument_mismatches.push_back(
              std::make_tuple(arg_num, gold_instance, pred_instance, jaccard));
        }
        gold_args_match_if_tp[gold_index] =
            gold_args_match_if_tp[gold_index] && args_match;
      }

      auto current_arg_metrics = new ArgumentMetrics(
          span_tps, connective_fps + span_fps + span_mismatches,
          connective_fns + span_fns + span_mismatches,
          jaccard_sum / num_matching_connectives, num_matching_connectives);
      argument_metrics[arg_num].reset(current_arg_metrics);
    }

    if (save_differences) {
      RecordInstanceDifferences(diff, sentence_gold, sentence_predicted);
      for (const unsigned gold_index : matching_gold) {
        if (gold_args_match_if_tp[gold_index]) {
          argument_matches.push_back(sentence_gold[gold_index]);
        }
      }
    }
  }

  void operator+=(const BecauseRelationMetrics<RelationType>& other) {
    *connective_metrics += *other.connective_metrics;
    for (size_t i = 0; i < argument_metrics.size(); ++i) {
      *argument_metrics[i] += *other.argument_metrics[i];
    }

    AppendToVector(other.argument_matches.begin(), other.argument_matches.end(),
                   &argument_matches);
    AppendToVector(other.argument_mismatches.begin(),
                   other.argument_mismatches.end(), &argument_mismatches);
    AppendToVector(other.fps.begin(), other.fps.end(), &fps);
    AppendToVector(other.fns.begin(), other.fns.end(), &fns);
  }

  void operator+=(BecauseRelationMetrics<RelationType>&& other) {
    using std::make_move_iterator;
    *connective_metrics += *other.connective_metrics;
    for (size_t i = 0; i < argument_metrics.size(); ++i) {
      *argument_metrics[i] += *other.argument_metrics[i];
    }

    // TODO: Why the @#$% doesn't vector<T>::insert() work here?? (and above)
    AppendToVector(make_move_iterator(other.argument_matches.begin()),
                   make_move_iterator(other.argument_matches.end()),
                   &argument_matches);
    AppendToVector(
        make_move_iterator(other.argument_mismatches.begin()),
        make_move_iterator(other.argument_mismatches.end()),
        &argument_mismatches);
    AppendToVector(make_move_iterator(other.fps.begin()),
                   make_move_iterator(other.fps.end()),
                   &fps);
    AppendToVector(make_move_iterator(other.fns.begin()),
                   make_move_iterator(other.fns.end()),
                   &fns);
  }

  BecauseRelationMetrics<RelationType> operator+(
      const BecauseRelationMetrics<RelationType>& other) const {
    BecauseRelationMetrics sum = *this;
    sum += other;
    return sum;
  }

  bool operator==(const BecauseRelationMetrics<RelationType>& other) {
    using namespace std::rel_ops;
    if (*connective_metrics != *other.connective_metrics)
      return false;
    for (unsigned i = 0; i < argument_metrics.size(); ++i) {
      if (*argument_metrics[i] != *other.argument_metrics[i])
        return false;
    }
    return true;
  }

  unsigned GetHead(const typename RelationType::IndexList& span,
                   const GraphEnhancedParseTree& parse,
                   const BecauseOracleTransitionCorpus& source_corpus) {
    int highest_token = -1;  // Using signed int makes this work for testing
    int highest_token_depth = std::numeric_limits<int>::max();
    for (unsigned token_id : span) {
      int depth = parse.GetTokenDepth(token_id);
      if (depth < highest_token_depth
          || (depth == highest_token_depth
              && TokenPreferredForHead(token_id, highest_token, parse,
                                       source_corpus))) {
        /*
        if (depth >= highest_token_depth) {
          std::cerr << "Preferring "
                    << parse.GetSentence().WordForToken(token_id)
                    << " to " << parse.GetSentence().WordForToken(highest_token)
                    << " in sentence " << parse.GetSentence() << std::endl;
        }
        //*/
        highest_token = token_id;
        highest_token_depth = depth;
      }
    }
    return highest_token;
  }

  static double CalculateJaccard(
      const typename RelationType::IndexList& gold,
      const typename RelationType::IndexList& predicted,
      const unsigned unknown_predicted) {
    if (gold.empty() && predicted.empty() && unknown_predicted == 0) {
      return 1.0;
    }
    IndexDiff diff(gold, predicted);
    unsigned num_matching = diff.LCS().size();
    return num_matching
        / static_cast<double>(gold.size() + predicted.size() + unknown_predicted
            - num_matching);
  }

  static bool TokenPreferredForHead(
      unsigned new_token, unsigned old_token,
      const GraphEnhancedParseTree& parse,
      const BecauseOracleTransitionCorpus& source_corpus) {
    // If the depths are equal, prefer verbs/copulas over nouns, and nouns over
    // others. This helps to get the correct heads for fragmented spans, such as
    // spans that consist of an xcomp and its subject, as well as a few other
    // edge cases.
    if (IsClauseHead(old_token, parse, source_corpus))
      return false;
    if (IsClauseHead(new_token, parse, source_corpus))
      return true;
    if (source_corpus.pos_is_noun[parse.GetSentence().poses.at(old_token)])
      return false;
    if (source_corpus.pos_is_noun[parse.GetSentence().poses.at(new_token)])
      return true;
    return false;
  }

  static bool IsClauseHead(unsigned token_id,
                           const GraphEnhancedParseTree& parse,
                           const BecauseOracleTransitionCorpus& source_corpus) {
    // Non-modal verbs are clause heads.
    if (source_corpus.pos_is_non_modal_verb[
          parse.GetSentence().poses.at(token_id)]) {
      return true;
    }
    // Copula heads are clause heads.
    for (unsigned child_id : parse.GetChildren(token_id)) {
      if (parse.GetArcLabel(child_id) == "cop")
        return true;
    }
    // Check for an incoming arc label that indicates a clause.
    if (boost::algorithm::any_of_equal(
          BecauseOracleTransitionCorpus::INCOMING_CLAUSE_EDGES,
          parse.GetArcLabel(token_id)))
      return true;
    return false;
  }

  const auto& GetArgumentMatches() const { return argument_matches; }

  const auto& GetArgumentMismatches() const { return argument_mismatches; }

  const auto& GetFPs() const { return fps; }

  const auto& GetFNs() const { return fns; }

  static unsigned NumArgs() { return RelationType::ARG_NAMES.size(); }

  std::unique_ptr<ClassificationMetrics> connective_metrics;
  std::vector<std::unique_ptr<ArgumentMetrics>> argument_metrics;

protected:
  friend std::ostream& operator<< <>(
      std::ostream& s, const BecauseRelationMetrics<RelationType>& metrics);

  template <typename Iterator, typename VectorType>
  void AppendToVector(Iterator start, Iterator end, VectorType* vector) {
    vector->reserve(vector->size() + (end - start));
    for (; start != end; ++start) {
      vector->push_back(*start);
    }
  }

  void RecordInstanceDifferences(
      const ConnectiveDiff& diff,
      const std::vector<RelationType>& sentence_gold,
      const std::vector<RelationType>& sentence_predicted) {
    for (unsigned gold_only_index : diff.OrigOnlyIndices()) {
      fns.push_back(sentence_gold[gold_only_index]);
    }
    for (unsigned pred_only_index : diff.NewOnlyIndices()) {
      fps.push_back(sentence_predicted[pred_only_index]);
    }
  }

  std::vector<CausalityRelation> argument_matches;
  std::vector<std::tuple<unsigned, CausalityRelation, CausalityRelation,
              double>> argument_mismatches;
  std::vector<CausalityRelation> fps;
  std::vector<CausalityRelation> fns;
  bool log_differences;
};

template <class RelationType>
class AveragedBecauseRelationMetrics
    : public BecauseRelationMetrics<RelationType> {
#ifndef NDEBUG
  FRIEND_TEST(DataMetricsTest, AveragingMetricsWorks);
#endif
public:
  template <class IteratorType>
  AveragedBecauseRelationMetrics(
      boost::iterator_range<IteratorType> all_metrics) {
    auto connectives_range = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                     *m.connective_metrics);
    this->connective_metrics.reset(
        new AveragedClassificationMetrics(connectives_range));
    for (unsigned i :
         boost::irange(0u, BecauseRelationMetrics<RelationType>::NumArgs())) {
      auto all_args_i = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                *m.argument_metrics[i]);
      this->argument_metrics[i].reset(new AveragedArgumentMetrics(all_args_i));
    }
  }
};


typedef BecauseRelationMetrics<CausalityRelation> CausalityMetrics;
typedef AveragedBecauseRelationMetrics<CausalityRelation>
    AveragedCausalityMetrics;


#endif /* LSTM_CAUSALITY_METRICS_H_ */
