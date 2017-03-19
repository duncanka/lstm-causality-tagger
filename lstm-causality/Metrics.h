#ifndef LSTM_CAUSALITY_METRICS_H_
#define LSTM_CAUSALITY_METRICS_H_

#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/equal.hpp>
#include <boost/algorithm/cxx11/any_of.hpp>
#include <boost/range/combine.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/numeric.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include "BecauseData.h"
#include "BecauseOracleTransitionCorpus.h"
#include "diff-cpp/lcs.h"
#include "diff-cpp/RandomAccessSequence.h"
#include "utilities.h"


constexpr unsigned UNSIGNED_NEG_1 = static_cast<unsigned>(-1);

class ClassificationMetrics {
public:
  unsigned tp;
  unsigned fp;
  unsigned fn;
  unsigned tn;

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
    return tp == other.tp && fp == other.fp && fn == other.fn && tn == other.tn;
  }

  virtual double GetAccuracy() const {
    if (tn != static_cast<unsigned>(-1)) {
      return static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    } else {
      return std::nan("");
    }
  }

  virtual double GetPrecision() const {
    if (tp + fp == 0)
      return 0;
    return static_cast<double>(tp) / (tp + fp);
  }

  virtual double GetRecall() const {
    if (tp + fn == 0)
      return 0;
    return static_cast<double>(tp) / (tp + fn);
  }

  virtual double GetF1() const {
    return CalculateF1(GetPrecision(), GetRecall());
  }

  virtual ~ClassificationMetrics() {}

  static double CalculateF1(double precision, double recall) {
    if (precision + recall == 0)
      return 0;
    return (2 * precision * recall) / (precision + recall);
  }
};

class AveragedClassificationMetrics : public ClassificationMetrics {
public:
  template <class IteratorType>
  AveragedClassificationMetrics(
      const boost::iterator_range<IteratorType> all_metrics)
      : // Start by initializing our counts to the sums of the individual counts
        ClassificationMetrics(
          boost::accumulate(all_metrics, ClassificationMetrics())),
        avg_f1(0), avg_recall(0), avg_precision(0), avg_accuracy(0) {
    // Now divide to turn those into averages.
    double count = static_cast<double>(all_metrics.size());
    tp = std::round(tp / count);
    fp = std::round(fp / count);
    fn = std::round(tn / count);
    if (tn != UNSIGNED_NEG_1)
      tn = std::round(tn / count);

    // Finally, compute averaged derived metrics.
    for (const ClassificationMetrics& metrics : all_metrics) {
      avg_f1 += metrics.GetF1();
      avg_recall += metrics.GetRecall();
      avg_precision += metrics.GetPrecision();
      avg_accuracy += metrics.GetAccuracy();
    }
    avg_f1 /= count;
    avg_recall /= count;
    avg_precision /= count;
    avg_accuracy /= count;
  }

  virtual double GetAccuracy() const { return avg_accuracy; }
  virtual double GetPrecision() const { return avg_precision; }
  virtual double GetRecall() const { return avg_recall; }
  virtual double GetF1() const { return avg_f1; }

protected:
  double avg_f1;
  double avg_recall;
  double avg_precision;
  double avg_accuracy;
};

inline std::ostream& operator<<(std::ostream& s,
                         const ClassificationMetrics& metrics) {
  s << "Accuracy: " << metrics.GetAccuracy()
    << "\nPrecision: " << metrics.GetPrecision()
    << "\nRecall: " << metrics.GetRecall()
    << "\nF1: " << metrics.GetF1();
  return s;
}


class AccuracyMetrics {
public:
  unsigned correct;
  unsigned incorrect;

  AccuracyMetrics(unsigned correct, unsigned incorrect) :
      correct(correct), incorrect(incorrect) {}

  virtual ~AccuracyMetrics() {}

  virtual double GetAccuracy() const {
    return static_cast<double>(correct) / incorrect;
  }

  bool operator==(const AccuracyMetrics& other) const {
    return correct == other.correct && incorrect == other.incorrect;
  }

  void operator+=(const AccuracyMetrics& other) {
    correct += other.correct;
    incorrect += other.incorrect;
  }

  AccuracyMetrics operator+(const AccuracyMetrics& other) const {
    return AccuracyMetrics(correct + other.correct,
                           incorrect + other.incorrect);
  }
};

class AveragedAccuracyMetrics : public AccuracyMetrics {
public:
  template <class Iterator>
  AveragedAccuracyMetrics(
      boost::iterator_range<Iterator> all_metrics)
      : AccuracyMetrics(boost::accumulate(all_metrics, AccuracyMetrics(0, 0))) {
    double count = all_metrics.size();
    correct = std::round(correct / count);
    incorrect = std::round(incorrect / count);
    for (const AccuracyMetrics& metrics : all_metrics) {
      avg_accuracy += metrics.GetAccuracy();
    }
    avg_accuracy /= count;
  }

  virtual double GetAccuracy() const {
    return avg_accuracy;
  }

protected:
  double avg_accuracy;
};

inline std::ostream& operator<<(std::ostream& s,
                                const AccuracyMetrics& metrics) {
  s << "Accuracy: " << metrics.GetAccuracy();
  return s;
}


struct ArgumentMetrics {
  std::unique_ptr<AccuracyMetrics> spans;
  std::unique_ptr<AccuracyMetrics> heads;
  double jaccard_index;
  unsigned instance_count;

  ArgumentMetrics(unsigned correct_spans = 0, unsigned incorrect_spans = 0,
                  unsigned heads_correct = 0, unsigned heads_incorrect = 0,
                  double jaccard_index = 0, unsigned instance_count = 0)
      : spans(new AccuracyMetrics(correct_spans, incorrect_spans)),
        heads(new AccuracyMetrics(heads_correct, heads_incorrect)),
        jaccard_index(jaccard_index), instance_count(instance_count) {}

  ArgumentMetrics(const ArgumentMetrics& other)
      : spans(new AccuracyMetrics(*other.spans)),
        heads(new AccuracyMetrics(*other.heads)),
        jaccard_index(other.jaccard_index),
        instance_count(other.instance_count) {}

  void operator+=(const ArgumentMetrics& other) {
    *spans += *other.spans;
    *heads += *other.heads;
    // Jaccard indices were already probably averages. So we have to preserve
    // the weighting for the new average.
    unsigned combined_instance_count = instance_count + other.instance_count;
    jaccard_index = (jaccard_index * instance_count
        + other.jaccard_index * other.instance_count) / combined_instance_count;
    instance_count = combined_instance_count;
  }

  ArgumentMetrics operator+(const ArgumentMetrics& other) const {
    ArgumentMetrics sum = *this;
    sum += other;
    return sum;
  }
};

class AveragedArgumentMetrics : public ArgumentMetrics {
public:
  template <class IteratorType>
  AveragedArgumentMetrics(boost::iterator_range<IteratorType> all_metrics) {
    auto all_spans = BOOST_TRANSFORMED_RANGE(all_metrics, m, *m.spans);
    auto all_heads = BOOST_TRANSFORMED_RANGE(all_metrics, m, *m.heads);
    auto all_jaccards = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                m.jaccard_index);
    auto all_instance_counts = BOOST_TRANSFORMED_RANGE(all_metrics, m,
                                                       m.instance_count);
    spans.reset(new AveragedAccuracyMetrics(all_spans));
    heads.reset(new AveragedAccuracyMetrics(all_heads));
    double weighted_jaccard_sum = boost::inner_product(
        all_instance_counts, all_jaccards, 0);
    instance_count = boost::accumulate(all_instance_counts, 0);
    jaccard_index = weighted_jaccard_sum / instance_count;
  }
};

inline std::ostream& operator<<(std::ostream& s,
                                const ArgumentMetrics& metrics) {
  s << "Spans:";
  {
    IndentingOStreambuf indent(std::cerr);
    s << '\n' << *metrics.spans;
  }
  s << "\nHeads:";
  {
    IndentingOStreambuf indent(std::cerr);
    s << '\n' << *metrics.heads;
  }
  s << "\nJaccard index: " << metrics.jaccard_index;
  return s;
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

// TODO: add partial matching?
template <class RelationType>
class BecauseRelationMetrics {
public:
  struct ConnectivesEqual {
    bool operator()(const RelationType& i1, const RelationType& i2) {
      return i1.GetConnectiveIndices() == i2.GetConnectiveIndices();
    }
  };

  std::unique_ptr<ClassificationMetrics> connective_metrics;
  std::vector<std::unique_ptr<ArgumentMetrics>> argument_metrics;
  typedef Diff<RandomAccessSequence<
      typename std::vector<RelationType>::const_iterator>, ConnectivesEqual>
      ConnectiveDiff;
  typedef Diff<RandomAccessSequence<
      typename BecauseRelation::IndexList::const_iterator>> IndexDiff;

  // Default constructor: initialize metrics with zero instances, and prepare
  // argument metrics to contain the correct number of entries for the relation
  // type.
  BecauseRelationMetrics()
      : connective_metrics(new ClassificationMetrics),
        argument_metrics(RelationType::ARG_NAMES.size()) {
    for (unsigned i = 0; i < argument_metrics.size(); ++i) {
      argument_metrics[i].reset(new ArgumentMetrics);
    }
  }

  BecauseRelationMetrics(const BecauseRelationMetrics& other)
      : connective_metrics(
          new ClassificationMetrics(*other.connective_metrics)),
          argument_metrics(other.argument_metrics.size()) {
    for (unsigned i = 0; i < argument_metrics.size(); ++i) {
      argument_metrics[i].reset(
          new ArgumentMetrics(*other.argument_metrics[i]));
    }
  }

  BecauseRelationMetrics(
      const std::vector<RelationType>& sentence_gold,
      const std::vector<RelationType>& sentence_predicted,
      const BecauseOracleTransitionCorpus& source_corpus,
      const GraphEnhancedParseTree& parse, const SpanTokenFilter& filter)
      : argument_metrics(NumArgs()) {
    unsigned tp = 0;
    unsigned fp = 0;
    unsigned fn = 0;

    ConnectiveDiff diff(sentence_gold, sentence_predicted);
    unsigned sentence_tp = diff.LCS().size();
    tp += sentence_tp;
    fp += sentence_predicted.size() - sentence_tp;
    fn += sentence_gold.size() - sentence_tp;

    const typename ConnectiveDiff::IndexList matching_gold =
        diff.OrigLCSIndices();
    const typename ConnectiveDiff::IndexList matching_pred =
        diff.NewLCSIndices();
    assert(matching_pred.size() == matching_gold.size());
    unsigned num_matching_connectives = matching_gold.size();

    connective_metrics.reset(new ClassificationMetrics(tp, fp, fn));

    for (unsigned arg_num : boost::irange(0u, NumArgs())) {
      unsigned spans_correct = 0;
      unsigned heads_correct = 0;
      unsigned gold_index;
      unsigned pred_index;
      double jaccard_sum = 0.0;
      for (auto both_indices : boost::combine(matching_gold, matching_pred)) {
        boost::tie(gold_index, pred_index) = both_indices;
        const auto& gold_span = sentence_gold[gold_index].GetArgument(arg_num);
        const auto& pred_span =
            sentence_predicted[pred_index].GetArgument(arg_num);
        // We're going to need to copy over the spans anyway for Jaccard index
        // calculations (diff requires random access). So we'll just copy and
        // filter the copy.
        BecauseRelation::IndexList filtered_gold(gold_span);
        ReallyDeleteIf(&filtered_gold, filter);
        BecauseRelation::IndexList filtered_pred(pred_span);
        ReallyDeleteIf(&filtered_pred, filter);

        if (boost::equal(filtered_gold, filtered_pred))
          ++spans_correct;
        if (GetHead(gold_span, parse, source_corpus)
            == GetHead(pred_span, parse, source_corpus))
          ++heads_correct;
        jaccard_sum += CalculateJaccard(filtered_gold, filtered_pred);
      }
      auto current_arg_metrics = new ArgumentMetrics(
          spans_correct, num_matching_connectives - spans_correct,
          heads_correct, num_matching_connectives - heads_correct,
          jaccard_sum / num_matching_connectives, num_matching_connectives);
      argument_metrics[arg_num].reset(current_arg_metrics);
    }
  }

  void operator+=(const BecauseRelationMetrics<RelationType>& other) {
    *connective_metrics += *other.connective_metrics;
    for (size_t i = 0; i < argument_metrics.size(); ++i) {
      *argument_metrics[i] += *other.argument_metrics[i];
    }
  }

  BecauseRelationMetrics<RelationType> operator+(
      const BecauseRelationMetrics<RelationType>& other) const {
    BecauseRelationMetrics sum = *this;
    sum += other;
    return sum;
  }

  unsigned GetHead(const typename RelationType::IndexList& span,
                   const GraphEnhancedParseTree& parse,
                   const BecauseOracleTransitionCorpus& source_corpus) {
    unsigned highest_token = -1;
    unsigned highest_token_depth = std::numeric_limits<unsigned>::max();
    for (unsigned token_id : span) {
      unsigned depth = parse.GetTokenDepth(token_id);
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
    // TODO: make this prefer certain heads over others of equal depth
    return highest_token;
  }

  static double CalculateJaccard(
      const typename RelationType::IndexList& gold,
      const typename RelationType::IndexList& predicted) {
    if (gold.empty() && predicted.empty()) {
      return 1.0;
    }
    IndexDiff diff(gold, predicted);
    unsigned num_matching = diff.LCS().size();
    return num_matching / static_cast<double>(gold.size() + predicted.size()
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

  static unsigned NumArgs() { return RelationType::ARG_NAMES.size(); }
};

template <class RelationType>
class AveragedBecauseRelationMetrics
    : public BecauseRelationMetrics<RelationType> {
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

  return s;
}


typedef BecauseRelationMetrics<CausalityRelation> CausalityMetrics;
typedef AveragedBecauseRelationMetrics<CausalityRelation>
    AveragedCausalityMetrics;


#endif /* LSTM_CAUSALITY_METRICS_H_ */
