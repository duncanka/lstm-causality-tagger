#ifndef LSTM_CAUSALITY_METRICS_H_
#define LSTM_CAUSALITY_METRICS_H_

#include <cmath>
#include <ostream>

#include "BecauseOracleTransitionCorpus.h"
#include "diff-cpp/lcs.h"
#include "utilities.h"


struct ClassificationMetrics {
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
    unsigned neg1 = static_cast<unsigned>(-1);
    tn = (tn == neg1 || other.tn == neg1 ? neg1 : tn + other.tn);
  }

  ClassificationMetrics operator+(const ClassificationMetrics& other) const {
    ClassificationMetrics sum = *this;
    sum += other;
    return sum;
  }

  bool operator==(const ClassificationMetrics& other) const {
    return tp == other.tp && fp == other.fp && fn == other.fn && tn == other.tn;
  }

  double GetAccuracy() const {
    if (tn != static_cast<unsigned>(-1)) {
      return static_cast<double>(tp + tn) / (tp + tn + fp + fn);
    } else {
      return std::nan("");
    }
  }

  double GetPrecision() const {
    return static_cast<double>(tp) / (tp + fp);
  }

  double GetRecall() const {
    return static_cast<double>(tp) / (tp + fn);
  }

  double GetF1() const {
    return CalculateF1(GetPrecision(), GetRecall());
  }

  static double CalculateF1(double precision, double recall) {
    return (2 * precision * recall) / (precision + recall);
  }
};

inline std::ostream& operator<<(std::ostream& s,
                         const ClassificationMetrics& metrics) {
  s << "Accuracy: " << metrics.GetAccuracy()
    << "\nPrecision: " << metrics.GetPrecision()
    << "\nRecall: " << metrics.GetRecall()
    << "\nF1: " << metrics.GetF1();
  return s;
}


struct AccuracyMetrics {
  unsigned correct;
  unsigned incorrect;

  AccuracyMetrics(unsigned correct, unsigned incorrect) :
      correct(correct), incorrect(incorrect) {}

  double GetAccuracy() const {
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

inline std::ostream& operator<<(std::ostream& s,
                                const AccuracyMetrics& metrics) {
  s << "Accuracy: " << metrics.GetAccuracy();
  return s;
}


struct ArgumentMetrics {
  AccuracyMetrics spans;
  AccuracyMetrics heads;
  double jaccard_index;

  ArgumentMetrics(unsigned correct = 0, unsigned incorrect = 0,
                  unsigned heads_correct = 0, unsigned heads_incorrect = 0,
                  double jaccard_index = 0)
      : spans(correct, incorrect), heads(heads_correct, heads_incorrect),
        jaccard_index(jaccard_index) {}

  void operator+=(const ArgumentMetrics& other) {
    spans += other.spans;
    heads += other.heads;
    jaccard_index += other.jaccard_index;
  }

  ArgumentMetrics operator+(const ArgumentMetrics& other) const {
    ArgumentMetrics sum = *this;
    sum += other;
    return sum;
  }
};

inline std::ostream& operator<<(std::ostream& s,
                                const ArgumentMetrics& metrics) {
  s << "Spans:";
  {
    IndentingOStreambuf indent(std::cout);
    s << '\n' << metrics.spans;
  }
  s << "\nHeads:";
  {
    IndentingOStreambuf indent(std::cout);
    s << '\n' << metrics.heads;
  }
  s << "\nJaccard index: " << metrics.jaccard_index;
  return s;
}



// TODO: add partial matching?
template <class RelationType>
struct BecauseRelationMetrics {
  struct ConnectivesEqual {
    bool operator()(const RelationType& i1, const RelationType& i2) {
      return i1.GetConnectiveIndices() == i2.GetConnectiveIndices();
    }
  };

  ClassificationMetrics connective_metrics;
  std::vector<ArgumentMetrics> argument_metrics;
  typedef Diff<RandomAccessSequence<
                   typename std::vector<RelationType>::const_iterator>,
               ConnectivesEqual> ConnectiveDiff;

  // Default constructor: initialize metrics with zero instances, and prepare
  // argument metrics to contain the correct number of entries for the relation
  // type.
  BecauseRelationMetrics()
      : argument_metrics(RelationType::ARG_NAMES.size()) {}

  BecauseRelationMetrics(
      const std::vector<RelationType>& sentence_gold,
      const std::vector<RelationType>& sentence_predicted)
  : argument_metrics(RelationType::ARG_NAMES.size()) {

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

    connective_metrics.tp = tp;
    connective_metrics.fp = fp;
    connective_metrics.fn = fn;

    // TODO: implement calculating argument metrics
    // TODO: implement head matching (requires having parse trees)
  }

  void operator+=(const BecauseRelationMetrics<RelationType>& other) {
    connective_metrics += other.connective_metrics;
    for (size_t i = 0; i < argument_metrics.size(); ++i) {
      argument_metrics[i] += other.argument_metrics[i];
    }
  }

  BecauseRelationMetrics<RelationType> operator+(
      const BecauseRelationMetrics<RelationType>& other) const {
    BecauseRelationMetrics sum = *this;
    sum += other;
    return sum;
  }
};

template <typename RelationType>
inline std::ostream& operator<<(std::ostream& s,
                         const BecauseRelationMetrics<RelationType>& metrics) {
  s << "Connectives:";
  {
    IndentingOStreambuf indent(s);
    s << '\n' << metrics.connective_metrics;
  }
  s << "\nArguments:";
  {
    IndentingOStreambuf indent(s);
    for (unsigned argi = 0; argi < RelationType::ARG_NAMES.size(); ++argi) {
      s << '\n' << RelationType::ARG_NAMES[argi] << ':';
      IndentingOStreambuf indent2(s);
      s << '\n' << metrics.argument_metrics[argi];
    }
  }

  return s;
}


typedef BecauseRelationMetrics<CausalityRelation> CausalityMetrics;


#endif /* LSTM_CAUSALITY_METRICS_H_ */
