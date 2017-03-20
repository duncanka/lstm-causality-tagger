#define METRICS_PRINT_RAW_COUNTS

#include <boost/range/iterator_range.hpp>
#include <gtest/gtest.h>

#include "../LSTMCausalityTagger.h"
#include "../Metrics.h"

using namespace std;
using boost::make_iterator_range;

constexpr unsigned CAUSE_INDEX = CausalityRelation::CAUSE;
constexpr unsigned EFFECT_INDEX = CausalityRelation::EFFECT;


TEST(MetricsTest, JaccardIndexWorks) {
  BecauseRelation::IndexList gold = {1, 2, 3, 4, 5, 6, 7};
  BecauseRelation::IndexList predicted = {1, 3, 5, 7, 9, 11};
  double jaccard = CausalityMetrics::CalculateJaccard(gold, predicted);
  EXPECT_DOUBLE_EQ(4/9., jaccard);
}

TEST(MetricsTest, F1Works) {
  unsigned tp = 1770;
  unsigned fp = 150;
  unsigned fn = 330;
  double f1 = ClassificationMetrics::CalculateF1(
      ClassificationMetrics::CalculatePrecision(tp, fp),
      ClassificationMetrics::CalculateRecall(tp, fn));
  EXPECT_DOUBLE_EQ(0.880597014925373, f1);
}


struct DataMetricsTest: public ::testing::Test {
protected:
  static void SetUpCorpus(const string& data_path,
                          unique_ptr<BecauseOracleTransitionCorpus>* corpus,
                          vector<vector<CausalityRelation>>* relations) {
    corpus->reset(
        new BecauseOracleTransitionCorpus(&vocab, data_path, true));
    for (unsigned i = 0; i < (*corpus)->sentences.size(); ++i) {
      const lstm_parser::Sentence& sentence = (*corpus)->sentences[i];
      const vector<unsigned>& actions = (*corpus)->correct_act_sent[i];
      relations->push_back(LSTMCausalityTagger::Decode(sentence, actions));
      /*
      cerr << sentence << endl;
      for (const auto& rel : relations->back()) {
        cerr << rel << endl;
      }
      //*/
    }
  }

  inline CausalityMetrics CompareCorpora(
      const BecauseOracleTransitionCorpus& corpus1,
      const BecauseOracleTransitionCorpus& corpus2,
      const vector<vector<CausalityRelation>>& rels1,
      const vector<vector<CausalityRelation>>& rels2) {
    CausalityMetrics total_metrics;

    for (unsigned i = 0; i < corpus1.sentences.size(); ++i) {
      const lstm_parser::Sentence& sentence = corpus1.sentences[i];
      const vector<CausalityRelation>& original_rels = rels1[i];
      const vector<CausalityRelation>& modified_rels = rels2[i];

      SpanTokenFilter filter = {false, sentence, corpus1.pos_is_punct};
      GraphEnhancedParseTree pseudo_parse(sentence);
      CausalityMetrics sentence_metrics(original_rels, modified_rels, corpus1,
                                        pseudo_parse, filter);
      total_metrics += sentence_metrics;
    }

    return total_metrics;
  }

  inline CausalityMetrics CompareOriginalAndModified() {
    return CompareCorpora(*original_corpus, *modified_corpus,
                          original_relations, modified_relations);
  }

  static void SetUpTestCase() {
    SetUpCorpus("lstm-causality/tests/data/original", &original_corpus,
                &original_relations);
    SetUpCorpus("lstm-causality/tests/data/modified", &modified_corpus,
                &modified_relations);
  }

  static void TearDownTestCase() {
    original_corpus.release();
  }

  static lstm_parser::CorpusVocabulary vocab;
  static unique_ptr<BecauseOracleTransitionCorpus> original_corpus;
  static vector<vector<CausalityRelation>> original_relations;
  static unique_ptr<BecauseOracleTransitionCorpus> modified_corpus;
  static vector<vector<CausalityRelation>> modified_relations;
};

lstm_parser::CorpusVocabulary DataMetricsTest::vocab;
unique_ptr<BecauseOracleTransitionCorpus> DataMetricsTest::original_corpus;
vector<vector<CausalityRelation>> DataMetricsTest::original_relations;
unique_ptr<BecauseOracleTransitionCorpus> DataMetricsTest::modified_corpus;
vector<vector<CausalityRelation>> DataMetricsTest::modified_relations;


#define TEST_METRICS(calculated_metrics, correct_connective_metrics, \
                     correct_cause_metrics, correct_effect_metrics) \
  EXPECT_EQ(correct_connective_metrics, \
            *calculated_metrics.connective_metrics); \
  EXPECT_EQ(correct_cause_metrics, \
            *calculated_metrics.argument_metrics[CAUSE_INDEX]); \
  EXPECT_EQ(correct_effect_metrics, \
            *calculated_metrics.argument_metrics[EFFECT_INDEX]);


TEST_F(DataMetricsTest, SameAnnotationGetsPerfectScores) {
  CausalityMetrics self_metrics = CompareCorpora(
      *original_corpus, *original_corpus, original_relations,
      original_relations);

  ClassificationMetrics correct_connective_metrics(7, 0, 0);
  ArgumentMetrics correct_arg_metrics(7, 0, 7, 0, 1, 7);
  ASSERT_EQ(correct_connective_metrics, correct_connective_metrics);
  ASSERT_EQ(correct_arg_metrics, correct_arg_metrics);
  TEST_METRICS(self_metrics, correct_connective_metrics, correct_arg_metrics,
               correct_arg_metrics);
}


TEST_F(DataMetricsTest, ModifiedAnnotationsGivesLessPerfectScores) {
  CausalityMetrics compared_metrics = CompareOriginalAndModified();
  ClassificationMetrics correct_connective_metrics(5, 2, 2);
  ArgumentMetrics correct_cause_metrics(3, 2, 3, 2, 0.6);
  ArgumentMetrics correct_effect_metrics(4, 1, 5, 0, 33/35.);
  TEST_METRICS(compared_metrics, correct_connective_metrics,
               correct_cause_metrics, correct_effect_metrics);
}


TEST_F(DataMetricsTest, AddingMetricsWorks) {
  CausalityMetrics compared_metrics = CompareOriginalAndModified();
  CausalityMetrics tweaked_metrics = compared_metrics;
  tweaked_metrics.argument_metrics[CAUSE_INDEX]->jaccard_index = 0.3;
  tweaked_metrics.argument_metrics[EFFECT_INDEX]->jaccard_index = 1.0;
  CausalityMetrics summed_metrics = compared_metrics + tweaked_metrics;

  ClassificationMetrics correct_connective_metrics(10, 4, 4);
  ArgumentMetrics correct_cause_metrics(6, 4, 6, 4, 0.45);
  ArgumentMetrics correct_effect_metrics(8, 2, 10, 0, 34/35.);
  TEST_METRICS(summed_metrics, correct_connective_metrics,
               correct_cause_metrics, correct_effect_metrics);
}


TEST_F(DataMetricsTest, AveragingMetricsWorks) {
  CausalityMetrics compared_metrics = CompareOriginalAndModified();

  vector<CausalityMetrics> repeated = {compared_metrics, compared_metrics,
                                       compared_metrics};
  AveragedCausalityMetrics avged_same(make_iterator_range(repeated));
  TEST_METRICS(avged_same, *compared_metrics.connective_metrics,
               *compared_metrics.argument_metrics[CAUSE_INDEX],
               *compared_metrics.argument_metrics[EFFECT_INDEX]);

  CausalityMetrics self_metrics = CompareCorpora(
      *original_corpus, *original_corpus, original_relations,
      original_relations);
  vector<CausalityMetrics> to_average = {compared_metrics, self_metrics};
  AveragedCausalityMetrics averaged_metrics(make_iterator_range(to_average));

  AveragedCausalityMetrics correct_metrics(
      make_iterator_range(to_average.begin(), to_average.begin()));
  // Manually set everything in the averaged metrics.
  AveragedClassificationMetrics* correct_conn_metrics =
      static_cast<AveragedClassificationMetrics*>(
          correct_metrics.connective_metrics.get());
  double precision = ClassificationMetrics::CalculatePrecision(6, 1);
  double recall = ClassificationMetrics::CalculateRecall(6, 1);
  double f1 = ClassificationMetrics::CalculateF1(precision, recall);
  tie(correct_conn_metrics->tp, correct_conn_metrics->fp,
      correct_conn_metrics->fn, correct_conn_metrics->avg_accuracy,
      correct_conn_metrics->avg_precision,
      correct_conn_metrics->avg_recall, correct_conn_metrics->avg_f1) =
          make_tuple(6, 1, 1, nan(""), precision, recall, f1);

  AveragedArgumentMetrics* correct_cause_metrics =
      static_cast<AveragedArgumentMetrics*>(
          correct_metrics.argument_metrics[CAUSE_INDEX].get());
  tie(correct_cause_metrics->spans->correct,
      correct_cause_metrics->spans->incorrect,
      static_cast<AveragedAccuracyMetrics*>(
          correct_cause_metrics->spans.get())->avg_accuracy,
      correct_cause_metrics->heads->correct,
      correct_cause_metrics->heads->incorrect,
      static_cast<AveragedAccuracyMetrics*>(
          correct_cause_metrics->heads.get())->avg_accuracy,
      correct_cause_metrics->jaccard_index) =
          make_tuple(5, 1, 0.8, 5, 1, 0.8, (1 + 0.6) / 2);

  AveragedArgumentMetrics* correct_effect_metrics =
      static_cast<AveragedArgumentMetrics*>(
          correct_metrics.argument_metrics[EFFECT_INDEX].get());
  tie(correct_effect_metrics->spans->correct,
      correct_effect_metrics->spans->incorrect,
      static_cast<AveragedAccuracyMetrics*>(
          correct_effect_metrics->spans.get())->avg_accuracy,
      correct_effect_metrics->heads->correct,
      correct_effect_metrics->heads->incorrect,
      static_cast<AveragedAccuracyMetrics*>(
          correct_effect_metrics->heads.get())->avg_accuracy,
      correct_effect_metrics->jaccard_index) =
          make_tuple(6, 1, 0.9, 6, 0, 0.9, 34 / 35.);

  TEST_METRICS(averaged_metrics, *correct_conn_metrics,
               *correct_cause_metrics, *correct_effect_metrics);
}
