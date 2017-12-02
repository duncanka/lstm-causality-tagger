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
  double jaccard = CausalityMetrics::CalculateJaccard(gold, predicted, 0);
  EXPECT_DOUBLE_EQ(4/9., jaccard);
  jaccard = CausalityMetrics::CalculateJaccard(gold, predicted, 2);
  EXPECT_DOUBLE_EQ(4/11., jaccard);
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

TEST(MetricsTest, F1WorksInMetricsClasses) {
  unsigned tp = 1770;
  unsigned fp = 150;
  unsigned fn = 330;

  ArgumentMetrics am(tp, fp, fn, 0, tp + fp + fn);
  EXPECT_DOUBLE_EQ(0.880597014925373, am.spans->GetF1());

  ClassificationMetrics cm(tp, fp, fn);
  EXPECT_DOUBLE_EQ(0.880597014925373, cm.GetF1());
}


struct DataMetricsTest: public ::testing::Test {
protected:
  static void SetUpCorpus(const string& data_path,
                          unique_ptr<BecauseOracleTransitionCorpus>* corpus,
                          vector<vector<CausalityRelation>>* relations) {
    corpus->reset(
        new BecauseOracleTransitionCorpus(vocab.get(), data_path, true, false));
    for (unsigned i = 0; i < (*corpus)->sentences.size(); ++i) {
      const lstm_parser::Sentence& sentence = (*corpus)->sentences[i];
      const vector<unsigned>& actions = (*corpus)->correct_act_sent[i];
      relations->push_back(LSTMCausalityTagger::Decode(sentence, actions,
                                                       false, true));
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
      const vector<vector<CausalityRelation>>& rels2,
      bool record_differences = false) {
    CausalityMetrics total_metrics;

    for (unsigned i = 0; i < corpus1.sentences.size(); ++i) {
      const lstm_parser::Sentence& sentence = corpus1.sentences[i];
      const vector<CausalityRelation>& original_rels = rels1[i];
      const vector<CausalityRelation>& modified_rels = rels2[i];

      SpanTokenFilter filter = {false, sentence, corpus1.pos_is_punct};
      GraphEnhancedParseTree pseudo_parse(sentence);
      CausalityMetrics sentence_metrics(original_rels, modified_rels, corpus1,
                                        pseudo_parse, filter, 0, {},
                                        record_differences);
      total_metrics += sentence_metrics;
    }

    return total_metrics;
  }

  inline CausalityMetrics CompareOriginalAndModified() {
    return CompareCorpora(*original_corpus, *modified_corpus,
                          original_relations, modified_relations);
  }

  static void SetUpTestCase() {
    vocab.reset(new lstm_parser::CorpusVocabulary);
    SetUpCorpus("lstm-causality/tests/data/original", &original_corpus,
                &original_relations);
    SetUpCorpus("lstm-causality/tests/data/modified", &modified_corpus,
                &modified_relations);
  }

  static void TearDownTestCase() {
    original_corpus.release();
  }

  static unique_ptr<lstm_parser::CorpusVocabulary> vocab;
  static unique_ptr<BecauseOracleTransitionCorpus> original_corpus;
  static vector<vector<CausalityRelation>> original_relations;
  static unique_ptr<BecauseOracleTransitionCorpus> modified_corpus;
  static vector<vector<CausalityRelation>> modified_relations;
};

unique_ptr<lstm_parser::CorpusVocabulary> DataMetricsTest::vocab;
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
  ArgumentMetrics correct_cause_metrics(6, 0, 0, 1, 7);
  ArgumentMetrics correct_effect_metrics(7, 0, 0, 1, 7);
  ASSERT_EQ(correct_connective_metrics, correct_connective_metrics);
  ASSERT_EQ(correct_cause_metrics, correct_cause_metrics);
  TEST_METRICS(self_metrics, correct_connective_metrics, correct_cause_metrics,
               correct_effect_metrics);
}


TEST_F(DataMetricsTest, ModifiedAnnotationsGivesLessPerfectScores) {
  CausalityMetrics compared_metrics = CompareOriginalAndModified();
  ClassificationMetrics correct_connective_metrics(5, 2, 2);
  // One cause changed -> arg FP+FN; one cause deleted -> arg FN. Plus the 2 FPs
  // and FNs from connectives.
  ArgumentMetrics correct_cause_metrics(2, 3, 4, 0.6);
  ArgumentMetrics correct_effect_metrics(4, 3, 3, 33/35.);
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
  ArgumentMetrics correct_cause_metrics(4, 6, 8, 0.45);
  ArgumentMetrics correct_effect_metrics(8, 6, 6, 34/35.);  // 34 = mean(33, 35)
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
  AveragedCausalityMetrics correct_metrics(  // start off averaging nothing
      make_iterator_range(to_average.begin(), to_average.begin()));

  // Manually set everything for the correct averaged metrics.
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
  AveragedClassificationMetrics *cause_spans =
      static_cast<AveragedClassificationMetrics *>(
          correct_cause_metrics->spans.get());
  tie(cause_spans->tp, cause_spans->fp, cause_spans->fn,
      correct_cause_metrics->jaccard_index) =
          make_tuple(4, 1.5, 2, (1 + 0.6) / 2);
  precision = (1 + 2/5.) / 2;
  recall = (1 + 2/6.) / 2;
  f1 = (1 + ClassificationMetrics::CalculateF1(2/5., 2/6.)) / 2;
  tie(cause_spans->avg_precision, cause_spans->avg_recall, cause_spans->avg_f1)
      = make_tuple(precision, recall, f1);

  AveragedArgumentMetrics* correct_effect_metrics =
      static_cast<AveragedArgumentMetrics*>(
          correct_metrics.argument_metrics[EFFECT_INDEX].get());
  AveragedClassificationMetrics *effect_spans =
      static_cast<AveragedClassificationMetrics *>(
          correct_effect_metrics->spans.get());
  tie(effect_spans->tp, effect_spans->fp, effect_spans->fn,
      correct_effect_metrics->jaccard_index) =
          make_tuple(5.5, 1.5, 1.5, 34 / 35.);
  double p_r_f1 = (4/7. + 1) / 2;
  tie(effect_spans->avg_precision, effect_spans->avg_recall,
      effect_spans->avg_f1) = make_tuple(p_r_f1, p_r_f1, p_r_f1);

  TEST_METRICS(averaged_metrics, *correct_conn_metrics,
               *correct_cause_metrics, *correct_effect_metrics);
}


TEST_F(DataMetricsTest, CorrectNumberOfDifferencesRecorded) {
  CausalityMetrics self_metrics = CompareCorpora(
      *original_corpus, *original_corpus, original_relations,
      original_relations, true);
  EXPECT_EQ(7, self_metrics.GetArgumentMatches().size());
  EXPECT_EQ(0, self_metrics.GetArgumentMismatches().size());
  EXPECT_EQ(0, self_metrics.GetFPs().size());
  EXPECT_EQ(0, self_metrics.GetFNs().size());

  CausalityMetrics modified_metrics = CompareCorpora(
      *original_corpus, *modified_corpus, original_relations,
      modified_relations, true);
  // 2 totally correct; 2 FPs; 2 FNs; 2 correct connectives with incorrect
  // arguments, with 3 incorrect arguments between them.
  EXPECT_EQ(2, modified_metrics.GetArgumentMatches().size());
  EXPECT_EQ(3, modified_metrics.GetArgumentMismatches().size());
  EXPECT_EQ(2, modified_metrics.GetFPs().size());
  EXPECT_EQ(2, modified_metrics.GetFNs().size());
}
