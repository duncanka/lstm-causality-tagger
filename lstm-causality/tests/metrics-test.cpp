#define METRICS_PRINT_RAW_COUNTS

#include <boost/range/combine.hpp>
#include <gtest/gtest.h>

#include "../LSTMCausalityTagger.h"
#include "../Metrics.h"

using namespace std;
using boost::combine;


TEST(MetricsTest, JaccardIndexWorks) {
  BecauseRelation::IndexList gold = {1, 2, 3, 4, 5, 6, 7};
  BecauseRelation::IndexList predicted = {1, 3, 5, 7, 9, 11};
  double jaccard = CausalityMetrics::CalculateJaccard(gold, predicted);
  EXPECT_DOUBLE_EQ(4/9., jaccard);
}


struct DataMetricsTest: public ::testing::Test {
protected:
  static void SetUpCorpus(const string& data_path,
                          unique_ptr<BecauseOracleTransitionCorpus>* corpus,
                          vector<vector<CausalityRelation>>* relations) {
    corpus->reset(
        new BecauseOracleTransitionCorpus(&vocab, data_path, true));
    for (const auto& sentence_and_actions : combine(
        (*corpus)->sentences, (*corpus)->correct_act_sent)) {
      relations->push_back(
          LSTMCausalityTagger::Decode(sentence_and_actions.head,
                                      sentence_and_actions.tail.head));
    }
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
            *calculated_metrics.argument_metrics[CausalityRelation::CAUSE]); \
  EXPECT_EQ(correct_effect_metrics, \
            *calculated_metrics.argument_metrics[CausalityRelation::EFFECT]);


TEST_F(DataMetricsTest, SameAnnotationGetsPerfectScores) {
  ClassificationMetrics correct_connective_metrics(7, 0, 0);
  ArgumentMetrics correct_arg_metrics(7, 0, 7, 0, 1, 7);

  CausalityMetrics total_metrics;

  for (const auto& sentence_and_relations : combine(original_corpus->sentences,
                                                    original_relations)) {
    const lstm_parser::Sentence& sentence = sentence_and_relations.head;
    const vector<CausalityRelation>& relations =
        sentence_and_relations.tail.head;
    SpanTokenFilter filter = {false, sentence, original_corpus->pos_is_punct};
    GraphEnhancedParseTree pseudo_parse(sentence);
    CausalityMetrics sentence_metrics(relations, relations, *original_corpus,
                                      pseudo_parse, filter);
    total_metrics += sentence_metrics;
  }

  ASSERT_EQ(correct_connective_metrics, correct_connective_metrics);
  ASSERT_EQ(correct_arg_metrics, correct_arg_metrics);
  TEST_METRICS(total_metrics, correct_connective_metrics, correct_arg_metrics,
               correct_arg_metrics);
}

TEST_F(DataMetricsTest, ModifiedAnnotationsGivesLessPerfectScores) {
  ClassificationMetrics correct_connective_metrics(5, 2, 2);
  ArgumentMetrics correct_cause_metrics(3, 2, 3, 2, 0.6);
  ArgumentMetrics correct_effect_metrics(4, 1, 5, 0, 33/35.);

  CausalityMetrics total_metrics;

  for (unsigned i = 0; i < original_corpus->sentences.size(); ++i) {
    const lstm_parser::Sentence& sentence = original_corpus->sentences[i];
    const vector<CausalityRelation>& original_rels = original_relations[i];
    const vector<CausalityRelation>& modified_rels = modified_relations[i];

    SpanTokenFilter filter = {false, sentence, original_corpus->pos_is_punct};
    GraphEnhancedParseTree pseudo_parse(sentence);
    CausalityMetrics sentence_metrics(original_rels, modified_rels,
                                      *original_corpus, pseudo_parse, filter);
    total_metrics += sentence_metrics;
  }

  TEST_METRICS(total_metrics, correct_connective_metrics, correct_cause_metrics,
               correct_effect_metrics);
}
