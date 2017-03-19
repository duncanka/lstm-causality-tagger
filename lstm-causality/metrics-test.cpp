#define METRICS_PRINT_RAW_COUNTS

#include <boost/range/combine.hpp>
#include <gtest/gtest.h>

#include "LSTMCausalityTagger.h"
#include "Metrics.h"

using namespace std;
using boost::combine;

struct MetricsTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    corpus.reset(
        new BecauseOracleTransitionCorpus(
            &vocab,
            "/home/jesse/Documents/Work/Research/Causality/Causeway/tests"
            "/resources/IAATest", true));
    for (const auto& sentence_and_actions : combine(corpus->sentences,
                                                    corpus->correct_act_sent)) {
      gold_relations.push_back(
          LSTMCausalityTagger::Decode(sentence_and_actions.head,
                                      sentence_and_actions.tail.head));
    }
  }

  static void TearDownTestCase() {
    corpus.release();
  }

  static lstm_parser::CorpusVocabulary vocab;
  static unique_ptr<BecauseOracleTransitionCorpus> corpus;
  static vector<vector<CausalityRelation>> gold_relations;
};

lstm_parser::CorpusVocabulary MetricsTest::vocab;
unique_ptr<BecauseOracleTransitionCorpus> MetricsTest::corpus;
vector<vector<CausalityRelation>> MetricsTest::gold_relations;


#define TEST_METRICS(calculated_metrics, correct_connective_metrics, \
                     correct_cause_metrics, correct_effect_metrics) \
  EXPECT_EQ(correct_connective_metrics, \
            *calculated_metrics.connective_metrics); \
  EXPECT_EQ(correct_cause_metrics, \
            *calculated_metrics.argument_metrics[CausalityRelation::CAUSE]); \
  EXPECT_EQ(correct_effect_metrics, \
            *calculated_metrics.argument_metrics[CausalityRelation::EFFECT]);


TEST_F(MetricsTest, SameAnnotationsTest) {
  ClassificationMetrics correct_connective_metrics(7, 0, 0);
  ArgumentMetrics correct_arg_metrics(7, 0, 7, 0, 1, 7);

  for (const auto& sentence_and_relations : combine(corpus->sentences,
                                                    gold_relations)) {
    const lstm_parser::Sentence& sentence = sentence_and_relations.head;
    const vector<CausalityRelation>& relations =
        sentence_and_relations.tail.head;
    SpanTokenFilter filter = {false, sentence, corpus->pos_is_punct};
    GraphEnhancedParseTree pseudo_parse(sentence);
    CausalityMetrics metrics(relations, relations, *corpus, pseudo_parse,
                             filter);
    ASSERT_EQ(correct_connective_metrics, correct_connective_metrics);
    ASSERT_EQ(correct_arg_metrics, correct_arg_metrics);
    TEST_METRICS(metrics, correct_connective_metrics, correct_arg_metrics,
                 correct_arg_metrics);
  }
}
