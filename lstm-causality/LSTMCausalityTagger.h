#ifndef LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_
#define LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_

#include <boost/serialization/split_member.hpp>
#include <csignal>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "parser/neural-transition-tagger.h"
#include "BecauseData.h"
#include "BecauseOracleTransitionCorpus.h"
#include "cnn/model.h"
#include "parser/corpus.h"
#include "parser/lstm-parser.h"
#include "Metrics.h"

class LSTMCausalityTagger : public lstm_parser::NeuralTransitionTagger {
public:
  struct TaggerOptions {
    unsigned word_dim;
    unsigned lstm_layers;
    unsigned token_dim;
    unsigned lambda_hidden_dim;
    unsigned actions_hidden_dim;
    unsigned span_hidden_dim;
    unsigned rels_hidden_dim;
    unsigned action_dim;
    unsigned pos_dim;
    unsigned state_dim;  // dimension for the concatenated tagger state

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & word_dim;
      ar & lstm_layers;
      ar & token_dim;
      ar & lambda_hidden_dim;
      ar & actions_hidden_dim;
      ar & span_hidden_dim;
      ar & rels_hidden_dim;
      ar & action_dim;
      ar & pos_dim;
      ar & state_dim;
    }
  };

  TaggerOptions options;

  LSTMCausalityTagger(const std::string& parser_model_path,
                      const TaggerOptions& options);

  // TODO: add constructor for loading from model

  virtual ~LSTMCausalityTagger() {}

  void Train(BecauseOracleTransitionCorpus* corpus,
             std::vector<unsigned> selections, double dev_pct,
             bool compare_punct, const std::string& model_fname,
             double epochs_cutoff = std::numeric_limits<double>::infinity(),
             const volatile sig_atomic_t* requested_stop = nullptr);

  std::vector<CausalityRelation> Tag(const lstm_parser::Sentence& sentence,
                                     lstm_parser::ParseTree* parse = nullptr) {
    cnn::ComputationGraph cg;
    cnn::expr::Expression parser_state;
    vector<unsigned> parse_actions = parser.LogProbTagger(
        sentence, &cg, true, &parser_state);
    if (parse) {
      double parser_lp = as_scalar(cg.incremental_forward());
      auto tree = parser.RecoverParseTree(sentence, parse_actions, parser_lp,
                                          parse->IsLabeled());
      *parse = std::move(tree);
    }
    std::vector<unsigned> actions = LogProbTagger(sentence, &cg, false,
                                                  &parser_state);
    return Decode(sentence, actions);
  }

  CausalityMetrics Evaluate(const BecauseOracleTransitionCorpus& corpus,
                            const std::vector<unsigned>& selections,
                            bool compare_punct = false);

  std::vector<CausalityRelation> Decode(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned> actions);

  void Reset() {
    model = cnn::Model();
    InitializeNetworkParameters();
  }

protected:
  lstm_parser::LSTMParser parser;

  cnn::LSTMBuilder L1_lstm;  // unprocessed words to left of current word
  cnn::LSTMBuilder L2_lstm;  // processed words to left of current word
  cnn::LSTMBuilder L3_lstm;  // unprocessed words to right of current word
  cnn::LSTMBuilder L4_lstm;  // processed words to left of current word
  cnn::LSTMBuilder action_history_lstm;
  cnn::LSTMBuilder relations_lstm;

  cnn::LSTMBuilder connective_lstm;
  cnn::LSTMBuilder cause_lstm;
  cnn::LSTMBuilder effect_lstm;
  cnn::LSTMBuilder means_lstm;

  // Lookup parameters
  cnn::LookupParameters* p_w;  // word embeddings
  cnn::LookupParameters* p_t;  // pretrained word embeddings (not updated)
  cnn::LookupParameters* p_a;  // action embeddings (for action_history_lstm)
  cnn::LookupParameters* p_pos;  // pos tag embeddings

  // Parameters for overall tagger state
  cnn::Parameters* p_sbias;       // tagger state bias
  cnn::Parameters* p_L1toS;       // lambda 1 lstm to tagger state
  cnn::Parameters* p_L2toS;       // lambda 2 lstm to tagger state
  cnn::Parameters* p_L3toS;       // lambda 3 lstm to tagger state
  cnn::Parameters* p_L4toS;       // lambda 4 lstm to tagger state
  cnn::Parameters* p_actions2S;  // action history lstm to tagger state
  cnn::Parameters* p_rels2S;     // relations lstm to tagger state
  cnn::Parameters* p_s2a;         // parser state to action
  cnn::Parameters* p_abias;       // bias for final action output

  // Parameters for composition function for embedding a relation
  cnn::Parameters* p_rbias;           // bias for composition function
  cnn::Parameters* p_connective2rel;  // connective to relation embedding
  cnn::Parameters* p_cause2rel;       // cause to relation embedding
  cnn::Parameters* p_effect2rel;      // effect to relation embedding
  cnn::Parameters* p_means2rel;       // means to relation embedding

  // Parameters for LSTM input for stacks containing words
  cnn::Parameters* p_w2t;           // word to LSTM input
  cnn::Parameters* p_p2t;           // POS to LSTM input
  cnn::Parameters* p_v2t;           // pretrained word embeddings to LSTM input
  cnn::Parameters* p_tbias;            // LSTM input bias
  cnn::Parameters* p_action_start;  // action bias

  // LSTM guards (create biases for different LSTMs)
  cnn::Parameters* p_relations_guard;
  cnn::Parameters* p_L1_guard;
  cnn::Parameters* p_L2_guard;
  cnn::Parameters* p_L3_guard;
  cnn::Parameters* p_L4_guard;
  // Guards for per-instance LSTMs
  cnn::Parameters* p_connective_guard;
  cnn::Parameters* p_cause_guard;
  cnn::Parameters* p_effect_guard;
  cnn::Parameters* p_means_guard;

  struct CausalityTaggerState : public TaggerState {
    std::map<unsigned, Expression> all_tokens;
    std::vector<Expression> L1; // unprocessed tokens to the left
    std::vector<Expression> L2; // processed tokens to the left (reverse order)
    std::vector<Expression> L3; // unprocessed tokens to the right
    std::vector<Expression> L4; // processed tokens to the right (reverse order)
    std::vector<unsigned> L1i;
    std::vector<unsigned> L2i;
    std::vector<unsigned> L3i;
    std::vector<unsigned> L4i;
    bool currently_processing_rel;
    unsigned prev_action;
    // Need lists of token Expressions to duplicate if we get a SPLIT.
    std::vector<unsigned> current_rel_conn_tokens;
    std::vector<unsigned> current_rel_cause_tokens;
    std::vector<unsigned> current_rel_effect_tokens;
    std::vector<unsigned> current_rel_means_tokens;
    Expression current_conn_token;
    unsigned current_conn_token_i;
    Expression current_arg_token;
    unsigned current_arg_token_i;

    CausalityTaggerState(const lstm_parser::Sentence& raw_sent,
                         const lstm_parser::Sentence::SentenceMap& sent)
    : TaggerState{raw_sent, sent}, currently_processing_rel(false),
      prev_action(-1), current_conn_token_i(sentence.begin()->first),
      current_arg_token_i(sentence.begin()->first) {}
  };

  virtual std::vector<cnn::Parameters*> GetParameters() override {
    return {p_sbias, p_L1toS, p_L2toS, p_L3toS, p_L4toS, p_actions2S, p_rels2S,
      p_s2a, p_abias, p_rbias, p_connective2rel, p_cause2rel, p_effect2rel,
      p_means2rel, p_w2t, p_p2t, p_v2t, p_tbias, p_action_start,
      p_relations_guard, p_L1_guard, p_L2_guard, p_L3_guard, p_L4_guard,
      p_connective_guard, p_cause_guard, p_effect_guard, p_means_guard};
  }

  virtual TaggerState* InitializeParserState(
      cnn::ComputationGraph* cg, const lstm_parser::Sentence& raw_sent,
      const lstm_parser::Sentence::SentenceMap& sentence,
      const std::vector<unsigned>& correct_actions) override;

  virtual void InitializeNetworkParameters() override;

  virtual bool ShouldTerminate(const TaggerState& state) const override {
    const CausalityTaggerState& real_state =
        static_cast<const CausalityTaggerState&>(state);
    // We're done when we're looking at the last word, and we're no longer
    // processing a relation, i.e., either the last word wasn't the start of a
    // connective or we already SHIFTed it off.
    return real_state.current_conn_token_i
              >= real_state.sentence.rbegin()->first
        && !real_state.currently_processing_rel;
  }

  virtual bool IsActionForbidden(const unsigned action,
                                 const TaggerState& state) const override;

  virtual cnn::expr::Expression GetActionProbabilities(const TaggerState& state)
      override;

  virtual void DoAction(unsigned action,
                        TaggerState* state, cnn::ComputationGraph* cg) override;

  virtual void DoSave(eos::portable_oarchive& archive) {
    archive << *this;
  }

  double DoDevEvaluation(BecauseOracleTransitionCorpus* corpus,
                         const std::vector<unsigned>& selections,
                         bool compare_punct, unsigned num_sentences_train,
                         unsigned iteration, unsigned sentences_seen,
                         double best_f1, const std::string& model_fname,
                         double* last_epoch_saved);

  void CacheParse(const lstm_parser::Sentence& sentence,
                  const std::vector<unsigned>& parse_actions, double parser_lp,
                  BecauseOracleTransitionCorpus* corpus,
                  unsigned sentence_index) const {
    if (!corpus->sentence_parses[sentence_index]) {
      auto tree = parser.RecoverParseTree(sentence, parse_actions, parser_lp,
                                          true);
      corpus->sentence_parses[sentence_index].reset(
          new GraphEnhancedParseTree(std::move(tree)));
    }
  }

private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar & options;
    ar & vocab;
    ar & model;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    finalized = false; // we'll need to re-finalize after resetting the network.

    ar & options;
    ar & vocab;
    // Don't finalize yet...we want to finalize once our model is initialized.

    model = cnn::Model();
    // Reset the LSTMs *before* reading in the network model, to make sure the
    // model knows how big it's supposed to be.
    // TODO: initialize LSTM builders.

    FinalizeVocab(); // OK, now finalize. :)

    ar & model;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
