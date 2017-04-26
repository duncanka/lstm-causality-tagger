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
    unsigned action_dim;
    unsigned pos_dim;
    unsigned state_dim;  // dimension for the concatenated tagger state
    double dropout;
    bool subtrees;
    bool gated_parse;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & word_dim;
      ar & lstm_layers;
      ar & token_dim;
      ar & lambda_hidden_dim;
      ar & actions_hidden_dim;
      ar & span_hidden_dim;
      ar & action_dim;
      ar & pos_dim;
      ar & state_dim;
      ar & dropout;
      ar & subtrees;
      ar & gated_parse;
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
             unsigned periods_between_evals = 25,
             double epochs_cutoff = std::numeric_limits<double>::infinity(),
             const volatile sig_atomic_t* requested_stop = nullptr);

  std::vector<CausalityRelation> Tag(const lstm_parser::Sentence& sentence,
                                     lstm_parser::ParseTree* parse = nullptr) {
    cnn::ComputationGraph cg;
    vector<unsigned> parse_actions = parser.LogProbTagger(
        &cg, sentence, true, &parser_states);
    if (parse) {
      double parser_lp = as_scalar(cg.incremental_forward());
      auto tree = parser.RecoverParseTree(sentence, parse_actions, parser_lp,
                                          parse->IsLabeled());
      *parse = std::move(tree);
    }
    std::vector<unsigned> actions = LogProbTagger(&cg, sentence, false);
    return Decode(sentence, actions);
  }

  CausalityMetrics Evaluate(const BecauseOracleTransitionCorpus& corpus,
                            const std::vector<unsigned>& selections,
                            bool compare_punct = false);

  static std::vector<CausalityRelation> Decode(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned> actions);

  void Reset() {
    model.release();
    cnn::ps->restore_state(initial_param_pool_state);
    InitializeModelAndBuilders();
    InitializeNetworkParameters();
  }

protected:
  lstm_parser::LSTMParser parser;
  cnn::AlignedMemoryPool::PoolState initial_param_pool_state;

  cnn::LSTMBuilder L1_lstm;  // unprocessed words to left of current word
  cnn::LSTMBuilder L2_lstm;  // processed words to left of current word
  cnn::LSTMBuilder L3_lstm;  // unprocessed words to right of current word
  cnn::LSTMBuilder L4_lstm;  // processed words to left of current word
  cnn::LSTMBuilder action_history_lstm;

  cnn::LSTMBuilder connective_lstm;
  cnn::LSTMBuilder cause_lstm;
  cnn::LSTMBuilder effect_lstm;
  cnn::LSTMBuilder means_lstm;

  // Lookup parameters
  cnn::LookupParameters* p_w;    // word embeddings
  cnn::LookupParameters* p_t;    // pretrained word embeddings (not updated)
  cnn::LookupParameters* p_a;    // action embeddings (for action_history_lstm)
  cnn::LookupParameters* p_pos;  // pos tag embeddings

  // Parameters for overall tagger state
  cnn::Parameters* p_sbias;         // tagger state bias
  cnn::Parameters* p_L1toS;         // lambda 1 lstm to tagger state
  cnn::Parameters* p_L2toS;         // lambda 2 lstm to tagger state
  cnn::Parameters* p_L3toS;         // lambda 3 lstm to tagger state
  cnn::Parameters* p_L4toS;         // lambda 4 lstm to tagger state
  cnn::Parameters* p_current2S;     // current token to tagger state
  cnn::Parameters* p_actions2S;     // action history lstm to tagger state
  cnn::Parameters* p_s2a;           // parser state to action
  cnn::Parameters* p_abias;         // bias for final action output
  cnn::Parameters* p_connective2S;  // connective to relation embedding
  cnn::Parameters* p_cause2S;       // cause to relation embedding
  cnn::Parameters* p_effect2S;      // effect to relation embedding
  cnn::Parameters* p_means2S;       // means to relation embedding
  // Parameters for mixing in parse info
  cnn::Parameters* p_parse_sel_bias;
  cnn::Parameters* p_state_to_parse_sel;
  cnn::Parameters* p_parse2sel;
  cnn::Parameters* p_full_state_bias;
  cnn::Parameters* p_parse2pstate;
  cnn::Parameters* p_state2pstate;

  // Parameters for LSTM input for stacks containing tokens
  cnn::Parameters* p_w2t;           // word to token representation
  cnn::Parameters* p_p2t;           // POS to token representation
  cnn::Parameters* p_v2t;           // pretrained word embeddings to token repr
  cnn::Parameters* p_tbias;         // LSTM input bias
  cnn::Parameters* p_subtree2t;     // node subtree parse info to token

  // LSTM guards (create biases for different LSTMs)
  cnn::Parameters* p_L1_guard;
  cnn::Parameters* p_L2_guard;
  cnn::Parameters* p_L3_guard;
  cnn::Parameters* p_L4_guard;
  cnn::Parameters* p_action_start;  // action bias
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
    std::vector<cnn::Parameters*> params = {
        p_sbias, p_L1toS, p_L2toS, p_L3toS, p_L4toS, p_current2S, p_actions2S,
        p_connective2S, p_cause2S, p_effect2S, p_means2S,
        p_abias, p_s2a,
        p_tbias, p_w2t, p_p2t, p_v2t,
        p_action_start, p_L1_guard, p_L2_guard, p_L3_guard, p_L4_guard,
        p_connective_guard, p_cause_guard, p_effect_guard, p_means_guard};
    if (options.gated_parse) {
      auto to_add = {p_parse_sel_bias, p_state_to_parse_sel, p_parse2sel,
                     p_full_state_bias, p_parse2pstate, p_state2pstate};
      params.insert(params.end(), to_add.begin(), to_add.end());
    }
    if (options.subtrees) {
      params.push_back(p_subtree2t);
    }
    return params;
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
              >= (++real_state.sentence.rbegin())->first  // skip ROOT
        && !real_state.currently_processing_rel;
  }

  virtual bool IsActionForbidden(const unsigned action,
                                 const TaggerState& state) const override;

  virtual cnn::expr::Expression GetActionProbabilities(const TaggerState& state)
      override;

  virtual void DoAction(
      unsigned action, TaggerState* state, cnn::ComputationGraph* cg,
      std::map<std::string, cnn::expr::Expression>* states_to_expose) override;

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

  void InitializeModelAndBuilders();

private:
  friend class boost::serialization::access;
  typedef std::map<std::string, cnn::expr::Expression> CachedExpressionMap;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar & options;
    ar & vocab;
    ar & *model;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    finalized = false; // we'll need to re-finalize after resetting the network.

    ar & options;
    ar & vocab;
    // Don't finalize yet...we want to finalize once our model is initialized.

    // Reset the LSTMs *before* reading in the network model, to make sure the
    // model knows how big it's supposed to be.
    InitializeModelAndBuilders();

    FinalizeVocab(); // OK, now finalize. :)

    ar & *model;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  void StartNewRelation() {
    connective_lstm.start_new_sequence();
    cause_lstm.start_new_sequence();
    effect_lstm.start_new_sequence();
    means_lstm.start_new_sequence();

    connective_lstm.add_input(GetParamExpr(p_connective_guard));
    cause_lstm.add_input(GetParamExpr(p_cause_guard));
    effect_lstm.add_input(GetParamExpr(p_effect_guard));
    means_lstm.add_input(GetParamExpr(p_means_guard));
  }

  Expression GetTokenExpression(cnn::ComputationGraph* cg, unsigned word_index,
                                unsigned word_id, unsigned pos_id);

  CachedExpressionMap* GetCachedParserStates() {
    return (options.gated_parse || options.subtrees) ? &parser_states : nullptr;
  }

  Expression RectifyWithDropout(Expression e) {
    if (in_training && options.dropout) {
      return cnn::expr::rectify(cnn::expr::dropout(e, options.dropout));
    } else {
      return cnn::expr::rectify(e);
    }
  }

  CachedExpressionMap parser_states;  // internal cache of parser NN states
  std::vector<std::reference_wrapper<cnn::LSTMBuilder>> all_lstms;
  std::vector<std::reference_wrapper<cnn::LSTMBuilder>> persistent_lstms;
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
