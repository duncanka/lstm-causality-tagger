#ifndef LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_
#define LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_

#include <boost/functional/hash.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <csignal>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    unsigned parse_path_arc_dim;
    unsigned parse_path_hidden_dim;
    unsigned span_hidden_dim;
    unsigned action_dim;
    unsigned state_dim;  // dimension for the concatenated tagger state
    double dropout;
    bool subtrees;
    bool gated_parse;
    bool new_conn_action;
    bool shift_action;
    bool known_conns_only;
    bool train_pairwise;
    // Non-serialized (runtime) options
    bool log_differences;
    bool oracle_connectives;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & word_dim;
      ar & lstm_layers;
      ar & token_dim;
      ar & lambda_hidden_dim;
      ar & actions_hidden_dim;
      ar & parse_path_arc_dim;
      ar & parse_path_hidden_dim;
      ar & span_hidden_dim;
      ar & action_dim;
      ar & state_dim;
      ar & dropout;
      ar & subtrees;
      ar & gated_parse;
      ar & new_conn_action;
      ar & shift_action;
      ar & known_conns_only;
      ar & train_pairwise;
    }
  };

  LSTMCausalityTagger(const std::string& parser_model_path,
                      const TaggerOptions& options);

  LSTMCausalityTagger(const std::string& parser_model_path,
                      const std::string& model_path)
      : parser(parser_model_path),
        sentence_lstms({L1_lstm, L2_lstm, L3_lstm, L4_lstm, action_history_lstm,
                        connective_lstm, cause_lstm, effect_lstm, means_lstm}),
        persistent_lstms({L1_lstm, L2_lstm, L3_lstm, L4_lstm,
                          action_history_lstm}) {
    initial_param_pool_state = cnn::ps->get_state();
    LoadModel(model_path);
  }

  virtual ~LSTMCausalityTagger() {}

  void Train(BecauseOracleTransitionCorpus* corpus,
             std::vector<unsigned> selections, double dev_pct,
             bool compare_punct, const std::string& model_fname,
             unsigned update_groups_between_evals = 25,
             double recent_improvements_cutoff = 0.85,
             double epochs_cutoff = std::numeric_limits<double>::infinity(),
             const volatile sig_atomic_t* requested_stop = nullptr);

  std::vector<CausalityRelation> Tag(const lstm_parser::Sentence& sentence,
                                     GraphEnhancedParseTree* parse);

  CausalityMetrics Evaluate(BecauseOracleTransitionCorpus* corpus,
                            const std::vector<unsigned>& selections,
                            bool compare_punct = false, bool pairwise = false);

  static std::vector<CausalityRelation> Decode(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned> actions, bool new_conn_is_action,
      bool shift_is_action);

  void Reset() {
    model.release();
    cnn::ps->restore_state(initial_param_pool_state);
    InitializeModelAndBuilders();
    InitializeNetworkParameters();
  }

  void LoadModel(const std::string& model_path);

  TaggerOptions options;

protected:
  struct CausalityTaggerState : public TaggerState {
    std::map<unsigned, Expression> all_tokens;
    std::map<unsigned, Expression> all_subtreeless_tokens;
    // For checking oracle connectives when we're using them.
    // Each oracle connective is stored as its initial connective word mapped to
    // a list of lists of connective fragments (one per connective).
    std::map<unsigned, std::vector<std::vector<unsigned>>> oracle_connectives;
    unsigned splits_seen_for_conn;

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
        : TaggerState {raw_sent, sent}, splits_seen_for_conn(0),
          currently_processing_rel(false), prev_action(-1),
          current_conn_token_i(sentence.begin()->first),
          current_arg_token_i(sentence.begin()->first) {
    }
  };

  virtual std::vector<cnn::Parameters*> GetParameters() override;

  virtual TaggerState* InitializeParserState(
      cnn::ComputationGraph* cg, const lstm_parser::Sentence& raw_sent,
      const lstm_parser::Sentence::SentenceMap& sentence,
      const std::vector<unsigned>& correct_actions) override;

  virtual void InitializeNetworkParameters() override;

  virtual bool ShouldTerminate(TaggerState* state) const override {
    const CausalityTaggerState& real_state =
        static_cast<const CausalityTaggerState&>(*state);
    // We're done when we're looking at the last word, and we're no longer
    // processing a relation, i.e., either the last word wasn't the start of a
    // connective or we already SHIFTed it off.
    return real_state.current_conn_token_i
              >= (++real_state.sentence.rbegin())->first  // skip ROOT
        && !real_state.currently_processing_rel;
  }

  virtual bool IsActionForbidden(const unsigned action,
                                 TaggerState* state) const override;

  virtual cnn::expr::Expression GetActionProbabilities(TaggerState* state)
      override;
  bool ShouldUseOracleTransition(CausalityTaggerState* state);

  virtual void DoAction(
      unsigned action, TaggerState* state, cnn::ComputationGraph* cg,
      std::map<std::string, cnn::expr::Expression>* states_to_expose) override;

  virtual void DoSave(eos::portable_oarchive& archive) {
    archive << *this;
  }

  double DoDevEvaluation(BecauseOracleTransitionCorpus* corpus,
                         const std::vector<unsigned>& selections,
                         bool compare_punct, unsigned num_sentences_train,
                         unsigned iteration, unsigned sentences_seen);

  void CacheParse(lstm_parser::Sentence* sentence,
                  BecauseOracleTransitionCorpus* corpus,
                  unsigned sentence_index,
                  const std::vector<unsigned>& parse_actions,
                  double parser_lp) const {
    if (!corpus->sentence_parses[sentence_index]) {
      auto tree = parser.RecoverParseTree(*sentence, parse_actions, parser_lp,
                                          true);
      auto tree_with_graph = new GraphEnhancedParseTree(std::move(tree));
      corpus->sentence_parses[sentence_index].reset(tree_with_graph);
      sentence->tree = tree_with_graph;
    }
  }

  void InitializeModelAndBuilders();

  lstm_parser::LSTMParser parser;
  cnn::AlignedMemoryPool::PoolState initial_param_pool_state;

  cnn::LSTMBuilder L1_lstm;  // unprocessed words to left of current word
  cnn::LSTMBuilder L2_lstm;  // processed words to left of current word
  cnn::LSTMBuilder L3_lstm;  // unprocessed words to right of current word
  cnn::LSTMBuilder L4_lstm;  // processed words to left of current word

  cnn::LSTMBuilder action_history_lstm;
  cnn::LSTMBuilder parse_path_lstm;  // path btw current word and current conn.

  cnn::LSTMBuilder connective_lstm;
  cnn::LSTMBuilder cause_lstm;
  cnn::LSTMBuilder effect_lstm;
  cnn::LSTMBuilder means_lstm;

  // Lookup parameters
  cnn::LookupParameters* p_w;    // word embeddings
  cnn::LookupParameters* p_a;    // action embeddings (for action_history_lstm)

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
  cnn::Parameters* p_parsepath2S;   // parse path to tagger state
  // Parameters for mixing in parse info (only used with --gated-parse)
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

  // Parameters for parse path embedding
  cnn::Parameters* p_pp_bias;
  cnn::Parameters* p_parse2pp;
  cnn::Parameters* p_token2pp;

  // LSTM guards
  cnn::Parameters* p_L1_guard;
  cnn::Parameters* p_L2_guard;
  cnn::Parameters* p_L3_guard;
  cnn::Parameters* p_L4_guard;
  cnn::Parameters* p_action_start;      // action bias
  cnn::Parameters* p_parse_path_start;  // parse path bias
  // Guards for per-instance LSTMs
  cnn::Parameters* p_connective_guard;
  cnn::Parameters* p_cause_guard;
  cnn::Parameters* p_effect_guard;
  cnn::Parameters* p_means_guard;

private:
  friend class boost::serialization::access;
  typedef std::map<std::string, cnn::expr::Expression> CachedExpressionMap;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar & options;
    ar & known_connectives;
    ar & vocab;
    ar & *model;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    finalized = false; // we'll need to re-finalize after resetting the network.

    ar & options;
    ar & known_connectives;
    ar & vocab;
    // Don't finalize yet...we want to finalize once our model is initialized.

    // Reset the LSTMs *before* reading in the network model, to make sure the
    // model knows how big it's supposed to be.
    InitializeModelAndBuilders();

    FinalizeVocab(); // OK, now finalize. :) (Also initializes network params.)

    ar & *model;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER();

  void StartNewRelation();

  Expression GetTokenEmbedding(cnn::ComputationGraph* cg, unsigned word_index,
                               unsigned word_id, unsigned pos_id,
                               bool no_subtrees=false);

  Expression GetParsePathEmbedding(CausalityTaggerState* state,
                                   unsigned source_token_id,
                                   unsigned dest_token_id);

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

  void RecordKnownConnectives(const std::vector<CausalityRelation>& rels);

  void FilterToPairwise(std::vector<CausalityRelation>* rels) {
    auto not_pairwise = [](const CausalityRelation& rel) {
      return rel.GetCause().empty() || rel.GetEffect().empty();
    };
    ReallyDeleteIf(rels, not_pairwise);
  }

  const std::vector<CausalityRelation>& GetDecodedGoldRelations(
      const lstm_parser::Sentence& sentence,
      const std::vector<unsigned>& actions);

  CachedExpressionMap parser_states;  // internal cache of parser NN states
  // Sets of word IDs that can go together in connectives.
  std::unordered_set<std::set<unsigned>,
                     boost::hash<std::set<unsigned>>> known_connectives;
  // Lists of LSTMs for convenience of iteration.
  std::vector<std::reference_wrapper<cnn::LSTMBuilder>> sentence_lstms;
  // LSTMs that persist across causal instances within a sentence.
  std::vector<std::reference_wrapper<cnn::LSTMBuilder>> persistent_lstms;
  std::unordered_map<const lstm_parser::Sentence*,
                     std::vector<CausalityRelation>> training_decoded_cache;
  unsigned conn_frag_action;  // Cached for GetActionProbabilities check
  unsigned split_action;      // Cached for GetActionProbabilities check
  unsigned oracle_actions_taken;
};

#endif /* LSTM_CAUSALITY_LSTMCAUSALITYTAGGER_H_ */
