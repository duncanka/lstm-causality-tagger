project(lstm-causality-tagger:tests)

ADD_EXECUTABLE(tests metrics-test.cpp ../LSTMCausalityTagger.cpp
           ../BecauseOracleTransitionCorpus.cpp ../BecauseData.cpp)
target_link_libraries(tests gtest gtest_main pthread lstm-parser-core ${Boost_LIBRARIES})
