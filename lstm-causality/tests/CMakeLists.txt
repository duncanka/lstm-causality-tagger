project(lstm-causality-tagger:tests)

set(CMAKE_BUILD_TYPE "Debug")

ADD_EXECUTABLE(tests metrics-test.cpp ../LSTMCausalityTagger.cpp
           ../BecauseOracleTransitionCorpus.cpp ../BecauseData.cpp)
target_link_libraries(tests gtest gtest_main pthread lstm-parser-core ${Boost_LIBRARIES})
