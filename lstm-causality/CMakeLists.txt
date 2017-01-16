project(lstm-causality-tagger:tagger)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(Boost COMPONENTS program_options filesystem)

ADD_EXECUTABLE(lstm-causality-tagger tagger-driver.cc LSTMCausalityTagger.cpp
               BecauseOracleTransitionCorpus.cpp)
target_link_libraries(lstm-causality-tagger lstm-parser-core ${Boost_LIBRARIES})
