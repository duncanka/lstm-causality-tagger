project(lstm-causality-tagger:tagger)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(Boost COMPONENTS program_options filesystem system)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_library(causality-tagger-core OBJECT LSTMCausalityTagger.cc
                BecauseOracleTransitionCorpus.cc BecauseData.cc)
    ADD_EXECUTABLE(lstm-causality-tagger tagger-driver.cc $<TARGET_OBJECTS:causality-tagger-core>)
    target_link_libraries(lstm-causality-tagger lstm-parser-core ${Boost_LIBRARIES})
    # Extra target: tests
    ADD_EXECUTABLE(tester tests/metrics-test.cc $<TARGET_OBJECTS:causality-tagger-core>)
    target_link_libraries(tester gtest gtest_main pthread lstm-parser-core ${Boost_LIBRARIES})
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ADD_EXECUTABLE(lstm-causality-tagger tagger-driver.cc LSTMCausalityTagger.cc
                   BecauseOracleTransitionCorpus.cc BecauseData.cc)
    target_link_libraries(lstm-causality-tagger lstm-parser-core ${Boost_LIBRARIES})
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
