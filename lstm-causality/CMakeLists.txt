project(lstm-causality-tagger:tagger)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

find_package(Boost COMPONENTS program_options filesystem system)

add_library(causality-tagger-core OBJECT LSTMCausalityTagger.cpp
            BecauseOracleTransitionCorpus.cpp BecauseData.cpp)

ADD_EXECUTABLE(lstm-causality-tagger tagger-driver.cc $<TARGET_OBJECTS:causality-tagger-core>)
target_link_libraries(lstm-causality-tagger lstm-parser-core ${Boost_LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ADD_EXECUTABLE(tester tests/metrics-test.cpp $<TARGET_OBJECTS:causality-tagger-core>)
    target_link_libraries(tester gtest gtest_main pthread lstm-parser-core ${Boost_LIBRARIES})
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
