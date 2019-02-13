# DeepCx
Transition-based shallow semantic parser for causal relations, with state embeddings computed by LSTM RNNs. Based on the [LSTM syntactic parser](https://github.com/clab/lstm-parser/tree/easy-to-use). This system was documented in a [2018 EMNLP paper](http://aclweb.org/anthology/D18-1196).

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * [gcc](https://gcc.gnu.org/gcc-5/) (only tested with gcc version 5.3.0, may be incompatible with earlier versions)
 * [Googletest](https://github.com/google/googletest) library, if you're going to compile in debug mode (which is necessary for unit tests)

#### Build instructions

    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
    make -j2

#### Train a parsing model

TODO

#### Parse data with your parsing model

TODO

#### Pretrained models

TODO

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact jdunietz@cs.cmu.edu.
