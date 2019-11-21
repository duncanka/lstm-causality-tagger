# DeepCx
Transition-based shallow semantic parser for causal relations, with state embeddings computed by LSTM RNNs. Based on the [LSTM syntactic parser](https://github.com/clab/lstm-parser/tree/easy-to-use). This system was documented in a [2018 EMNLP paper](http://aclweb.org/anthology/D18-1196).

Note: the instructions below assume that you have a dataset annotated in the same brat format as [BECAUSE](https://github.com/duncanka/BECauSE).

#### Required software

 * A C++ compiler supporting the [C++11 language standard](https://en.wikipedia.org/wiki/C%2B%2B11)
 * [Boost](http://www.boost.org/) libraries
 * [Eigen](http://eigen.tuxfamily.org) (newer versions strongly recommended)
 * [CMake](http://www.cmake.org/)
 * [gcc](https://gcc.gnu.org/) (only tested with gcc version 5.3.0 and above; may be incompatible with earlier versions)
 * [Googletest](https://github.com/google/googletest) library, if you're going to compile in debug mode (which is necessary for unit tests)

#### Build instructions

    git clone --recursive https://github.com/duncanka/lstm-causality-tagger.git # --recursive pulls in lstm-parser submodule
    cd lstm-causality-tagger
    mkdir build
    cd build
    cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen -DCMAKE_BUILD_TYPE=RelWithDebInfo
    make -j2 # or your desired number of compilation threads

#### Running DeepCx

To get DeepCx set up, you will first need to perform steps 1-7 from the [Causeway README](https://github.com/duncanka/Causeway) to set up the Causeway causal language tagger. This is necessary only for running the transition oracle (apologies for the overkill). Skip step 5 if you already have the BECAUSE corpus or equivalent set up.

The instructions below assume you have Causeway set up at `$CAUSEWAY_DIR` and that your working directory is the root `lstm-causality-tagger` directory.

You should also download the pretrained LSTM syntactic parser [model](http://www.cs.cmu.edu/~jdunietz/hosted/english_pos_2_32_100_20_100_12_20.params) to the `lstm-parser` subdirectory.

##### Training DeepCx

To train a causal language tagging model:
1. Run the [transition oracle](../master/scripts/transition_oracle.py) on your training data.
   ```
   export PYTHONPATH=$CAUSEWAY_DIR/NLPypline/src:$CAUSEWAY_DIR/src
   scripts/transition_oracle.py $PATH_TO_TRAINING_DATA
   ```
   This will output `.trans` files alongside the `.txt` and `.ann` files in `$PATH_TO_TRAINING_DATA`.

2. Run the binary in training mode:
   ```
   build/lstm-causality/lstm-causality-tagger --cnn-mem 800 --train --training-data $PATH_TO_TRAINING_DATA
   ```
   This will create a `models` directory within your working directory with the model file inside it (the name will be something like `tagger_10_2_48_48_32_20_32_8_72_0_new-conn__pid5797.params`).

   Training will stop automatically once a given number of epochs has passed without substantial improvement. (You can adjust this behavior with the `--epochs-cutoff`, `--recent-improvements-cutoff`, and `--recent-improvements-epsilon` flags.) You can also stop it with Ctrl+C.

##### Tagging data with DeepCx

DeepCx outputs causal language annotations in the BECAUSE brat format. It *appends to* (not overwrites!) any `.ann` files in the input directory.

To tag new data with a trained model located at `$MODEL_FILEPATH`:
1. Create blank `.ann` files for the transition oracle to read (and clear any existing annotations):
   ```
   scripts/clear-anns.sh $PATH_TO_TEST_DATA
   ```
   The script will ask you to confirm that you do want to clear existing files.

2. Run the [transition oracle](../master/scripts/transition_oracle.py) on your blank `.ann` files to transform the text corpus into the transition format that DeepCx can ingest:
   ```
   export PYTHONPATH=$CAUSEWAY_DIR/NLPypline/src:$CAUSEWAY_DIR/src
   scripts/transition_oracle.py $PATH_TO_TEST_DATA
   ```

3. Run the binary in test mode:
   ```
   build/lstm-causality/lstm-causality-tagger --cnn-mem 800 --test --test-data $PATH_TO_TEST_DATA --model $MODEL_FILEPATH
   ```

##### Evaluating on gold data with DeepCx

To evaluate a trained model located at `$MODEL_FILEPATH`:
1. Run the transition oracle as for training to produce `.trans` files.
2. Run the tagger in evaluation mode:
   ```
   build/lstm-causality/lstm-causality-tagger --cnn-mem 800 --evaluate --test-data $PATH_TO_TEST_DATA --model $MODEL_FILEPATH
   ```
   In this mode, it will not output results to the `.ann` files unless you specifically request it with the `--write-results` option. Note that doing so will append to the annotations from which the gold transitions were generated, which you probably do not want to do!

##### Reproducing the EMNLP results

The EMNLP results were produced with cross-validation, which you can accomplish by adding the `--folds` option:
```
build/lstm-causality/lstm-causality-tagger --cnn-mem 800 --train --training-data $BECAUSE_DIR --folds 20
```
Note that cross-validation will keep overwriting the model for each fold.

#### Pretrained models

TODO

#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact jdunietz@cs.cmu.edu.
