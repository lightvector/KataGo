# KataGo

Research and experimentation with self-play training in Go. Contains a working implementation of AlphaZero-like training with a lot of modifications and enhancements, as well as a GTP engine that can run using neural nets trained via that process. Due to these enhancements, early training is immensely faster - self-play with a only few strong GPUs for between one day and several days (depending on your GPUs) should already be sufficient to reach somewhere in the range of strong-kyu up to mid-dan strength on the full 19x19 board.

I'll be releasing a paper shortly describing the major techniques used in KataGo. Many thanks to [Jane Street](https://www.janestreet.com/) for providing the computation power necessary to do a real run (as well as numerous many smaller testing runs). As described in the paper, although it was nowhere near as long as a "full" run, it still achieved close to LZ130 strength before it was halted. See the [releases page](https://github.com/lightvector/KataGo/releases) for the final trained models. The self-play training data from the run has not yet been uploaded but should be available soon as well.

I hope to do more experiments at some point in the future and maybe some longer and even better runs testing out many more of the new ideas from LC0 and others that have surfaced in the last few months. KataGo also in theory supports Japanese rules self-play training, *including* details like "no points in seki" and "bent-four-in-the-corner is dead even if there are unremovable ko threats". Many short internal test runs did observe neural nets trained with these rules behaving consistent with learning Japanese rules, but it was not fully tested, since Japanese rules training was turned off in later and longer runs to reduce the complexity of moving parts for the paper. I hope to revisit this in the future.

### Notes and License
This repo is essentially a continuation of https://github.com/lightvector/GoNN but has a lot of changes that in all likelihood break backwards compatibility, so it's being uploaded here as a separate repo. This also leaves the old repo intact to continue as a showcase of the many earlier experiments performed there and described in the readme for its main page.

See LICENSE for the software license for this repo. License aside, informally, if do you successfully use any of the code or any wacky ideas explored in this repo in your own neural nets or to run any of your own experiments, I (lightvector) would to love hear about it and/or might also appreciate a casual acknowledgement where appropriate. Yay.

### Compiling the GTP engine and/or most of the self-play framework:
There is an implementation of MCTS in this repo along with a GTP engine and an simple SGF command-line analysis tool backed by the MCTS engine. Along with most of the self-play framework, all of this is written in C++. Once compiled, you should be able to run it using the trained neural nets for KataGo that you can download from the releases page.

   * Requirements
      * CMake with a minimum version of 3.12.3 (https://cmake.org/download/)
      * CUDA 10.0 and CUDNN 7.4.1 (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * zlib (zlib1g-dev), libzip (libzip-dev), boost filesystem (libboost-filesystem-dev).
      * If you want to do self-play, probably Google perftools (google-perftools, libgoogle-perftools-dev) for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc, but better mallocs handle it fine.
      * Some version of g++ that supports at least C++14.
   * Compile using CMake and make in the cpp directory:
      * `cd cpp`
      * `cmake . -DBUILD_MCTS=1 -DUSE_CUDA_BACKEND=1` OR if you're using TCMalloc then `cmake . -DBUILD_MCTS=1 -DUSE_CUDA_BACKEND=1 -DUSE_TCMALLOC=1`
      * `make`
   * You can now run the compiled `main` executable to do various things. Edit the configs to change parameters as desired.
      * Example: `./main gtp -model <NEURALNET>.txt.gz -config configs/gtp_example.cfg` - Run a simple GTP engine using a given neural net and example provided config.
      * Example: `./main evalsgf <SGF>.sgf -model <NEURALNET>.txt.gz -move-num <MOVENUM> -config configs/eval_sgf.cfg` - Have the bot analyze the specified move of the specified SGF.
   * Pre-trained neural nets are available on the [releases page](https://github.com/lightvector/KataGo/releases).

### Selfplay training:
If you'd also like to run the full self-play loop and train your own neural nets you must have [Python3](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/install/) installed. The version of Tensorflow known to work with the current code and with which KataGo's main run was trained is 1.12.0. Possibly later or earlier versions could work too, but they have not been tested. You'll also probably need a decent amount of GPU power.

There are 5 things that need to all run concurrently to form a closed self-play training loop.
   * Selfplay engine (C++ - `cpp/main selfplay`) - continously plays games using the latest neural net in some directory, writing the data to somewhere.
   * Shuffler (python - `python/shuffle.py`) - scans a directories of data from selfplay and shuffles it to produce TFRecord files to write to some directory.
   * Training (python - `python/train.py`) - continuously trains a neural net using TFRecord files from some directory, saving models periodically to some directory.
   * Exporter (python - `python/export_model.py`) - scans a directory of saved models and converts from Tensorflow's format to the format that all the C++ uses, saving to some directory.
   * Gatekeeper (C++ - `cpp/main gatekeeper`) - polls a directory of exported models from some candidate models directory, plays games against the latest model in an accepted models directory, and if the new model passes, moves it to the accepted models directory.

On the cloud, a reasonable small-scale setup for all these things might be:
   * A machine with a decent amount of cores and memory to run the shuffler and exporter.
   * A machine with one or two powerful GPUs and a lot of cpus and memory to run the selfplay engine.
   * A machine with a medium GPU and a lot of cpus and memory to run the gatekeeper.
   * A machine with a modest GPU to run the training.
   * A well-performing shared filesystem accessible by all four of these machines.

You may need to play with learning rates, batch sizes, and the balance between training and self-play. If the training GPU is too strong, you may overfit more since it will be on the same data over and over because self-play won't be generating new data fast enough, and it's possible you will want to adjust hyperparmeters or even add an artifical delay for each loop of training. Overshooting the other way and having too much GPU power on self-play is harder since generally you need at least an order of magnitude more power on self-play than training. If you do though maybe you'll start seeing diminishing returns as the training becomes the limiting factor in improvement.

Example instructions to start up these things (assuming you have appropriate machines set up), with some base directory $BASEDIR to hold the all the models and training data generated with a few hundred GB of disk space. The below commands assume you're running from the root of the repo and that you can run bash scripts.
   * `cpp/main selfplay -output-dir $BASEDIR/selfplay -models-dir $BASEDIR/models -config-file cpp/configs/SELFPLAYCONFIG.cfg -inputs-version 4 >> log.txt 2>&1 & disown`
     * Some example configs for different numbers of GPUs are: configs/selfplay{1,2,4,8a,8b,8c}.cfg. You may want to edit them depending on your specs - for example to change the sizes of various tables depending on how much memory you have, or to specify gpu indices if you're doing things like putting some mix of training, gatekeeper, and self-play on the same machines or GPUs instead of on separate ones. Note that the number of game threads in these configs is very large, probably far larger than the number of cores on your machine. This is intentional, as each thread only currently runs synchronously with respect to neural net queries, so a large number of parallel games is needed to take advantage of batching.
     * Inputs version 4 is the version of input features KataGo currently uses, to be written down for training.
   * `cd python; ./selfplay/shuffle_and_export_loop.sh $BASEDIR/ $SCRATCH_DIRECTORY $NUM_THREADS`
     * This starts both the shuffler and exporter. The shuffler will use the scratch directory with the specified number of threads to shuffle in parallel. Make sure you have some disk space. You probably want as many threads as you have cores.
     * Also, if you're low on disk space, take a look also at the `./selfplay/shuffle.sh` script (which is called by `shuffle_and_export_loop.sh`). Right now it's *very* conservative about cleaning up old shuffles but you could tweak it to be a bit more aggressive.
   * `cd python; ./selfplay/train.sh $BASEDIR/ $TRAININGNAME b6c96 >> log.txt 2>&1 & disown`
     * This starts the training. You may want to look at or edit the train.sh script, it also snapshots the state of the repo for logging, as well as contains some training parameters that can be tweaked.
   * `cpp/main gatekeeper -rejected-models-dir $BASEDIR/rejectedmodels -accepted-models-dir $BASEDIR/models/ -sgf-output-dir $BASEDIR/gatekeepersgf/ -test-models-dir $BASEDIR/modelstobetested/ -config-file cpp/configs/GATEKEEPERCONFIG.cfg >> log.txt 2>&1 & disown`
     * This starts the gatekeeper. Some example configs for different numbers of GPUs are: configs/gatekeeper{1,2a,2b,2c}.cfg. Again, you may want to edit these. The number of simultaneous game threads here is also large for the same reasons as for selfplay.


