# KataGo

Research and experimentation with self-play training in Go. Contains a working implementation of AlphaZero-like training with a lot of modifications and enhancements. Due to these enhancements, early training is immensely faster than in other zero-style bots - with a only few strong GPUs for a few days, even a single person should be able to train to mid or even high amateur dan strength on the full 19x19 board on consumer hardware. Also contains a GTP engine and pre-trained neural nets competitive with other top open-source Go engines. KataGo is also capable of estimating score and territory, and due to this right-out-of-the-box plays handicap games somewhat better than many other zero trained bots, without any special hacks.

The latest KataGo network is a 20-block network that at visit parity is probably slightly stronger than ELF and is comparable to LZ200. See the [releases page](https://github.com/lightvector/KataGo/releases) for the final trained models. The self-play training data from the run and full history of accepted models is available [here](https://d3dndmfyhecmj0.cloudfront.net/).

Many thanks to [Jane Street](https://www.janestreet.com/) for providing the computation power necessary to train KataGo, as well to run numerous many smaller testing runs and experiements. Blog post here about the initial release: [Accelerating Self-Play Learning in Go (Jane Street Tech Blog)](https://blog.janestreet.com/accelerating-self-play-learning-in-go/).

Paper about the major new ideas and techniques used in KataGo: [Accelerating Self-Play Learning in Go (arXiv)](https://arxiv.org/abs/1902.10565).

## Current Status and History
The first serious run of KataGo ran for 7 days in Februrary 2019 on up to 35xV100 GPUs. This is the run featured in the paper. It achieved close to LZ130 strength before it was halted, or up to just barely superhuman.

Following some further improvements and much-improved hyperparameters, KataGo performed a second serious run in May-June a max of 28xV100 GPUs, surpassing the February run after just three and a half days. The run was halted after 19 days, with the final 20-block networks reaching a final strength slightly stronger than LZ-ELFv2! (This is Facebook's very strong 20-block ELF network, running on Leela Zero's search architecture). Comparing to the yet larger Leela Zero 40-block networks, KataGo's network falls somewhere around LZ200 at visit parity, despite only itself being 20 blocks. An updated paper is forthcoming. Training data and models are [here](https://d3dndmfyhecmj0.cloudfront.net/).

Several promising possibilities for major further improvements are still being researched. In the next few months, I (lightvector) hope to conduct a third run that should surpass both of the previous two. KataGo also in theory supports Japanese rules self-play training, with a fully working implementation of those rules, *including* details like "no points in seki" and "bent-four-in-the-corner is dead even if there are unremovable ko threats". Many short internal test runs did observe neural nets trained with these rules behaving consistent with learning Japanese rules, but it was not fully tested, since Japanese rules training was turned off in later and longer runs to reduce the complexity of moving parts for the paper. I hope to revisit this and other alternative rulesets (ancient group tax rules, non-square boards, button Go) in the future.

See also https://github.com/lightvector/GoNN for some earlier research. KataGo is essentially a continuation of that research, but that old repo has been preserved since the changes in this repo are not backwards compatible, and to leave the old repo intact to continue as a showcase of the many earlier experiments performed there.

## Running KataGo
KataGo is written in C++ and has a fully working GTP engine. Once compiled, KataGo should be able to work with any GUI program that supports GTP, as well as any analysis program that supports Leela Zero's `lz-analyze` command, such as [Lizzie](https://github.com/featurecat/lizzie) or [Sabaki](https://sabaki.yichuanshen.de/). KataGo is also included in the (Windows) release of Lizzie, which supports KataGo score estimate visualizations.

Trained models AND precompiled executables are available on the [releases page](https://github.com/lightvector/KataGo/releases)!

### OpenCL vs CUDA, Supported Operating Systems
KataGo has both an OpenCL backend and a CUDA backend. The OpenCL backend may be easier to compile and get working, as it doesn't require downloading and installing CUDA and CUDNN. It should be usable on systems with non-NVIDIA GPUs, Intel integrated graphics, etc., as long as they have a working OpenCL installation.

   * KataGo should compile on Linux or OSX via g++ that supports at least C++14.
   * KataGo should compile on Windows via MSVC 15 (2017) and later.
   * Other compilers and systems have not been tested yet.

Depending on hardware and settings, in practice the OpenCL version seems to range from anywhere to several times slower to about similarly fast as the CUDA version. More optimization work may happen in the future though - the OpenCL version has definitely not reached the limit of how well it can be optimized. It also has not been tested for self-play training with extremely large batch sizes to run hundreds of games in parallel, all of KataGo's main training runs so far have been performed with the CUDA implementation.

Extensive testing across different OSs and versions and compilers has not been done, so if you encounter issues, feel free to open an issue, and if compiling is not working for some system or compiler but would only require relatively minor changes to make it work, I'd be happy to add support.

### Tuning for Performance
Regardless of whether compiled yourself or using a precompiled version, you will very likely want to tune some of the parameters in `gtp_example.cfg` or `configs/gtp_example.cfg` for your system for good performance, including the number of threads, fp16 usage (CUDA only), NN cache size, pondering settings, and so on. You can also adjust things like KataGo's resign threshold or utility function. Most of the relevant parameters should be be reasonably well documented directly inline in that [example config file](cpp/configs/gtp_example.cfg).

The OpenCL version will also automatically run a tuner on the first startup to optimize the neural net for your particular device or GPU. This might take a little time, but is completely normal and expected.

### GTP Extensions (for developers):
In addition to a basic set of [GTP commands](https://www.lysator.liu.se/~gunnar/gtp/), KataGo supports a few additional commands, for use with analysis tools and other programs.
KataGo's GTP extensions are documented [here](GTP_Extensions.md).

   * Notably: KataGo exposes a GTP command `kata-analyze` that in addition to policy and winrate, also reports an estimate of the *expected score* and a heatmap of the predicted territory ownership of every location of the board. Expected score should be particularly useful for reviewing handicap games or games of weaker players. Whereas the winrate for black will often remain pinned at nearly 100% in a handicap game even as black makes major mistakes (until finally the game becomes very close), expected score should make it more clear which earlier moves are losing points that allow white to catch up, and exactly how much or little those mistakes lose. If you're interested in adding support for this to any analysis tool, feel free to reach out, I'd be happy to answer questions and help.


## Compiling
KataGo is written in C++ and has a fully working GTP engine. Once compiled, you should be able to run it using the trained neural nets for KataGo that you can download from the releases page. See also LICENSE for the software license for this repo. Additionally, if you end up using any of the code in this repo to do any of your own cool new self-play or neural net training experiments, I (lightvector) would to love hear about it.

### Linux
   * Requirements
      * CMake with a minimum version of 3.12.3 (https://cmake.org/download/)
      * Some version of g++ that supports at least C++14.
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. (Of course, CPU implementations may be quite slow).
      * If using the CUDA backend, CUDA 10.1 and CUDNN 7.6.1 (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * zlib, libzip, boost filesystem. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`, `libboost-filesystem-dev`.
      * If you want to do self-play, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc, but better mallocs handle it fine.
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DBUILD_MCTS=1 -DUSE_BACKEND=OPENCL` or `cmake . -DBUILD_MCTS=1 -DUSE_BACKEND=CUDA` depending on which backend you want. Specify also `-DUSE_TCMALLOC=1` if using TCMalloc. Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
      * `make`
   * You can now run the compiled `katago` executable to do various things. You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
      * Example: `./katago gtp -model <NEURALNET>.txt.gz -config configs/gtp_example.cfg` - Run a simple GTP engine using a given neural net and example provided config.
      * Example: `./katago tuner -model <NEURALNET>.txt.gz` - (For OpenCL) run or re-run the tuner to optimize for your particular GPU.
      * Example: `./katago evalsgf <SGF>.sgf -model <NEURALNET>.txt.gz -move-num <MOVENUM> -config configs/evalsgf_example.cfg` - Have the bot analyze the specified move of the specified SGF.
   * Pre-trained neural nets are available on the [releases page](https://github.com/lightvector/KataGo/releases).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

### Windows
   * Requirements
      * CMake with a minimum version of 3.12.3, GUI version strongly recommended (https://cmake.org/download/)
      * Microsoft Visual Studio for C++. Version 15 (2017) has been tested and should work, other versions might work as well.
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. (Of course, CPU implementations may be quite slow).
      * If using the CUDA backend, CUDA 10.1 and CUDNN 7.6.1 (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * Boost. You can obtain prebuilt libraries for Windows at: https://www.boost.org/users/download/ -> "Prebuilt windows binaries" -> "1.70.0". For example, boost_1_70_0-msvc-14.1-64.exe if you're on 64-bit windows. Note that MSVC 14.1 libraries (2015) are directly-compatible with MSVC 15 (2017).
      * zlib. The following package might work, https://www.nuget.org/packages/zlib-vc140-static-64/, or alternatively you can build it yourself via something like: https://github.com/kiyolee/zlib-win-build
      * libzip (optional, needed only for self-play training) - for example https://github.com/kiyolee/libzip-win-build
   * Download/clone this repo to some folder `KataGo`.
   * Configure using CMake GUI and compile in MSVC:
      * Select `KataGo/cpp` as the source code directory in [CMake GUI](https://cmake.org/runningcmake/).
      * Set the build directory to wherever you would like the built executable to be produced.
      * Click "Configure". For the generator select your MSVC version, and also select "x64" for the platform if you're on 64-bit windows.
      * If you get errors where CMake has not automatically found Boost, ZLib, etc, point it to the appropriate places according to the error messages (by setting `BOOST_ROOT`, `ZLIB_INCLUDE_DIR`, `ZLIB_LIBRARY`, etc). Note that "*_LIBRARY" expects to be pointed to the ".lib" file, whereas the ".dll" file is what you actually need to run.
      * Also set `USE_BACKEND` to `OPENCL` or `CUDA`, and adjust options like `NO_GIT_REVISION` if needed, and run "Configure" again as needed.
      * Once running "Configure" looks good, run "Generate" and then open MSVC and build as normal in MSVC.
   * You can now run the compiled `katago.exe` executable to do various things.
      * Note: You may need to copy the ".dll" files corresponding to the various ".lib" files you compiled with into the directory containing katago.exe.
      * Note: If you had to update or install CUDA or GPU drivers, you will likely need to reboot before they will work.
      * Example: `katago.exe gtp -model <NEURALNET>.txt.gz -config configs/gtp_example.cfg` - Run a simple GTP engine using a given neural net and example provided config.
      * Example: `katago.exe tuner -model <NEURALNET>.txt.gz` - (For OpenCL) run or re-run the tuner to optimize for your particular GPU.
      * Example: `katago.exe evalsgf <SGF>.sgf -model <NEURALNET>.txt.gz -move-num <MOVENUM> -config configs/evalsgf_example.cfg` - Have the bot analyze the specified move of the specified SGF.
   * Pre-trained neural nets are available on the [releases page](https://github.com/lightvector/KataGo/releases).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

## Selfplay training:
If you'd also like to run the full self-play loop and train your own neural nets you must have [Python3](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/install/) installed. The version of Tensorflow known to work with the current code and with which KataGo's main run was trained is 1.12.0. Possibly later or earlier versions could work too, but they have not been tested. You'll also probably need a decent amount of GPU power.

There are 5 things that need to all run concurrently to form a closed self-play training loop.
   * Selfplay engine (C++ - `cpp/katago selfplay`) - continously plays games using the latest neural net in some directory of accepted models, writing the data to some directory.
   * Shuffler (python - `python/shuffle.py`) - scans directories of data from selfplay and shuffles it to produce TFRecord files to write to some directory.
   * Training (python - `python/train.py`) - continuously trains a neural net using TFRecord files from some directory, saving models periodically to some directory.
   * Exporter (python - `python/export_model.py`) - scans a directory of saved models and converts from Tensorflow's format to the format that all the C++ uses, exporting to some directory.
   * Gatekeeper (C++ - `cpp/katago gatekeeper`) - polls a directory of newly exported models, plays games against the latest model in an accepted models directory, and if the new model passes, moves it to the accepted models directory. OPTIONAL, it is also possible to train just accepting every new model.

On the cloud, a reasonable small-scale setup for all these things might be:
   * A machine with a decent amount of cores and memory to run the shuffler and exporter.
   * A machine with one or two powerful GPUs and a lot of cpus and memory to run the selfplay engine.
   * A machine with a medium GPU and a lot of cpus and memory to run the gatekeeper.
   * A machine with a modest GPU to run the training.
   * A well-performing shared filesystem accessible by all four of these machines.

You may need to play with learning rates, batch sizes, and the balance between training and self-play. If the training GPU is too strong, you may overfit more since it will be on the same data over and over because self-play won't be generating new data fast enough, and it's possible you will want to adjust hyperparmeters or even add an artifical delay for each loop of training. Overshooting the other way and having too much GPU power on self-play is harder since generally you need at least an order of magnitude more power on self-play than training. If you do though maybe you'll start seeing diminishing returns as the training becomes the limiting factor in improvement.

Example instructions to start up these things (assuming you have appropriate machines set up), with some base directory $BASEDIR to hold the all the models and training data generated with a few hundred GB of disk space. The below commands assume you're running from the root of the repo and that you can run bash scripts.
   * `cpp/katago selfplay -output-dir $BASEDIR/selfplay -models-dir $BASEDIR/models -config-file cpp/configs/SELFPLAYCONFIG.cfg -inputs-version 4 >> log.txt 2>&1 & disown`
     * Some example configs for different numbers of GPUs are: configs/selfplay{1,2,4,8a,8b,8c}.cfg. You may want to edit them depending on your specs - for example to change the sizes of various tables depending on how much memory you have, or to specify gpu indices if you're doing things like putting some mix of training, gatekeeper, and self-play on the same machines or GPUs instead of on separate ones. Note that the number of game threads in these configs is very large, probably far larger than the number of cores on your machine. This is intentional, as each thread only currently runs synchronously with respect to neural net queries, so a large number of parallel games is needed to take advantage of batching.
     * Inputs version 4 is the version of input features KataGo currently uses, to be written down for training.
   * `cd python; ./selfplay/shuffle_and_export_loop.sh $BASEDIR/ $SCRATCH_DIRECTORY $NUM_THREADS $USE_GATING`
     * This starts both the shuffler and exporter. The shuffler will use the scratch directory with the specified number of threads to shuffle in parallel. Make sure you have some disk space. You probably want as many threads as you have cores. If not using the gatekeeper, specify `0` for `$USE_GATING`, else specify `1`.
     * Also, if you're low on disk space, take a look also at the `./selfplay/shuffle.sh` script (which is called by `shuffle_and_export_loop.sh`). Right now it's *very* conservative about cleaning up old shuffles but you could tweak it to be a bit more aggressive.
   * `cd python; ./selfplay/train.sh $BASEDIR/ $TRAININGNAME b6c96 >> log.txt 2>&1 & disown`
     * This starts the training. You may want to look at or edit the train.sh script, it also snapshots the state of the repo for logging, as well as contains some training parameters that can be tweaked.
   * `cpp/katago gatekeeper -rejected-models-dir $BASEDIR/rejectedmodels -accepted-models-dir $BASEDIR/models/ -sgf-output-dir $BASEDIR/gatekeepersgf/ -test-models-dir $BASEDIR/modelstobetested/ -config-file cpp/configs/GATEKEEPERCONFIG.cfg >> log.txt 2>&1 & disown`
     * This starts the gatekeeper. Some example configs for different numbers of GPUs are: configs/gatekeeper{1,2a,2b,2c}.cfg. Again, you may want to edit these. The number of simultaneous game threads here is also large for the same reasons as for selfplay. No need to start this if specifying `0` for `$USE_GATING`.

## Contributors

Many thanks to the various people who have contributed to this project! See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.

## License

Except for several external libraries that have been included together in this repo under `cpp/external/` as well as the single file `cpp/core/sha2.cpp`, which all have their own individual licenses, all code and other content in this repo is released for free use, modification, or other purposes under a BSD-style license. See [LICENSE](LICENSE) file for details.
