# KataGo

KataGo is an implementation of AlphaZero-like training in Go with a lot of modifications and enhancements that greatly improve upon basic self-play learning. Many of these take advantage of game-specific features and training targets, but also a suprising number of them are non-specific and could easily be tried in other games. Due to these enhancements, early training is immensely faster than in other zero-style bots - with only a few strong GPUs for a few days, even a single person should be able to train a neural net from nothing to high amateur dan strength on the full 19x19 board. KataGo's latest run used about 29 GPUs, rather than thousands (like AlphaZero and ELF) first reached superhuman levels on that hardware in perhaps just three to six days, and reached strength similar to ELF in about 14 days.

Paper about the major new ideas and techniques used in KataGo: [Accelerating Self-Play Learning in Go (arXiv)](https://arxiv.org/abs/1902.10565).
A few further improvements have been found and incorporated into the latest run that are not described in this paper - some post about this might happen eventually.

Many thanks to [Jane Street](https://www.janestreet.com/) for providing the computation power necessary to train KataGo, as well to run numerous many smaller testing runs and experiements. Blog posts about the initial release and some interesting subsequent experiments:
* [Accelerating Self-Play Learning in Go](https://blog.janestreet.com/accelerating-self-play-learning-in-go/)
* [Deep-Learning the Hardest Go Problem in the World](https://blog.janestreet.com/deep-learning-the-hardest-go-problem-in-the-world/).

KataGo's engine also aims to be a useful tool for Go players and developers, and supports the following features:
* Estimates territory and score, rather than only "winrate", helping analyze kyu and amateur dan games besides only on moves that actually would swing the game outcome at pro/superhuman-levels of play.
* Cares about maximizing score, enabling strong play in handicap games when far behind, and reducing slack play in the endgame when winning.
* Supports board sizes ranging from 7x7 to 19x19, and alternative values of komi (including integer values).
* Supports a wide variety of [rules](https://lightvector.github.io/KataGo/rules.html), including rules that match Japanese rules in almost all common cases, and ancient stone-counting-like rules.
* For tool/back-end developers - supports a JSON-based analysis engine that can batch multiple-game evaluations efficiently and be easier to use than GTP.

## Current Status and History

As of mid January 2020, KataGo is on its third major official run! It has reached the strength of the the prior 19 day official run from June 2019 in about 14 days, using fairly similar hardware, and despite mostly *not* yet making a final finetuning step that should be worth a further final strength boost later. This is due to various training improvements which were not present in prior runs. In addition to reaching stronger faster, this third run adds support for Japanese rules, stronger handicap play, and greatly more accurate score estimation.

Now available for download are a new 20 block x 256 channel network that with equal playouts should be slightly stronger than KataGo's prior strongest net from June, which itself was around the strength of ELF and/or Leela Zero's LZ200. Also available for download is a very strong smaller net, including a fast 10 block network that roughly matches the strength of KataGo's 15 block net from June, which might be somewhere around Leela Zero's LZ150.

See the [releases page](https://github.com/lightvector/KataGo/releases) for the latest release and these neural nets. Download more trained neural nets [here](https://d3dndmfyhecmj0.cloudfront.net/g170/index.html).

Here is a graph of the improvement so far as of about 16 days:

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/KataGo/master/images/readme/jan2020vsjune2019.png" height="350"/></td></tr>
<tr><td><sub>X axis is days of training, log scale. (note: hardware is about the same but not entirely identical). Y axis is relative Elo rating based on some 1200-visit test matches. The abrupt jump at the end of the June 2019 run is due to a LR drop that the Jan 2020 run mostly has not done yet, but will do in the future for hopefully a similar boost.</sub></tr></td>
</table>

As of mid January 2020, the run is still ongoing! Since surpassing the strongest prior KataGo nets, KataGo has since increased up to about 36 GPUs (from about 28) and hopes to continue to improve further.

### Older Runs

The first serious run of KataGo ran for 7 days in Februrary 2019 on up to 35xV100 GPUs. This is the run featured the [early versions](https://arxiv.org/abs/1902.10565v2) of KataGo's research paper. It achieved close to LZ130 strength before it was halted, or up to just barely superhuman.

Following some further improvements and much-improved hyperparameters, KataGo performed a second serious run in May-June a max of 28xV100 GPUs, surpassing the February run after just three and a half days. The run was halted after 19 days, with the final 20-block networks reaching a final strength slightly stronger than LZ-ELFv2! (This is Facebook's very strong 20-block ELF network, running on Leela Zero's search architecture). Comparing to the yet larger Leela Zero 40-block networks, KataGo's network falls somewhere around LZ200 at visit parity, despite only itself being 20 blocks. [Recent versions](https://arxiv.org/abs/1902.10565) of the paper have been updated to reflect this run. Here is a graph of Elo ratings of KataGo's June run compared to Leela Zero and ELF based on a set of test games, where the X axis is an approximate measure of self-play computation required (note: log scale).

<table class="image">
<tr><td><img src="https://raw.githubusercontent.com/lightvector/KataGo/master/images/readme/katajunevslz.png" height="350"/></td></tr>
<tr><td><sub>June run of KataGo vs LZ and ELF. X axis is approx selfplay compute spent, log scale. Y axis is relative Elo rating. Leela Zero goes up to LZ225 on this graph. KataGo trains efficiently compared to other bots. See paper for details.</sub></tr></td>
</table>

The self-play training data from the run and full history of accepted models of these two older runs is available [here](https://d3dndmfyhecmj0.cloudfront.net/).

See also https://github.com/lightvector/GoNN for some earlier research. KataGo is essentially a continuation of that research, but that old repo has been preserved since the changes in this repo are not backwards compatible, and to leave the old repo intact to continue as a showcase of the many earlier experiments performed there.

## Where To Download Stuff
You can download precompiled executables for KataGo on the [releases page](https://github.com/lightvector/KataGo/releases) for Windows and Linux.

You can download a few selected neural nets from the [releases page](https://github.com/lightvector/KataGo/releases) or download additional other neural nets from [here](https://d3dndmfyhecmj0.cloudfront.net/g170/index.html)).

**See sections below, particularly "How To Use" and "Tuning for Performance" sections below for how to use KataGo and things you may want to do before actually using KataGo.**

**If using the OpenCL version, KataGo will need to tune itself the first time it starts - try running the `benchmark` command described below.**

## Setting Up and Running KataGo
KataGo implements just a GTP engine - GTP is a simple text protocol that Go software uses to communicate with engines. It does NOT have a graphical interface on its own. So generally, you will want to use KataGo along with a GUI or analysis program, such as [Lizzie](https://github.com/featurecat/lizzie) or [Sabaki](https://sabaki.yichuanshen.de/). Both of these programs also support KataGo's score estimates and visualization as well. NOTE: a version of KataGo is also packaged directly with Lizzie's latest Windows release, but this is a somewhat older version and doesn't support some recent features.

KataGo currently officially supports both Windows and Linux. KataGo should also run fine on OSX, but I don't personally have a good way to compile or test an OSX version - so for this you may need to compile it yourself or check with OSX users who have been able to compile it. Generally, testing across different OS versions and compilers has not been done, so if you encounter problems, feel free to open an issue.

### OpenCL vs CUDA
KataGo has both an OpenCL version and a CUDA version.

  * The CUDA version requires installing [CUDA](https://developer.nvidia.com/cuda-zone) and [CUDNN](https://developer.nvidia.com/cudnn) and a modern NVIDIA GPU.
  * The OpenCL version should be able to run with many other GPUs or accelerators that support [OpenCL](https://en.wikipedia.org/wiki/OpenCL), such AMD GPUs, as well CPU-based OpenCL implementations or things like Intel Integrated Graphics. (Note: Intel integrated graphics though is a toss-up - many versions of Intel's OpenCL seem to be buggy). It also doesn't require the hassle of CUDA and CUDNN and is more likely to work out of the box so long as you do have a decently modern GPU.

Depending on hardware and settings, in practice the OpenCL version seems to range from anywhere to several times slower to moderately faster than the CUDA version. More optimization work may happen in the future though - the OpenCL version has definitely not reached the limit of how well it can be optimized. It also has not been tested for self-play training with extremely large batch sizes to run hundreds of games in parallel, all of KataGo's main training runs so far have been performed with the CUDA implementation.

If your GPU is a top-end NVIDIA GPU and supports FP16 and tensor cores, then the CUDA version is likely to be by far the fastest and strongest, so long you specifically set KataGo to use FP16, and you increase and tune threads and batch size appropriately. See "Tuning for Performance" below.

### How To Use
KataGo supports a few commands. All of these commands require a "model" file that contains the neural net. Most also require a "config" file that specifies parameters for how KataGo behaves. KataGo's precompiled releases should come packaged with example configs (`gtp_example.cfg`). If you care about performance, you will likely want to edit this config for yourself - extensive comments and notes are provided in the config.

Run a benchmark to test performance and help you choose how many threads to use for best performance. You can then edit your GTP config to use this many threads.
**If you are running KataGo for the first time, you probably want to do this before anything else, to test if KataGo works and pick a number of threads. And on the OpenCL version, to give KataGo a chance to autotune itself, which could take a while.**

   * `./katago benchmark -model <NEURALNET>.txt.gz -config <GTP_CONFIG>.cfg`

Run a GTP engine using a downloaded KataGo neural net and example provided config. **This is the command you want to tell your GUI (Lizzie, Sabaki, GoGui, etc) to use to run KataGo** (with the actual paths to your neural net and config files substituted in, of course).

   * `./katago gtp -model <NEURALNET>.txt.gz -config <GTP_CONFIG>.cfg`

Run a JSON-based [analysis engine](docs/Analysis_Engine.md) that can do efficient batched evaluations for a backend Go service:

   * `./katago analysis -model <NEURALNET>.txt.gz -config <ANALYSIS_CONFIG>.cfg -analysis-threads N`

For OpenCL only: run or re-run the tuner to optimize for your particular GPU.

   * `./katago tuner -model <NEURALNET>.txt.gz`


### Tuning for Performance
You will very likely want to tune some of the parameters in `gtp_example.cfg` or `configs/gtp_example.cfg` for your system for good performance, including the number of threads, fp16 usage (CUDA only), NN cache size, pondering settings, and so on. You can also adjust things like KataGo's resign threshold or utility function. All of this should be documented already directly in that [example config file](cpp/configs/gtp_example.cfg).

The OpenCL version will also automatically run a tuner on the first startup to optimize the neural net for your particular device or GPU. This might take a little time, but is completely normal and expected.

To test KataGo's performance with different settings you're trying and also to help choose a number of threads:

`./katago benchmark -model <NEURALNET>.txt.gz -config <GTP_CONFIG>.cfg`

### Features for Developers

#### GTP Extensions:
In addition to a basic set of [GTP commands](https://www.lysator.liu.se/~gunnar/gtp/), KataGo supports a few additional commands, for use with analysis tools and other programs.

KataGo's GTP extensions are documented **[here](docs/GTP_Extensions.md)**.

   * Notably: KataGo exposes a GTP command `kata-analyze` that in addition to policy and winrate, also reports an estimate of the *expected score* and a heatmap of the predicted territory ownership of every location of the board. Expected score should be particularly useful for reviewing handicap games or games of weaker players. Whereas the winrate for black will often remain pinned at nearly 100% in a handicap game even as black makes major mistakes (until finally the game becomes very close), expected score should make it more clear which earlier moves are losing points that allow white to catch up, and exactly how much or little those mistakes lose. If you're interested in adding support for this to any analysis tool, feel free to reach out, I'd be happy to answer questions and help.

   * KataGo also exposes a few GTP extensions that allow setting what rules are in effect (Chinese, AGA, Japanese, etc). See again [here](docs/GTP_Extensions.md) for details.

#### Analysis Engine:
KataGo also implements a separate engine that can evaluate much faster due to batching if you want to analyze whole games at once and might be much less of a hassle than GTP if you are working in an environment where JSON parsing is easy. See [here](docs/Analysis_Engine.md) for details.

## Compiling KataGo
KataGo is written in C++. It should compile on Linux or OSX via g++ that supports at least C++14, or on Windows via MSVC 15 (2017) and later. Other compilers and systems have not been tested yet. This is recommended if you want to run the full KataGo self-play training loop on your own and/or do your own research and experimentation, or if you want to run KataGo on an operating system for which there is no precompiled executable available.

### Linux
   * Requirements
      * CMake with a minimum version of 3.10.2 - for example `sudo apt install cmake` on Debian, or download from https://cmake.org/download/ if that doesn't give you a recent-enough version.
      * Some version of g++ that supports at least C++14.
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. (Of course, CPU implementations may be quite slow).
      * If using the CUDA backend, CUDA 10.1 and CUDNN 7.6.1 (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * zlib, libzip, boost filesystem. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`, `libboost-filesystem-dev`.
      * If you want to do self-play training and research, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DBUILD_MCTS=1 -DUSE_BACKEND=OPENCL` or `cmake . -DBUILD_MCTS=1 -DUSE_BACKEND=CUDA` depending on which backend you want. Specify also `-DUSE_TCMALLOC=1` if using TCMalloc. Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
      * `make`
   * Done! You should now have a compiled `katago` executable in your working directory.
   * Pre-trained neural nets are available on the [releases page](https://github.com/lightvector/KataGo/releases) or more from [here](https://d3dndmfyhecmj0.cloudfront.net/g170/index.html).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device when you run it (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

### Windows
   * Requirements
      * CMake with a minimum version of 3.10.2, GUI version strongly recommended (https://cmake.org/download/)
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
   * Done! You should now have a compiled `katago.exe` executable in your working directory.
   * Note: You may need to copy the ".dll" files corresponding to the various ".lib" files you compiled with into the directory containing katago.exe.
   * Note: If you had to update or install CUDA or GPU drivers, you will likely need to reboot before they will work.
   * Pre-trained neural nets are available on the [releases page](https://github.com/lightvector/KataGo/releases) or more from [here](https://d3dndmfyhecmj0.cloudfront.net/g170/index.html).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

## Selfplay training:
If you'd also like to run the full self-play loop and train your own neural nets you must have [Python3](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/install/) installed. The version of Tensorflow known to work with the current code and with which KataGo's main run was trained is 1.15. Earlier versions than 1.15 will probably not work, and KataGo has NOT been been tested with TF2.0. You'll also probably need a decent amount of GPU power.

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
   * `cpp/katago selfplay -output-dir $BASEDIR/selfplay -models-dir $BASEDIR/models -config-file cpp/configs/SELFPLAYCONFIG.cfg >> log.txt 2>&1 & disown`
     * Some example configs for different numbers of GPUs are: configs/selfplay{1,2,4,8a,8b,8c}.cfg. You may want to edit them depending on your specs - for example to change the sizes of various tables depending on how much memory you have, or to specify gpu indices if you're doing things like putting some mix of training, gatekeeper, and self-play on the same machines or GPUs instead of on separate ones. Note that the number of game threads in these configs is very large, probably far larger than the number of cores on your machine. This is intentional, as each thread only currently runs synchronously with respect to neural net queries, so a large number of parallel games is needed to take advantage of batching.
   * `cd python; ./selfplay/shuffle_and_export_loop.sh $BASEDIR/ $SCRATCH_DIRECTORY $NUM_THREADS $USE_GATING`
     * This starts both the shuffler and exporter. The shuffler will use the scratch directory with the specified number of threads to shuffle in parallel. Make sure you have some disk space. You probably want as many threads as you have cores. If not using the gatekeeper, specify `0` for `$USE_GATING`, else specify `1`.
     * Also, if you're low on disk space, take a look also at the `./selfplay/shuffle.sh` script (which is called by `shuffle_and_export_loop.sh`). Right now it's *very* conservative about cleaning up old shuffles but you could tweak it to be a bit more aggressive.
   * `cd python; ./selfplay/train.sh $BASEDIR/ $TRAININGNAME main b6c96 -lr-scale 1.0 >> log.txt 2>&1 & disown`
     * This starts the training. You may want to look at or edit the train.sh script, it also snapshots the state of the repo for logging, as well as contains some training parameters that can be tweaked.
     * The third argument controls some export behavior:
        * `main` - this is the main net for selfplay, save it regularly to `$BASEDIR/tfsavedmodels_toexport` which the export loop will export regularly for gating.
        * `extra` - save models to `$BASEDIR/tfsavedmodels_toexport_extra`, which the export loop will then export to `$BASEDIR/models_extra`, a directory that does not feed into gating or selfplay.
        * `trainonly` - the neural net without exporting anything. This is useful for when you are trying to jointly train additional models of different sizes and there's no point to have them export anything yet (maybe they're too weak to bother testing).
     * Any additional arguments, like "-lr-scale 1.0" will simply get forwarded on to train.py.
   * `cpp/katago gatekeeper -rejected-models-dir $BASEDIR/rejectedmodels -accepted-models-dir $BASEDIR/models/ -sgf-output-dir $BASEDIR/gatekeepersgf/ -test-models-dir $BASEDIR/modelstobetested/ -config-file cpp/configs/GATEKEEPERCONFIG.cfg >> log.txt 2>&1 & disown`
     * This starts the gatekeeper. Some example configs for different numbers of GPUs are: configs/gatekeeper{1,2a,2b,2c}.cfg. Again, you may want to edit these. The number of simultaneous game threads here is also large for the same reasons as for selfplay. No need to start this if specifying `0` for `$USE_GATING`.

## Contributors

Many thanks to the various people who have contributed to this project! See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.

## License

Except for several external libraries that have been included together in this repo under `cpp/external/` as well as the single file `cpp/core/sha2.cpp`, which all have their own individual licenses, all code and other content in this repo is released for free use, modification, or other purposes under a BSD-style license. See [LICENSE](LICENSE) file for details.

License aside, if you end up using any of the code in this repo to do any of your own cool new self-play or neural net training experiments, I (lightvector) would to love hear about it.
