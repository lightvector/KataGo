# KataGo

Research and experimentation with self-play training in Go. Contains a working implementation of AlphaZero-like training with a lot of modifications and enhancements. Due to these enhancements, early training is immensely faster than in other zero-style bots - with a only few strong GPUs for a few days, even a single person should be able to train to mid or even high amateur dan strength on the full 19x19 board on consumer hardware. Also contains a GTP engine and pre-trained neural nets competitive with other top open-source Go engines. KataGo is also capable of estimating score and territory, and due to this right-out-of-the-box plays handicap games somewhat better than many other zero trained bots, without any special hacks. 

The latest KataGo network is a 20-block network that at visit parity is probably slightly stronger than ELF and is comparable to LZ200. See the [releases page](https://github.com/lightvector/KataGo/releases) for the final trained models. The self-play training data from the run and full history of accepted models is available [here](https://d3dndmfyhecmj0.cloudfront.net/).

Many thanks to [Jane Street](https://www.janestreet.com/) for providing the computation power necessary to train KataGo, as well to run numerous many smaller testing runs and experiements. Blog post here about the initial release: [Accelerating Self-Play Learning in Go (Jane Street Tech Blog)](https://blog.janestreet.com/accelerating-self-play-learning-in-go/).

Paper about the major new ideas and techniques used in KataGo: [Accelerating Self-Play Learning in Go (arXiv)](https://arxiv.org/abs/1902.10565).

### Current Status and History

The first serious run of KataGo ran for 7 days in Februrary 2019 on up to 35xV100 GPUs. This is the run featured in the paper. It achieved close to LZ130 strength before it was halted, or up to just barely superhuman.

Following some further improvements and much-improved hyperparameters, KataGo performed a second serious run in May-June a max of 28xV100 GPUs, surpassing the February run after just three and a half days. The run was halted after 19 days, with the final 20-block networks reaching a final strength slightly stronger than LZ-ELFv2! (This is Facebook's very strong 20-block ELF network, running on Leela Zero's search architecture). Comparing to the yet larger Leela Zero 40-block networks, KataGo's network falls somewhere around LZ200 at visit parity, despite only itself being 20 blocks. An updated paper is forthcoming. 

Several promising possibilities for major further improvements are still being researched. In the next few months, I hope to conduct a third run that should surpass both of the previous two. KataGo also in theory supports Japanese rules self-play training, with a fully working implementation of those rules, *including* details like "no points in seki" and "bent-four-in-the-corner is dead even if there are unremovable ko threats". Many short internal test runs did observe neural nets trained with these rules behaving consistent with learning Japanese rules, but it was not fully tested, since Japanese rules training was turned off in later and longer runs to reduce the complexity of moving parts for the paper. I hope to revisit this and other alternative rulesets (ancient group tax rules, non-square boards, button Go) in the future.

See also https://github.com/lightvector/GoNN for some earlier research. KataGo is essentially a continuation of that research, but that old repo has been preserved since the changes in this repo are not backwards compatible, and to leave the old repo intact to continue as a showcase of the many earlier experiments performed there.

### Compatibility with other Tools
KataGo is written in C++ and has a fully working GTP engine. Once compiled, KataGo should be able to work with any GUI program that supports GTP, as well as any analysis program that supports Leela Zero's `lz-analyze` command, such as [Lizzie](https://github.com/featurecat/lizzie).

   * One slight detail with Lizzie - as of June 2019, the tip of Lizzie currently performs a hardcoded check for Leela Zero's version number. To work around this issue, you can pass `-override-version 0.16` or `-override-version 0.17` to make KataGo pretend to be different versions of Leela Zero when Lizzie attempts to query the version number.
   * For developers: KataGo also exposes a GTP command `kata-analyze` that in addition to policy and winrate, also reports an estimate of the *expected score* and a heatmap of the predicted territory ownership of every location of the board. Expected score should be particularly useful for reviewing handicap games or games of weaker players. Whereas the winrate for black will often remain pinned at nearly 100% in a handicap game even as black makes major mistakes (until finally the game becomes very close), expected score should make it more clear which earlier moves are losing points that allow white to catch up, and exactly how much or little those mistakes lose. If you're interested in adding support for this to any analysis tool, feel free to reach out, I'd be happy to answer questions and help.

### OpenCL branch
There is an experimental [branch](https://github.com/lightvector/KataGo/tree/opencl) with an OpenCL implementation, so as to no longer be dependent on CUDA. It should be functional as of now, but the implementations of the GPU kernels are some of the simplest and most un-optimized possible (just to first get it working), making it more than 10x slower than the current CUDA implementation. Additionally, it makes a few assumptions about possible block sizes that may not be entirely general across devices. Work is in-progress.

### Compiling
KataGo is written in C++ and has a fully working GTP engine. Once compiled, you should be able to run it using the trained neural nets for KataGo that you can download from the releases page. See also LICENSE for the software license for this repo. Additionally, if you end up using any of the code in this repo to do any of your own cool new self-play or neural net training experiments, I (lightvector) would to love hear about it.

NOTE: These instructions are for Linux (and possibly MacOS). If you're looking for a Windows version, take a look at https://github.com/lightvector/KataGo/issues/2 for some very generous work by another user in figuring out how to get it working on Windows, with a modified and pre-compiled version for download. You may need to explicitly ungzip the neural net ".txt.gz" file to get KataGo to load the neural net properly, since in that version, zlib is not working for an unknown reason. You may also need to make sure your GPU drivers are up to date. More official windows support is still being worked on and will be coming before too long.

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
   * Selfplay engine (C++ - `cpp/main selfplay`) - continously plays games using the latest neural net in some directory of accepted models, writing the data to some directory.
   * Shuffler (python - `python/shuffle.py`) - scans directories of data from selfplay and shuffles it to produce TFRecord files to write to some directory.
   * Training (python - `python/train.py`) - continuously trains a neural net using TFRecord files from some directory, saving models periodically to some directory.
   * Exporter (python - `python/export_model.py`) - scans a directory of saved models and converts from Tensorflow's format to the format that all the C++ uses, exporting to some directory.
   * Gatekeeper (C++ - `cpp/main gatekeeper`) - polls a directory of newly exported models, plays games against the latest model in an accepted models directory, and if the new model passes, moves it to the accepted models directory.

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


