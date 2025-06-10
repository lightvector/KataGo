# KataGo

* [Overview](#overview)
* [Training History and Research](#training-history-and-research)
* [Where To Download Stuff](#where-to-download-stuff)
* [Setting Up and Running KataGo](#setting-up-and-running-katago)
  * [GUIs](#guis)
  * [Windows and Linux](#windows-and-linux)
  * [MacOS](#macos)
  * [OpenCL vs CUDA vs TensorRT vs Eigen](#opencl-vs-cuda-vs-tensorrt-vs-eigen)
  * [How To Use](#how-to-use)
  * [Tuning for Performance](#tuning-for-performance)
  * [Common Questions and Issues](#common-questions-and-issues)
    * [Issues with specific GPUs or GPU drivers](#issues-with-specific-gpus-or-gpu-drivers)
    * [Common Problems](#common-problems)
    * [Other Questions](#other-questions)
* [Features for Developers](#features-for-developers)
  * [GTP Extensions](#gtp-extensions)
  * [Analysis Engine](#analysis-engine)
* [Compiling KataGo](#compiling-katago)
* [Source Code Overview](#source-code-overview)
* [Selfplay Training](#selfplay-training)
* [Contributors](#contributors)
* [License](#license)

## Overview

KataGo's public distributed training run is ongoing! See https://katagotraining.org/ for more details, to download the latest and strongest neural nets, or to learn how to contribute if you want to help KataGo improve further! Also check out the computer Go [discord channel](https://discord.gg/bqkZAz3)!

As of 2025, KataGo remains one of the strongest open source Go bots available online. KataGo was trained using an AlphaZero-like process with many enhancements and improvements, and is capable of reaching top levels rapidly and entirely from scratch with no outside data, improving only via self-play. Some of these improvements take advantage of game-specific features and training targets, but also many of the techniques are general and could be applied in other games. As a result, early training is immensely faster than in other self-play-trained bots - with only a few strong GPUs for a few days, any researcher/enthusiast should be able to train a neural net from nothing to high amateur dan strength on the full 19x19 board. If tuned well, a training run using only a *single* top-end consumer GPU could possibly train a bot from scratch to superhuman strength within a few months.

Experimentally, KataGo did also try some limited ways of using external data at the end of its June 2020 run, and has continued to do so into its most recent public distributed run, "kata1" at https://katagotraining.org/. External data is not necessary for reaching top levels of play, but still appears to provide some mild benefits against some opponents, and noticeable benefits in a useful analysis tool for a variety of kinds of situations that don't occur in self-play but that do occur in human games and  games that users wish to analyze.

KataGo's engine aims to be a useful tool for Go players and developers, and supports the following features:
* Estimates territory and score, rather than only "winrate", helping analyze kyu and amateur dan games besides only on moves that actually would swing the game outcome at pro/superhuman-levels of play.
* Cares about maximizing score, enabling strong play in handicap games when far behind, and reducing slack play in the endgame when winning.
* Supports alternative values of komi (including integer values) and good high-handicap game play.
* Supports board sizes ranging from 7x7 to 19x19, and as of May 2020 may be the strongest open-source bot on both 9x9 and 13x13 as well.
* Supports a wide variety of [rules](https://lightvector.github.io/KataGo/rules.html), including rules that match Japanese rules in almost all common cases, and ancient stone-counting-like rules.
* For tool/back-end developers - supports a JSON-based analysis engine that can batch multiple-game evaluations efficiently and be easier to use than GTP.

## Training History and Research and Docs

Here are some links to some docs/papers/posts about KataGo's research and training!

* Paper about the major new ideas and techniques used in KataGo: [Accelerating Self-Play Learning in Go (arXiv)](https://arxiv.org/abs/1902.10565). Many of the specific parameters are outdated, but the general methods continue to be used.

* Many major further improvements have been found since then, which have been incorporated into KataGo's more recent runs and are documented here: [KataGoMethods.md](docs/KataGoMethods.md).

* KataGo has a fully working implementation of Monte-Carlo Graph Search, extending MCTS to operate on graphs instead of just trees! An explanation can be found here [Monte-Carlo Graph Search from First Principles](docs/GraphSearch.md). This explanation is written to be general (not specific to KataGo) and to fill a big gap in explanatory material in the academic literature and hopefully it can be useful to others!

* Many thanks to [Jane Street](https://www.janestreet.com/) for supporting the training of KataGo's major earlier published runs, as well as numerous many smaller testing runs and experiments. Blog posts about the initial release and some interesting subsequent experiments:
    * [Accelerating Self-Play Learning in Go](https://blog.janestreet.com/accelerating-self-play-learning-in-go/)
    * [Deep-Learning the Hardest Go Problem in the World](https://blog.janestreet.com/deep-learning-the-hardest-go-problem-in-the-world/).

For more details about KataGo's older training runs, including comparisons to other bots, see [Older Training History and Research](TrainingHistory.md)!

Also if you're looking to ask about general information about KataGo or how it works, or about some past Go bots besides KataGo, consider the computer Go [discord channel](https://discord.gg/bqkZAz3).

## Where To Download Stuff
Precompiled executables for KataGo can be found at the [releases page](https://github.com/lightvector/KataGo/releases) for Windows and Linux.

And the latest neural nets are available at [https://katagotraining.org/](https://katagotraining.org/).

## Setting Up and Running KataGo
KataGo implements just a GTP engine, which is a simple text protocol that Go software uses. It does NOT have a graphical interface on its own. So generally, you will want to use KataGo along with a GUI or analysis program. A few of them bundle KataGo in their download so that you can get everything from one place rather than downloading separately and managing the file paths and commands.

### GUIs
This is by no means a complete list - there are lots of things out there. But, writing as of 2020, a few of the easier and/or popular ones might be:

* [KaTrain](https://github.com/sanderland/katrain) - KaTrain might be the easiest to set up for non-technical users, offering an all-in-one package (no need to download KataGo separately!), modified-strength bots for weaker players, and good analysis features.
* [Lizzie](https://github.com/featurecat/lizzie) - Lizzie is very popular for running long interactive analyses and visualizing them as they happen. Lizzie also offers an all-in-one package. However keep mind that KataGo's OpenCL version may take quite a while to tune and load on the very first startup as described [here](#opencl-vs-cuda), and Lizzie does a poor job of displaying this progress as it happens. And in case of an actual error or failure, Lizzie's interface is not the best at explaining these errors and will appear to hang forever. The version of KataGo packaged with Lizzie is quite strong but might not always be the newest or strongest, so once you have it working, you may want to download KataGo and a newer network from [releases page](https://github.com/lightvector/KataGo/releases) and replace Lizzie's versions with them.
* [Ogatak](https://github.com/rooklift/ogatak) is a KataGo-specific GUI with an emphasis on displaying the basics in a snappy, responsive fashion. It does not come with KataGo included.
* [q5Go](https://github.com/bernds/q5Go) and [Sabaki](https://sabaki.yichuanshen.de/) are general SGF editors and GUIs that support KataGo, including KataGo's score estimation, and many high-quality features.

Generally, for GUIs that don't offer an all-in-one package, you will need to download KataGo (or any other Go engine of your choice!) and tell the GUI the proper command line to run to invoke your engine, with the proper file paths involved. See [How To Use](#how-to-use) below for details on KataGo's command line interface.

### Windows and Linux

KataGo currently officially supports both Windows and Linux, with [precompiled executables provided each release](https://github.com/lightvector/KataGo/releases). On Windows, the executables should generally work out of the box, on Linux if you encounter issues with system library versions, as an alternative [building from source](Compiling.md) is usually straightforward. Not all different OS versions and compilers have been tested, so if you encounter problems, feel free to open an issue. KataGo can also of course be compiled from source on Windows via MSVC on Windows or on Linux via usual compilers like g++, documented further down.

### MacOS
The community also provides KataGo packages for [Homebrew](https://brew.sh) on MacOS - releases there may lag behind official releases slightly.

Use `brew install katago`. The latest config files and networks are installed in KataGo's `share` directory. Find them via `brew list --verbose katago`. A basic way to run katago will be `katago gtp -config $(brew list --verbose katago | grep 'gtp.*\.cfg') -model $(brew list --verbose katago | grep .gz | head -1)`. You should choose the Network according to the release notes here and customize the provided example config as with every other way of installing KataGo.

### OpenCL vs CUDA vs TensorRT vs Eigen
KataGo has four backends, OpenCL (GPU), CUDA (GPU), TensorRT (GPU), and Eigen (CPU).

The quick summary is:
  * **To easily get something working, try OpenCL if you have any good or decent GPU.**
  * **For often much better performance on NVIDIA GPUs, try TensorRT**, but you may need to install TensorRT from Nvidia.
  * Use Eigen with AVX2 if you don't have a GPU or if your GPU is too old/weak to work with OpenCL, and you just want a plain CPU KataGo.
  * Use Eigen without AVX2 if your CPU is old or on a low-end device that doesn't support AVX2.
  * The CUDA backend can work for NVIDIA GPUs with CUDA+CUDNN installed but is likely worse than TensorRT.

More in detail:
  * OpenCL is a general GPU backend should be able to run with any GPUs or accelerators that support [OpenCL](https://en.wikipedia.org/wiki/OpenCL), including NVIDIA GPUs, AMD GPUs, as well CPU-based OpenCL implementations or things like Intel Integrated Graphics. This is the most general GPU version of KataGo and doesn't require a complicated install like CUDA does, so is most likely to work out of the box as long as you have a fairly modern GPU. **However, it also need to take some time when run for the very first time to tune itself.** For many systems, this will take 5-30 seconds, but on a few older/slower systems, may take many minutes or longer. Also, the quality of OpenCL implementations is sometimes inconsistent, particularly for Intel Integrated Graphics and for AMD GPUs that are older than several years, so it might not work for very old machines, as well as specific buggy newer AMD GPUs, see also [Issues with specific GPUs or GPU drivers](#issues-with-specific-gpus-or-gpu-drivers).
  * CUDA is a GPU backend specific to NVIDIA GPUs (it will not work with AMD or Intel or any other GPUs) and requires installing [CUDA](https://developer.nvidia.com/cuda-zone) and [CUDNN](https://developer.nvidia.com/cudnn) and a modern NVIDIA GPU. On most GPUs, the OpenCL implementation will actually beat NVIDIA's own CUDA/CUDNN at performance. The exception is for top-end NVIDIA GPUs that support FP16 and tensor cores, in which case sometimes one is better and sometimes the other is better.
  * TensorRT is similar to CUDA, but only uses NVIDIA's TensorRT framework to run the neural network with more optimized kernels. For modern NVIDIA GPUs, it should work whenever CUDA does and will usually be faster than CUDA or any other backend.
  * Eigen is a *CPU* backend that should work widely *without* needing a GPU or fancy drivers. Use this if you don't have a good GPU or really any GPU at all. It will be quite significantly slower than OpenCL or CUDA, but on a good CPU can still often get 10 to 20 playouts per second if using the smaller (15 or 20) block neural nets. Eigen can also be compiled with AVX2 and FMA support, which can provide a big performance boost for Intel and AMD CPUs from the last few years. However, it will not run at all on older CPUs (and possibly even some recent but low-power modern CPUs) that don't support these fancy vector instructions.

For **any** implementation, it's recommended that you also tune the number of threads used if you care about optimal performance, as it can make a factor of 2-3 difference in the speed. See "Tuning for Performance" below. However, if you mostly just want to get it working, then the default untuned settings should also be still reasonable.

### How To Use
KataGo is just an engine and does not have its own graphical interface. So generally you will want to use KataGo along with a [GUI or analysis program](#guis).
If you encounter any problems while setting this up, check out [Common Questions and Issues](#common-questions-and-issues).

**First**: Run a command like this to make sure KataGo is working, with the neural net file you [downloaded](https://github.com/lightvector/KataGo/releases/tag/v1.4.5). On OpenCL, it will also tune for your GPU.
```
./katago.exe benchmark                                                   # if you have default_gtp.cfg and default_model.bin.gz
./katago.exe benchmark -model <NEURALNET>.bin.gz                         # if you have default_gtp.cfg
./katago.exe benchmark -model <NEURALNET>.bin.gz -config gtp_custom.cfg  # use this .bin.gz neural net and this .cfg file
```
It will tell you a good number of threads. Edit your .cfg file and set "numSearchThreads" to that many to get best performance.

**Or**: Run this command to have KataGo generate a custom gtp config for you based on answering some questions:
```
./katago.exe genconfig -model <NEURALNET>.bin.gz -output gtp_custom.cfg
```

**Next**: A command like this will run KataGo's engine. This is the command to give to your [GUI or analysis program](#guis) so that it can run KataGo.
```
./katago.exe gtp                                                   # if you have default_gtp.cfg and default_model.bin.gz
./katago.exe gtp -model <NEURALNET>.bin.gz                         # if you have default_gtp.cfg
./katago.exe gtp -model <NEURALNET>.bin.gz -config gtp_custom.cfg  # use this .bin.gz neural net and this .cfg file
```

You may need to specify different paths when entering KataGo's command for a GUI program, e.g.:
```
path/to/katago.exe gtp -model path/to/<NEURALNET>.bin.gz
path/to/katago.exe gtp -model path/to/<NEURALNET>.bin.gz -config path/to/gtp_custom.cfg
```

#### Human-style Play and Analysis

You can also have KataGo imitate human play if you download the human SL model b18c384nbt-humanv0.bin.gz from https://github.com/lightvector/KataGo/releases/tag/v1.15.0, and run a command like the following, providing both the normal model and the human SL model:
```
./katago.exe gtp -model <NEURALNET>.bin.gz -human-model b18c384nbt-humanv0.bin.gz -config gtp_human5k_example.cfg
```

The [gtp_human5k_example.cfg](cpp/configs/gtp_human5k_example.cfg) configures KataGo to imitate 5-kyu-level players. You can change it to imitate other ranks too, as well as to do many more things, including making KataGo play in a human style but still at a strong level or analyze in interesting ways. Read the config file itself for documentation on some of these possibilities!

And see also [this guide](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md#human-sl-analysis-guide) to using the human SL model, which is written from the perspective of the JSON-based analysis engine mentioned below, but is also applicable to gtp as well.

#### Other Commands:

Run a JSON-based [analysis engine](docs/Analysis_Engine.md) that can do efficient batched evaluations for a backend Go service:

   * `./katago analysis -model <NEURALNET>.gz -config <ANALYSIS_CONFIG>.cfg`

Run a high-performance match engine that will play a pool of bots against each other sharing the same GPU batches and CPUs with each other:

   * `./katago match -config <MATCH_CONFIG>.cfg -log-file match.log -sgf-output-dir <DIR TO WRITE THE SGFS>`

Force OpenCL tuner to re-tune:

   * `./katago tuner -config <GTP_CONFIG>.cfg`

Print version:

   * `./katago version`


### Tuning for Performance

The most important parameter to optimize for KataGo's performance is the number of threads to use - this can easily make a factor of 2 or 3 difference.

Secondarily, you can also read over the parameters in your GTP config (`default_gtp.cfg` or `gtp_example.cfg` or `configs/gtp_example.cfg`, etc). A lot of other settings are described in there that you can set to adjust KataGo's resource usage, or choose which GPUs to use. You can also adjust things like KataGo's resign threshold, pondering behavior or utility function. Most parameters are documented directly inline in the [example config file](cpp/configs/gtp_example.cfg). Many can also be interactively set when generating a config via the `genconfig` command described above.


### Common Questions and Issues
This section summarizes a number of common questions and issues when running KataGo.

#### Issues with specific GPUs or GPU drivers
If you are observing any crashes in KataGo while attempting to run the benchmark or the program itself, and you have one of the below GPUs, then this is likely the reason.

* **AMD Radeon RX 5700** - AMD's drivers for OpenCL for this GPU have been buggy ever since this GPU was released, and as of May 2020 AMD has still never released a fix. If you are using this GPU, you will just not be able to run KataGo (Leela Zero and other Go engines will probably fail too) and will probably also obtain incorrect calculations or crash if doing anything else scientific or mathematical that uses OpenCL. See for example these reddit threads: [[1]](https://www.reddit.com/r/Amd/comments/ebso1x/its_not_just_setihome_any_mathematic_or/) or [[2]](https://www.reddit.com/r/BOINC/comments/ebiz18/psa_please_remove_your_amd_rx5700xt_from_setihome/) or this [L19 thread](https://lifein19x19.com/viewtopic.php?f=18&t=17093).
* **OpenCL Mesa** - These drivers for OpenCL are buggy. Particularly if on startup before crashing you see KataGo printing something like
`Found OpenCL Platform 0: ... (Mesa) (OpenCL 1.1 Mesa ...) ...`
then you are using the Mesa drivers. You will need to change your drivers, see for example this [KataGo issue](https://github.com/lightvector/KataGo/issues/182#issuecomment-607943405) which links to [this thread](https://bbs.archlinux.org/viewtopic.php?pid=1895516#p1895516).
* **Intel Integrated Graphics** - For weaker/older machines or laptops or devices that don't have a dedicated GPU, KataGo might end up using the weak "Intel Integrated Graphics" that is built in with the CPU. Often this will work fine (although KataGo will be slow and only get a tiny number of playouts compared to using a real GPU), but various versions of Intel Integrated Graphics can also be buggy and not work at all. If a driver update doesn't work for you, then the only solution is to upgrade to a better GPU. See for example this [issue](https://github.com/lightvector/KataGo/issues/54) or this [issue](https://github.com/lightvector/KataGo/issues/78), or this [other Github's issue](https://github.com/CNugteren/CLBlast/issues/280).

#### Common Problems
* **KataGo seems to hang or is "loading" forever on startup in Lizzie/Sabaki/q5go/GoReviewPartner/etc.**
   * Likely either you have some misconfiguration, have specified file paths incorrectly, a bad GPU, etc. Many of these GUIs do a poor job of reporting errors and may completely swallow the error message from KataGo that would have told you what was wrong. Try running KataGo's `benchmark` or `gtp` directly on the command line, as described [above](#how-to-use).
   * Sometimes there is no error at all, it is merely that the *first* time KataGo runs on a given network size, it needs to do some expensive tuning, which may take a few minutes. Again this is clearer if you run the `benchmark` command directly in the command line. After tuning, then subsequent runs will be faster.

* **KataGo works on the command line but having trouble specifying the right file paths for the GUI.**
   * As described [above](#how-to-use), you can name your config `default_gtp.cfg` and name whichever network file you've downloaded to `default_model.bin.gz` (for newer `.bin.gz` models) or `default_model.txt.gz` (for older `.txt.gz` models). Stick those into the same directory as KataGo's executable, and then you don't need to specify `-config` or `-model` paths at all.

* **KataGo gives an error like `Could not create file` when trying to run the initial tuning.**
   * KataGo probably does not have access permissions to write files in the directory where you placed it.
   * On Windows for example, the `Program Files` directory and its subdirectories are often restricted to only allow writes with admin-level permissions. Try placing KataGo somewhere else.

* **I'm new to the command line and still having trouble knowing what to tell Lizzie/q5go/Sabaki/whatever to make it run KataGo**.
   * Again, make sure you have your directory paths right.
   * A common issue: AVOID having any spaces in any file or directory names anywhere, since depending on the GUI, this may require you to have to quote or character-escape the paths or arguments in various ways.
   * If you don't understand command line arguments and flags, relative vs absolute file paths, etc, search online. Try pages like https://superuser.com/questions/1270591/how-to-use-relative-paths-on-windows-cmd or https://www.bleepingcomputer.com/tutorials/understanding-command-line-arguments-and-how-to-use-them/ or other pages you find, or get someone tech-savvy to help you in a chat or even in-person if you can.
   * Consider using https://github.com/sanderland/katrain instead - this is an excellent GUI written by someone else for KataGo that usually automates all of the technical setup for you.

* **I'm getting a different error or still want further help.**
   * Check out [the discord chat where Leela Zero, KataGo, and other bots hang out](https://discord.gg/bqkZAz3) and ask in the "#help" channel.
   * If you think you've found a bug in KataGo itself, feel free also to [open an issue](https://github.com/lightvector/KataGo/issues). Please provide as much detail as possible about the exact commands you ran, the full error message and output (if you're in a GUI, please make sure to check that GUI's raw GTP console or log), the things you've tried, your config file and network, your GPU and operating system, etc.

#### Other Questions
* **How do I make KataGo use Japanese rules or other rules?**
   * KataGo supports some [GTP extensions](docs/GTP_Extensions.md) for developers of GUIs to set the rules, but unfortunately as of June 2020, only a few of them make use of this. So as a workaround, there are a few ways:
     * Edit KataGo's config (`default_gtp.cfg` or `gtp_example.cfg` or `gtp.cfg`, or whatever you've named it) to use `rules=japanese` or `rules=chinese` or whatever you need, or set the individual rules `koRule`,`scoringRule`,`taxRule`, etc. to what they should be. See [here](https://github.com/lightvector/KataGo/blob/master/cpp/configs/gtp_example.cfg#L91) for where this is in the config, or and see [this webpage](https://lightvector.github.io/KataGo/rules.html) for the full description of KataGo's ruleset.
     * Use the `genconfig` command (`./katago genconfig -model <NEURALNET>.gz -output <PATH_TO_SAVE_GTP_CONFIG>.cfg`) to generate a config, and it will interactively help you, including asking you for what default rules you want.
     * If your GUI allows access directly to the GTP console (for example, press `E` in Lizzie), then you can run `kata-set-rules japanese` or similar for other rules directly in the GTP console, to change the rules dynamically in the middle of a game or an analysis session.

* **Which model/network should I use?**
   * Generally, use the strongest or most recent b18-sized net (b18c384nbt) from [the main training site](https://katagotraining.org/). This will be the best neural net even for weaker machines, since despite being a bit slower than old smaller nets, it is much stronger and more accurate per evaluation.
   * If you care a lot about theoretical purity - no outside data, bot learns strictly on its own - use the 20 or 40 block nets from [this release](https://github.com/lightvector/KataGo/releases/tag/v1.4.0), which are pure in this way and still much stronger than Leela Zero, but also much weaker than more recent nets.
   * If you want some nets that are much faster to run, and each with their own interesting style of play due to their unique stages of learning, try any of the "b10c128" or "b15c192" Extended Training Nets [here](https://katagoarchive.org/g170/neuralnets/index.html) which are 10 block and 15 block networks from earlier in the run that are much weaker but still pro-level-and-beyond.


## Features for Developers

#### GTP Extensions:
In addition to a basic set of [GTP commands](https://www.lysator.liu.se/~gunnar/gtp/), KataGo supports a few additional commands, for use with analysis tools and other programs.

KataGo's GTP extensions are documented **[here](docs/GTP_Extensions.md)**.

   * Notably: KataGo exposes a GTP command `kata-analyze` that in addition to policy and winrate, also reports an estimate of the *expected score* and a heatmap of the predicted territory ownership of every location of the board. Expected score should be particularly useful for reviewing handicap games or games of weaker players. Whereas the winrate for black will often remain pinned at nearly 100% in a handicap game even as black makes major mistakes (until finally the game becomes very close), expected score should make it more clear which earlier moves are losing points that allow white to catch up, and exactly how much or little those mistakes lose. If you're interested in adding support for this to any analysis tool, feel free to reach out, I'd be happy to answer questions and help.

   * KataGo also exposes a few GTP extensions that allow setting what rules are in effect (Chinese, AGA, Japanese, etc). See again [here](docs/GTP_Extensions.md) for details.

#### Analysis Engine:
KataGo also implements a separate engine that can evaluate much faster due to batching if you want to analyze whole games at once and might be much less of a hassle than GTP if you are working in an environment where JSON parsing is easy. See [here](docs/Analysis_Engine.md) for details.

KataGo also includes example code demonstrating how you can invoke the analysis engine from Python, see [here](python/query_analysis_engine_example.py)!

## Compiling KataGo
KataGo is written in C++. It should compile on Linux or OSX via g++ that supports at least C++14, or on Windows via MSVC 15 (2017) and later. Instructions may be found at [Compiling KataGo](Compiling.md).

## Source Code Overview:
See the [cpp readme](cpp/README.md) or the [python readme](python/README.md) for some high-level overviews of the source code in this repo, if you want to get a sense of what is where and how it fits together.

## Selfplay Training:
If you'd also like to run the full self-play loop and train your own neural nets using the code here, see [Selfplay Training](SelfplayTraining.md).

## Contributors

Many thanks to the various people who have contributed to this project! See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.

## License

Except for several external libraries that have been included together in this repo under `cpp/external/` as well as the single file `cpp/core/sha2.cpp`, which all have their own individual licenses, all code and other content in this repo is released for free use or modification under the license in the following file: [LICENSE](LICENSE).

License aside, if you end up using any of the code in this repo to do any of your own cool new self-play or neural net training experiments, I (lightvector) would to love hear about it.
