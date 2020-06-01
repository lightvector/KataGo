## C++ Source Code Overview

Summary of source folders, in approximate dependency order, from lowest level to highest, along with a partial list of the most notable files in each directory.

* `external` - External open-source libraries that KataGo depends on that are small or self-contained enough to just include inline with this repo.
* `core` - Low-level utilities, sort of a layer on top of the standard library. Hashing, portable rand, string formatting and parsing, filesystem helpers, etc.
* `game` - Board representation and rules.
  * `rules.{cpp,h}` - Lightweight struct representing all the combinations of rules KataGo supports.
  * `board.{cpp,h}` - Raw board implementation, without move history. Helper functions for Benson's algorithm and ladder search.
  * `boardhistory.{cpp,h}` - Datastructure that does include move history - handles superko, passing, game end, final scoring, komi, handicap detection, etc.
* `neuralnet` - Neural net GPU implementation and interface. Contains both OpenCL and CUDA backends them.
  * `desc.{cpp.h}` - Data structure holding neural net structure and weights.
  * `modelversion.{cpp,h}` - Enumerates the various versions of neural net features and models.
  * `nninputs.{cpp.h}` - Implements the input features for the neural net.
  * `nninterface.h` - Common interface that is implemented by every low-level neural net backend.
  * `{cuda,opencl,dummy}backend.cpp` - Various backends.
  * `nneval.{cpp.h}` - Top-level handle to the neural net used by the rest of the engine, implements thread-safe batching of queries.
* `search` - The main search engine.
  * `timecontrols.cpp` - Basic handling of a few possible time controls.
  * `searchparams.{cpp,h}` - Configurable coefficients and parameters for the search.
  * `search.{cpp,h}` - Multithreaded MCTS implementation.
  * `searchresults.cpp` - Functions to inspect the results of finished searches, select moves, etc.
  * `asyncbot.{cpp,h}` - Simple thread-safe layer on top of main engine to implement pondering.
* `dataio` - SGF reading and writing, writing of self-play training data.
  * `sgf.{cpp.h}` - SGF reading and writing.
  * `trainingwrite.{cpp,h}` - Writing of self-play training data.
* `program` - Top-level helper functions.  neural net, running matches and selfplay games, handicap placement, computing stats to report, etc.
  * `setup.{cpp,h}` - Functions for parsing configs for search parameters, parsing parameters for initializing the neural net.
  * `playutils.{cpp,h}` - Miscellaneous: handicap placement, ownership and final stone status, computing high-level stats to report, benchmarking.
  * `play.{cpp,h}` - Running matches and self-play games.
* `distributed` - Code for talking to https webserver for volunteers to contribute distributed self-play games for training.
* `tests` - A variety of tests.
  * `models` - A directory with a small number of small-sized (and not very strong) models for running tests.
* `command` - Top-level subcommands callable by users. GTP, analysis commands, benchmarking, selfplay data generation, etc.
  * `commandline.{cpp,h}` - Common command line logic shared by all subcommands.
  * `gtp.cpp` - Main GTP engine.
  * `analysis.cpp` - JSON-based analysis engine that can use large batch sizes to analyze positions in parallel.
  * `benchmark.cpp` - Performance benchmarking.
  * `contribute.cpp` - Command for volunteers to contribute distributed self-play games for training.
  * `selfplay.cpp` - Selfplay data generation engine.
  * `gatekeeper.cpp` - Gating engine to filter neural nets for selfplay data generation.
  * `match.cpp` - Match engine for testing different parameters that can use huge batch sizes to efficiently play games in parallel.

Other folders:

* `configs` - Default or example configs for many of the different subcommands.
