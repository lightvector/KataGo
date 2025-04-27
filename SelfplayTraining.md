
## Selfplay Training:
If you'd also like to run the full self-play loop and train your own neural nets, in addition to probably wanting to [compile KataGo yourself](https://github.com/lightvector/KataGo#compiling-katago), you must have [Python3](https://www.python.org/) and [Pytorch](https://pytorch.org/) installed. You'll also probably need a decent amount of GPU power.

There are 5 things that need to all run to form a closed self-play training loop.
   * Selfplay engine (C++ - `cpp/katago selfplay`) - continuously plays games using the latest neural net in some directory of accepted models, writing the data to some directory.
   * Shuffler (python - `python/shuffle.py`) - scans directories of data from selfplay and shuffles it to produce npz files to write to some directory.
   * Training (python - `python/train.py`) - continuously trains a neural net using npz files from some directory, saving models periodically to some directory.
   * Exporter (python - `python/export_model.py`) - scans a directory of saved models and converts from Pytorch .ckpt format to the format that all the C++ uses, exporting to some directory.
   * Gatekeeper (C++ - `cpp/katago gatekeeper`) - polls a directory of newly exported models, plays games against the latest model in an accepted models directory, and if the new model passes, moves it to the accepted models directory. OPTIONAL, it is also possible to train just accepting every new model.

### Simple One-Machine Synchronous Training
Although for large-scale runs, KataGo is designed to run asynchronously across many machines for smaller runs it's also possible to run KataGo synchronously on a single machine. An example script is provided in [python/selfplay/synchronous_loop.sh](python/selfplay/synchronous_loop.sh) for how to do this. It loops around, running each of the 5 above steps sequentially. Notably, it passes some extra arguments to each different piece so that it quits rather than runs indefinitely as in asynchronous training:

  * Selfplay engine - Sets `-max-games-total` to the selfplay so it terminates after a certain number of games.
  * Shuffler - Sets smaller values of `-keep-target-rows` for the shuffler to reduce the data per cycle.
  * Training - Sets `-stop-when-train-bucket-limited` for the training to terminate if it has taken too many steps given the amount of new data instead of waiting for more data.
  * Exporter - Only runs it one-shot each cycle of the loop rather than polling forever as in the asynchronous script `python/selfplay/shuffle_and_export_loop.sh`.
  * Gatekeeper - If being used at all, has `-quit-if-no-nets-to-test` so that it terminates after gatekeeping any nets produced by training.

**You probably want to read the comments and edit the parameters in python/selfplay/synchronous_loop.sh and in the selfplay and gatekeeper .cfg files that it points to, to set the board size, the number of parallel threads or other computational parameters, etc.**

Note that not using gating (passing in 0 for `USEGATING` on `python/selfplay/synchronous_loop.sh`) will be faster and will save compute power, and the whole loop works perfectly fine without it, but having it at first can be nice to help debugging and make sure that things are working and that the net is actually getting stronger.

The default parameters in the example synchronous loop script are NOT heavily tested, and unlike the asynchronous setup, have NOT been used for KataGo's primary training runs, so it is quite possible that they are suboptimal, and will need some experimentation. The right parameters may also vary depending on what you're training - for example a 9x9-only run may prefer a different number of samples and windowing policy than 19x19, etc.

The default parameters are also deliberately set to be probably suboptimally/inefficiently small in terms of the amount of work done each cycle of the loop, so as to make cycles of the loop faster while you're trying to get set up and make sure everything is working. You may want to increase these values once things are working and/or a run is mature and successive models are changing by less, depending on what you're doing.

### Asynchronous training
For KataGo's official runs, normally all 5 steps above run simultaneously and _asynchronously_ without ever stopping. Selfplay continuously produces data and polls for new nets, shuffle repeatedly takes the data and shuffles it, training continuously uses the data to produce new nets, etc. This is the most efficient method if using more than one machine in the training loop, since different processes can simply just keep running on their own machine without waiting for steps on any other. To do so, simply just start up each separate process as described above, each one on an appropriate machine, each one using the same base directory. It's expected that this base directory resides on a fast networked file system shared between all machines.

It's recommended to be spending anywhere from 4x to 40x more GPU power on the selfplay than on the training. For the normal asynchronous setup, this is done by simply using more and/or stronger GPUs on the selfplay processes than on training. For synchronous, this can be done by playing around with the various parameters (number of games, visits per move, samples per epoch, etc) and seeing how long each step takes, to find a good balance for your hardware. Note however that very early in a run may be misleading for timing these steps though, since with early barely-better-than-random nets games will last a lot longer than a little further into a run.

On the cloud, a reasonable small-scale setup for all these things might be:
   * A machine with a decent amount of cores and memory to run the shuffler and exporter.
   * A machine with one or two powerful GPUs and a lot of cpus and memory to run the selfplay engine.
   * A machine with a medium GPU and a lot of cpus and memory to run the gatekeeper. (optional, useful for early testing but not needed)
   * A machine with a modest GPU to run the training.
   * A well-performing shared filesystem accessible by all four of these machines.

You may need to play with learning rates, batch sizes, and the balance between training and self-play. If you allocate too much GPU power to training, it will end up waiting a lot of the time for more data to fill the train bucket so it can resume training. Overshooting the other way and having too much GPU power on self-play will result in training being unable to keep up with your configured `-max-train-bucket-per-new-data`, which is fine, but probably means that you should switch more GPU power to training so that it can keep up.

Example instructions to start up these things (assuming you have appropriate machines set up), with some base directory $BASEDIR to hold the all the models and training data generated with a few hundred GB of disk space. The below commands assume you're running from the root of the repo and that you can run bash scripts.
   * **Selfplay engine:** `cpp/katago selfplay -output-dir $BASEDIR/selfplay -models-dir $BASEDIR/models -config cpp/configs/training/SELFPLAYCONFIG.cfg >> log.txt 2>&1 & disown`
     * Some example configs for different numbers of GPUs are: cpp/configs/training/selfplay*.cfg. See [cpp/configs/training/README.md](cpp/configs/training/README.md) for some notes about what the configs are. You may want to copy and edit them depending on your specs - for example to change the sizes of various tables depending on how much memory you have, or to specify gpu indices if you're doing things like putting some mix of training, gatekeeper, and self-play on the same machines or GPUs instead of on separate ones. Note that the number of game threads in these configs is very large, probably far larger than the number of cores on your machine. This is intentional, as each thread only currently runs synchronously with respect to neural net queries, so a large number of parallel games is needed to take advantage of batching.
     * Take a look at the generated `log.txt` for any errors and/or for running stats on started games and occasional neural net query stats.
     * Edit the config to change the number of playouts used or other parameters, or to set a cap on the number of games generated after which selfplay should terminate.
     * If `models-dir` is empty, selfplay will use a random number generator instead to produce data, so selfplay is the **starting point** of setting up the full closed loop.
     * Multiple selfplays across many machines can coexist using the same output dirs on a shared filesystem. Having multiple instances running all pointing to the same directory, ne per machine, is the intended way to run selfplay across a cluster and make use of multiple machines.
   * **Shuffler and exporter:** `cd python; ./selfplay/shuffle_and_export_loop.sh $NAMEOFRUN $BASEDIR/ $SCRATCH_DIRECTORY $NUM_THREADS $BATCH_SIZE $USE_GATING`
     * `$NAMEOFRUN` should be a short alphanumeric string that ideally should be globally unique, to distinguish models from your run if you choose to share your results with others. It will get prefixed on to the internal names of exported models, which will appear in log messages when KataGo loads the model.
     * This starts both the shuffler and exporter. The shuffler will use the scratch directory with the specified number of threads to shuffle in parallel. Make sure you have some disk space. You probably want as many threads as you have cores. If not using the gatekeeper, specify `0` for `$USE_GATING`, else specify `1`.
     * KataGo uses a batch size of 256, but you might have to use a smaller batch size if your GPU has less memory or you are training a very big net.
     * Also, if you're low on disk space, take a look also at the `./selfplay/shuffle.sh` script (which is called by `shuffle_and_export_loop.sh`). Right now it's *very* conservative about cleaning up old shuffles so that it doesn't accidentally delete a shuffle that the training is still reading from, but you could tweak it to be a bit more aggressive.
     * You can also edit `./selfplay/shuffle.sh` if you want to change any details about the lookback window for training data, see `shuffle.py` for more possible arguments.
     * The loop script will output `$BASEDIR/logs/outshuffle.txt` and `$BASEDIR/logs/outexport.txt`, take a look at these to see the output of the shuffle program and/or any errors it encountered.
     * Run `python ./shuffle.py -help` to for information about how the window size is computed, if you want to adjust window size parameters.
   * **Training:** `cd python; ./selfplay/train.sh $BASEDIR/ $TRAININGNAME b6c96 $BATCH_SIZE main -lr-scale 1.0 -max-train-bucket-per-new-data 4 -max-train-bucket-size 5000000 -no-repeat-files >> log.txt 2>&1 & disown`
     * This starts the training. You may want to look at or edit the train.sh script, it also snapshots the state of the repo for logging, as well as contains some training parameters that can be tweaked.
     * `$TRAININGNAME` is a name prefix for the neural net, whose name will follow the convention `$NAMEOFRUN-$TRAININGNAME-s(# of samples trained on)-d(# of data samples generated)`.
     * The batch size specified here MUST match the batch size given to the shuffle script.
     * The fourth argument controls some export behavior:
        * `main` - this is the main net for selfplay, save it regularly to `$BASEDIR/tfsavedmodels_toexport` which the export loop will export regularly for gating.
        * `extra` - save models to `$BASEDIR/tfsavedmodels_toexport_extra`, which the export loop will then export to `$BASEDIR/models_extra`, a directory that does not feed into gating or selfplay.
        * `trainonly` - the neural net without exporting anything. This is useful for when you are trying to jointly train additional models of different sizes and there's no point to have them export anything yet (maybe they're too weak to bother testing).
     * Any additional arguments, like "-lr-scale 1.0" to adjust learning rate will simply get forwarded on to train.py. The argument `-max-epochs-this-instance` can be used to make training terminate after a few epochs, instead of running forever. Run train.py with -help for other arguments.
     * The arguments `-max-train-bucket-per-new-data 4 -max-train-bucket-size 10000000` instruct the training script that it is allowed to perform 4 training steps (measured in rows or samples, not batches) per data row generated by selfplay, the total of which is reported by the shuffler to the training whenever the shuffler outputs a new sampling of rows from the data. However, if more than 2.5M training rows are added at a time, the training will cap out at doing 10M steps at once rather than more. The value of "4" is conservative, you can increase it to train more/faster, but too large will risk problems from overfitting.
     * The argument `-no-repeat-files` makes the training script wait for the shuffler to rerandomize a new sampling of rows if it's already made a full pass over all the shuffled data files from the current shuffle.
     * Take a look at the generated `log.txt` for any possible errors, as well as running stats on training and loss statistics.
     * You can choose a different size than b6c96 if desired. Configuration is in `python/modelconfigs.py`, which you can also edit to add other sizes.
     * If you have a GPU that supports FP16, some other arguments like `-use-fp16` may make the training faster.
     * If you want to export models less frequently, particularly later in a run to reduce model-switching overhead for selfplay, you can also set `-epochs-per-export`.
     * There are many other options, see `--help` and/or look at the source code of `python/train.py` and `python/selfplay/train.sh`.
   * **Gatekeeper:** `cpp/katago gatekeeper -rejected-models-dir $BASEDIR/rejectedmodels -accepted-models-dir $BASEDIR/models/ -sgf-output-dir $BASEDIR/gatekeepersgf/ -test-models-dir $BASEDIR/modelstobetested/ -selfplay-dir $BASEDIR/selfplay/ -config cpp/configs/training/GATEKEEPERCONFIG.cfg >> log.txt 2>&1 & disown`
     * This starts the gatekeeper. Some example configs for different numbers of GPUs are: configs/training/gatekeeper{1,2a,2b,2c}.cfg. Again, you may want to edit these. The number of simultaneous game threads here is also large for the same reasons as for selfplay. No need to start this if specifying `0` for `$USE_GATING`.
     * Take a look at the generated `log.txt` for any errors and/or for the game-by-game progress of each testing match that the gatekeeper runs.
     * The argument `-quit-if-no-nets-to-test` can make gatekeeper terminate after testing all nets queued for testing, instead of running forever and waiting for more. Run with -help to see other arguments as well.
     * Gatekeeper takes `-selfplay-dir` as an argument so as to pre-create the directory so that if there are multiple self-play machines, they don't corrupt a shared filesystem in a race to create the dir.

To manually pause a run, sending `SIGINT` or `SIGKILL` to all the relevant processes is the recommended method. The selfplay and gatekeeper processes will terminate gracefully when receiving such a signal and finish writing all pending data (this may take a minute or two), and any python or bash scripts will be terminated abruptly but are all implemented to write to disk in a way that is safe if killed at any point. To resume the run, just restart everything again with the same `$BASEDIR` and everything will continue where it left off.
