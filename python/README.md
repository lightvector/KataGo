## Python Source Code Overview
This is a summary of some of the most notable python files here. Some of the less important or minor scripts are left out, but this covers the most notable ones:

### Main Training Loop
These are the critical scripts needed for training neural nets.

* `tfrecordio.py` - Used by `shuffle.py`, helpers for the TFRecord format for shuffling.
* `shuffle.py` - Shuffles the data for neural net training and writes it into TFRecords, `used by selfplay/shuffle.sh`.
* `model.py` - Implementation of the neural net and loss functions in Tensorflow 1.15.
  * Also contains implementations of the input features, which are NOT used during training because all of it has been computed by the C++ already. But they should work and allong with `board.py` and `data.py` should allow the ability to apply the raw neural net to board positions and SGF files from the python code.
* `modelconfigs.py` - Specific channel and block configurations for different-sized nets. These are codes that you can supply to `train.py` as an argument to pick the net size.
* `train.py` - Trains the neural net using the shuffled data and saves it to a SavedModel periodically, used by `selfplay/train.sh`.
* `export_model.py` - Exports the trained neural net SavedModel to KataGo's .bin.gz format. Used by `selfplay/export_model_for_selfplay.sh`.

And:

* `selfplay/*.sh` - A variety of bash scripts used in [selfplay training](../SelfplayTraining.md) that wrap the above python scripts with logic to loop repeatedly, to pipe output to files, to daemonize and disown so as to keep running when a given shell session ends, and to move stuff around between directories to hook everything up.

### Auxiliary scripts
These are some scripts on the side that might be useful for debugging or experimentation. These are more or less NOT used in training, because normally the C++ selfplay code does all of the work in implementing the board, writing out the input features and data produced from selfplay games, etc. These scripts are a bit more likely than the ones above to suffer from code rot and to be stale, since they aren't used a lot.

* `board.py` - Implementation of Go board, analogous to ../cpp/game/board.{cpp,h}. NOT used in training.
* `data.py` - Loads SGF files.
* `test.py` - Runs a model on some shuffled TFRecord data *without* doing gradient updates - instead just reports the stats.
* `visualize.py` - Slightly sandboxy script, lets you dump out the raw weight tensors from a model.

And:
* `play.py` - A working very primitive GTP engine that just uses the raw neural net to make moves. Also implements a ton of GTP extensions to draw colored heatmaps in GoGUI. This or `test.py` are good places to look if you're interested in how to use the raw Tensorflow weights for direct inference in python.


### Distributed backend
These are some backend scripts for the distributed training. You should never need them, unless somehow you are running your own full website (https://github.com/katago/katago-server) on Docker or Kubernetes.
* `upload_model.py`
* `upload_poses.py`

### Genboard
These are a separate set of scripts that have nothing to do with any of the above training scripts, or even with KataGo in general. They use Pytorch to train a neural net that can "fill in" the rest of a board with plausible stone configurations given part of the board. See https://github.com/lightvector/KataGo/pull/347 for a little bit of info on usage, including a pretrained neural net.
* `genboard_common.py`
* `genboard_run.py`
* `genboard_train.py`


