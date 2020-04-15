#!/bin/bash -eu
set -o pipefail

# Runs the entire self-play process synchronously in a loop, training a single size of neural net appropriately.
# Assumes you have the cpp directory compiled and the katago executable is there.

# If using multiple machines, or even possibly many GPUs on one machine in some cases, then this is NOT the
# recommended method, instead it is better to run all steps simultaneously and asynchronously. See README.md in
# the root of the KataGo repo for more details.

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND USEGATING"
    echo "Assumes katago is built in the `cpp` directory of the KataGo repo and the executable is present at cpp/katago."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like 'b10c128', see ../modelconfigs.py"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIRRAW="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
USEGATING="$1"
shift

BASEDIR="$(realpath "$BASEDIRRAW")"

GITROOTDIR="$(git rev-parse --show-toplevel)"

LOGSDIR="$BASEDIR"/logs
SCRATCHDIR="$BASEDIR"/shufflescratch
mkdir -p "$BASEDIR"
mkdir -p "$LOGSDIR"
mkdir -p "$SCRATCHDIR"
mkdir -p "$BASEDIR"/selfplay
mkdir -p "$BASEDIR"/gatekeepersgf

# NOTE: You probably want to edit settings in the cpp/configs/selfplay1.cfg.
# NOTE: You may want to adjust these numbers.
NUM_GAMES_PER_CYCLE=1000
NUM_THREADS_FOR_SHUFFLING=8
NUM_TRAIN_SAMPLES_PER_CYCLE=500000
BATCHSIZE=128 # KataGo normally uses batch size 256, and you can do that too, but for lower-end GPUs 64 or 128 may be needed to avoid running out of memory.
SHUFFLE_MINROWS=80000
SHUFFLE_KEEPROWS=600000 # A little larger than NUM_TRAIN_SAMPLES_PER_CYCLE

set -x
while true
do
    echo "Selfplay"
    time "$GITROOTDIR"/cpp/katago selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$GITROOTDIR"/cpp/configs/selfplay1.cfg | tee -a "$BASEDIR"/selfplay/stdout.txt

    echo "Shuffle"
    (
        cd "$GITROOTDIR"/python
        time ./selfplay/shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" "$BATCHSIZE" -min-rows "$SHUFFLE_MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS"
    )

    echo "Train"
    time "$GITROOTDIR"/python/selfplay/train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main -max-epochs-this-instance 1 -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_CYCLE"

    echo "Export"
    (
        cd "$GITROOTDIR"/python
        time ./selfplay/export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" "$USEGATING"
    )

    echo "Gatekeeper"
    time "$GITROOTDIR"/cpp/katago gatekeeper -rejected-models-dir "$BASEDIR"/rejectedmodels -accepted-models-dir "$BASEDIR"/models/ -sgf-output-dir "$BASEDIR"/gatekeepersgf/ -test-models-dir "$BASEDIR"/modelstobetested/ -config "$GITROOTDIR"/cpp/configs/gatekeeper1.cfg -quit-if-no-nets-to-test | tee -a "$BASEDIR"/gatekeepersgf/stdout.txt
done
