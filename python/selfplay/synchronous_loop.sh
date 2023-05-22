#!/bin/bash -eu
set -o pipefail
{

# Runs the entire self-play process synchronously in a loop, training a single size of neural net appropriately.
# Assumes you have the cpp directory compiled and the katago executable is there.

# If using multiple machines, or even possibly many GPUs on one machine in some cases, then this is NOT the
# recommended method, instead it is better to run all steps simultaneously and asynchronously. See SelfplayTraining.md in
# the root of the KataGo repo for more details.

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND USEGATING"
    echo "Assumes katago is already built in the 'cpp' directory of the KataGo repo and the executable is present at cpp/katago."
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

# Create all the directories we need
mkdir -p "$BASEDIR"
mkdir -p "$LOGSDIR"
mkdir -p "$SCRATCHDIR"
mkdir -p "$BASEDIR"/selfplay
mkdir -p "$BASEDIR"/gatekeepersgf

# Parameters for the training run
# NOTE: You may want to adjust the below numbers.
# NOTE: You probably want to edit settings in the cpp/configs/training/selfplay1.cfg
# Such as what board sizes and rules, you want to learn, number of visits to use, etc.
NUM_GAMES_PER_CYCLE=1000
NUM_THREADS_FOR_SHUFFLING=8
NUM_TRAIN_SAMPLES_PER_CYCLE=500000
NUM_TRAIN_SAMPLES_PER_SWA=200000
BATCHSIZE=128 # For lower-end GPUs 64 or smaller may be needed to avoid running out of GPU memory.
SHUFFLE_MINROWS=80000
SHUFFLE_KEEPROWS=600000 # A little larger than NUM_TRAIN_SAMPLES_PER_CYCLE
SELFPLAY_CONFIG="$GITROOTDIR"/cpp/configs/training/selfplay1.cfg
GATING_CONFIG="$GITROOTDIR"/cpp/configs/training/gatekeeper1.cfg

# Copy all the relevant scripts and configs and the katago executable to a dated directory.
# For archival and logging purposes - you can look back and see exactly the python code on a particular date
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp "$GITROOTDIR"/cpp/katago "$DATED_ARCHIVE"
cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
cp "$GATING_CONFIG" "$DATED_ARCHIVE"/gatekeeper.cfg
git show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt
git diff --no-color > "$DATED_ARCHIVE"/diff.txt
git diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt

# Also run the code out of the archive, so that we don't unexpectedly crash or change behavior if the local repo changes.
cd "$DATED_ARCHIVE"

# Begin looping forever, running each step in order.
set -x
while true
do
    echo "Selfplay"
    time ./katago selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$DATED_ARCHIVE"/selfplay.cfg | tee -a "$BASEDIR"/selfplay/stdout.txt

    echo "Shuffle"
    (
        time ./shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" "$BATCHSIZE" -min-rows "$SHUFFLE_MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS" | tee -a "$BASEDIR"/logs/outshuffle.txt
    )

    echo "Train"
    time ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" main -max-epochs-this-instance 1 -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_CYCLE" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA"

    echo "Export"
    (
        time ./export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" "$USEGATING" | tee -a "$BASEDIR"/logs/outexport.txt
    )

    echo "Gatekeeper"
    time ./katago gatekeeper -rejected-models-dir "$BASEDIR"/rejectedmodels -accepted-models-dir "$BASEDIR"/models/ -sgf-output-dir "$BASEDIR"/gatekeepersgf/ -test-models-dir "$BASEDIR"/modelstobetested/ -config "$DATED_ARCHIVE"/gatekeeper.cfg -quit-if-no-nets-to-test | tee -a "$BASEDIR"/gatekeepersgf/stdout.txt
done

exit 0
}
