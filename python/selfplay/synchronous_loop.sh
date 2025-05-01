#!/bin/bash -eu
set -o pipefail
{

# Runs the entire self-play process synchronously in a loop, training a single size of neural net appropriately.
# Assumes you have the cpp directory compiled and the katago executable is there.

# If using multiple machines, or even possibly many GPUs on one machine in some cases, then this is NOT the
# recommended method, instead it is better to run all steps simultaneously and asynchronously. See SelfplayTraining.md in
# the root of the KataGo repo for more details.

if [[ $# -lt 8 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TRAININGNAME MODELKIND USEGATING NOISESCALE ITERATIONS LRSCALE"
    echo "Assumes katago is already built in the 'cpp' directory of the KataGo repo and the executable is present at cpp/katago."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like 'b10c128', see ../modelconfigs.py"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper"
    echo "NOISESCALE amount of noise to add to the model (e.g. 0.1)"
    echo "ITERATIONS number of iterations to add noise (e.g. 1000)"
    echo "LRSCALE learning rate scaler for the model, default 1.0"
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
NOISESCALE="$1"
shift
ITERATIONS="$1"
shift
LRSCALE="$1"
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
# NOTE: You probably want to edit settings in cpp/configs/training/selfplay1.cfg
# NOTE: You probably want to edit settings in cpp/configs/training/gatekeeper1.cfg
# Such as what board sizes and rules, you want to learn, number of visits to use, etc.

# Also, the parameters below are relatively small, and probably
# good for less powerful hardware and tighter turnaround during very early training, but if
# you have strong hardware or are later into a run you may want to reduce the overhead by scaling
# these numbers up and doing more games and training per cycle, exporting models less frequently, etc.

# 标准 128 盘本地自对弈参数设置
NUM_GAMES_PER_CYCLE=128 # 每周期自对弈游戏数; Every cycle, play this many games
NUM_THREADS_FOR_SHUFFLING=8 # 混洗线程数
NUM_TRAIN_SAMPLES_PER_EPOCH=25000  # 每训练周期样本数. Training will proceed in chunks of this many rows, subject to MAX_TRAIN_PER_DATA.
MAX_TRAIN_PER_DATA=8 # 每数据行最大训练次数（8，避免过拟合）; On average, train only this many times on each data row. Larger numbers may cause overfitting.
NUM_TRAIN_SAMPLES_PER_SWA=80000  # 随机权重平均频率; Stochastic weight averaging frequency.
BATCHSIZE=32 # 训练批次大小; For lower-end GPUs 64 or smaller may be needed to avoid running out of GPU memory.
SHUFFLE_MINROWS=25000 # 开始训练前的最小数据行数; Require this many rows at the very start before beginning training.
MAX_TRAIN_SAMPLES_PER_CYCLE=1280000  # 每周期最大训练步数; Each cycle will do at most this many training steps.
TAPER_WINDOW_SCALE=10000 # 混洗窗口增长的缩放因子; Parameter setting the scale at which the shuffler will make the training window grow sublinearly.
SHUFFLE_KEEPROWS=2560000 # 混洗保留的行数（需大于最大训练步数）; Needs to be larger than MAX_TRAIN_SAMPLES_PER_CYCLE, so the shuffler samples enough rows each cycle for the training to use.

# # 标准 256 盘本地自对弈参数设置
# NUM_GAMES_PER_CYCLE=256 # 每周期自对弈游戏数; Every cycle, play this many games
# NUM_THREADS_FOR_SHUFFLING=8 # 混洗线程数
# NUM_TRAIN_SAMPLES_PER_EPOCH=52000  # 每训练周期样本数. Training will proceed in chunks of this many rows, subject to MAX_TRAIN_PER_DATA.
# MAX_TRAIN_PER_DATA=8 # 每数据行最大训练次数（8，避免过拟合）; On average, train only this many times on each data row. Larger numbers may cause overfitting.
# NUM_TRAIN_SAMPLES_PER_SWA=80000  # 随机权重平均频率; Stochastic weight averaging frequency.
# BATCHSIZE=32 # 训练批次大小; For lower-end GPUs 64 or smaller may be needed to avoid running out of GPU memory.
# SHUFFLE_MINROWS=52000 # 开始训练前的最小数据行数; Require this many rows at the very start before beginning training.
# MAX_TRAIN_SAMPLES_PER_CYCLE=1280000  # 每周期最大训练步数; Each cycle will do at most this many training steps.
# TAPER_WINDOW_SCALE=10000 # 混洗窗口增长的缩放因子; Parameter setting the scale at which the shuffler will make the training window grow sublinearly.
# SHUFFLE_KEEPROWS=2560000 # 混洗保留的行数（需大于最大训练步数）; Needs to be larger than MAX_TRAIN_SAMPLES_PER_CYCLE, so the shuffler samples enough rows each cycle for the training to use.

# NUM_GAMES_PER_CYCLE=500 # 每周期自对弈游戏数; Every cycle, play this many games
# NUM_THREADS_FOR_SHUFFLING=8 # 混洗线程数
# NUM_TRAIN_SAMPLES_PER_EPOCH=270000  # 每训练周期样本数. Training will proceed in chunks of this many rows, subject to MAX_TRAIN_PER_DATA.
# MAX_TRAIN_PER_DATA=8 # 每数据行最大训练次数（8，避免过拟合）; On average, train only this many times on each data row. Larger numbers may cause overfitting.
# NUM_TRAIN_SAMPLES_PER_SWA=80000  # 随机权重平均频率; Stochastic weight averaging frequency.
# BATCHSIZE=32 # 训练批次大小; For lower-end GPUs 64 or smaller may be needed to avoid running out of GPU memory.
# SHUFFLE_MINROWS=270000 # 开始训练前的最小数据行数; Require this many rows at the very start before beginning training.
# MAX_TRAIN_SAMPLES_PER_CYCLE=128000000  # 每周期最大训练步数; Each cycle will do at most this many training steps.
# TAPER_WINDOW_SCALE=50000 # 混洗窗口增长的缩放因子; Parameter setting the scale at which the shuffler will make the training window grow sublinearly.
# SHUFFLE_KEEPROWS=256000000 # 混洗保留的行数（需大于最大训练步数）; Needs to be larger than MAX_TRAIN_SAMPLES_PER_CYCLE, so the shuffler samples enough rows each cycle for the training to use.

# # 其它参数设置较大，因使用社区数据集进行训练
# NUM_GAMES_PER_CYCLE=500 # 每周期自对弈游戏数; Every cycle, play this many games
# NUM_THREADS_FOR_SHUFFLING=8 # 混洗线程数
# NUM_TRAIN_SAMPLES_PER_EPOCH=380000 # 每训练周期样本数. Training will proceed in chunks of this many rows, subject to MAX_TRAIN_PER_DATA.
# MAX_TRAIN_PER_DATA=8 # 每数据行最大训练次数（8，避免过拟合）; On average, train only this many times on each data row. Larger numbers may cause overfitting.
# NUM_TRAIN_SAMPLES_PER_SWA=80000 # 随机权重平均频率; Stochastic weight averaging frequency.
# BATCHSIZE=32 # 训练批次大小; For lower-end GPUs 64 or smaller may be needed to avoid running out of GPU memory.
# SHUFFLE_MINROWS=380000 # 开始训练前的最小数据行数; Require this many rows at the very start before beginning training.
# MAX_TRAIN_SAMPLES_PER_CYCLE=128000000 # 每周期最大训练步数; Each cycle will do at most this many training steps.
# TAPER_WINDOW_SCALE=500000 # 混洗窗口增长的缩放因子; Parameter setting the scale at which the shuffler will make the training window grow sublinearly.
# SHUFFLE_KEEPROWS=256000000 # 混洗保留的行数（需大于最大训练步数）; Needs to be larger than MAX_TRAIN_SAMPLES_PER_CYCLE, so the shuffler samples enough rows each cycle for the training to use.


# Paths to the selfplay and gatekeeper configs that contain board sizes, rules, search parameters, etc.
# See cpp/configs/training/README.md for some notes on other selfplay configs.
SELFPLAY_CONFIG="$GITROOTDIR"/cpp/configs/training/selfplay1.cfg
GATING_CONFIG="$GITROOTDIR"/cpp/configs/training/gatekeeper1.cfg

# Copy all the relevant scripts and configs and the katago executable to a dated directory.
# For archival and logging purposes - you can look back and see exactly the python code on a particular date
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$BASEDIR"/scripts/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/*.sh "$DATED_ARCHIVE"
cp "$GITROOTDIR"/cpp/katago "$DATED_ARCHIVE"
cp "$GITROOTDIR"/cpp/*.dll "$DATED_ARCHIVE"

cp "$SELFPLAY_CONFIG" "$DATED_ARCHIVE"/selfplay.cfg
cp "$GATING_CONFIG" "$DATED_ARCHIVE"/gatekeeper.cfg
git show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt
git diff --no-color > "$DATED_ARCHIVE"/diff.txt
git diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt

# Also run the code out of the archive, so that we don't unexpectedly crash or change behavior if the local repo changes.
cd "$DATED_ARCHIVE"

i=1 # Initialize the loop counter

# 改为可自定义训练周期数
set -x
while [ "$i" -le 1 ] 
do
    echo "Selfplay"
    time ./katago selfplay -max-games-total "$NUM_GAMES_PER_CYCLE" -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config "$DATED_ARCHIVE"/selfplay.cfg | tee -a "$BASEDIR"/selfplay/stdout.txt

    echo "Shuffle"
    (
        # Skip validate since peeling off 5% of data is actually a bit too chunky and discrete when running at a small scale, and validation data
        # doesn't actually add much to debugging a fast-changing RL training.
        time SKIP_VALIDATE=1 ./shuffle.sh "$BASEDIR" "$SCRATCHDIR" "$NUM_THREADS_FOR_SHUFFLING" "$BATCHSIZE" -min-rows "$SHUFFLE_MINROWS" -keep-target-rows "$SHUFFLE_KEEPROWS" -taper-window-scale "$TAPER_WINDOW_SCALE" | tee -a "$BASEDIR"/logs/outshuffle.txt
    )

    echo "Train"
    time ./train.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$BATCHSIZE" "$LRSCALE" main -samples-per-epoch "$NUM_TRAIN_SAMPLES_PER_EPOCH" -swa-period-samples "$NUM_TRAIN_SAMPLES_PER_SWA" -quit-if-no-data -stop-when-train-bucket-limited -no-repeat-files -max-train-bucket-per-new-data "$MAX_TRAIN_PER_DATA" -max-train-bucket-size "$MAX_TRAIN_SAMPLES_PER_CYCLE"

    echo "Export"
    (
        time ./export_model_for_selfplay.sh "$NAMEPREFIX" "$BASEDIR" "$USEGATING" | tee -a "$BASEDIR"/logs/outexport.txt
    )

    echo "Noise"
    time ./noise.sh "$BASEDIR" "$TRAININGNAME" "$MODELKIND" "$NOISESCALE" "$ITERATIONS" 2>&1 | tee -a "$BASEDIR"/logs/outnoise.txt

    echo "Gatekeeper"
    time ./katago gatekeeper -rejected-models-dir "$BASEDIR"/rejectedmodels -accepted-models-dir "$BASEDIR"/models/ -sgf-output-dir "$BASEDIR"/gatekeepersgf/ -test-models-dir "$BASEDIR"/modelstobetested/ -config "$DATED_ARCHIVE"/gatekeeper.cfg -quit-if-no-nets-to-test | tee -a "$BASEDIR"/gatekeepersgf/stdout.txt

    i=$((i + 1))  # 递增循环计数器; Increment the loop counter
done


exit 0
}
