#!/usr/bin/env bash
#
# ladder_step.sh — one autonomous step of the Human-SL rank-ladder λ calibration.
#
# Uses tune_decide.py (the decision brain: pools all (λ,winrate,games) data for the current
# rank, isotonic-fits the 50% crossing, decides GRIND/LOCK/STOP) to drive the ladder without
# human noise-chasing. Each invocation does exactly one of:
#   GRIND -> run ONE ~23-min tunehuman chunk at the recommended λ (komi-0.5 Japanese handicap)
#   LOCK  -> winrate CI ⊂ [40,60]: write gtp_human<rank>.cfg, build next ANE baseline, advance
#   STOP  -> best-effort (λ>1e8 or >500g at best λ): write config anyway, advance, flag it
#
# Invoke repeatedly (the /loop re-invokes on each chunk completion). State persists in
# ladder_state.txt + the per-λ jpn<rank>_ane_L*.samples checkpoints, so it survives the
# process-kill cap. Run THIS in the background; it foregrounds one chunk then exits.
set -u
ROOT=/Users/chinchangyang/Code/KataGo-MLX
TUNE=$HOME/.katago_tune
CONFIGS=$ROOT/cpp/configs
STATE=$TUNE/ladder_state.txt
LOCKLOG=$TUNE/ladder_locks.txt
TIMEOUT=${TIMEOUT:-1400}

# rank chain 9d..1d then 1k..20k (functions, not assoc arrays — macOS bash 3.2 lacks `declare -A`)
stronger() { case "$1" in
  8d)echo 9d;;7d)echo 8d;;6d)echo 7d;;5d)echo 6d;;4d)echo 5d;;3d)echo 4d;;2d)echo 3d;;1d)echo 2d;;
  1k)echo 1d;; *k) echo "$(( ${1%k} - 1 ))k";; *)echo "";; esac; }
weaker()   { case "$1" in
  9d)echo 8d;;8d)echo 7d;;7d)echo 6d;;6d)echo 5d;;5d)echo 4d;;4d)echo 3d;;3d)echo 2d;;2d)echo 1d;;1d)echo 1k;;
  *k) n=${1%k}; if [ "$n" -ge 20 ]; then echo DONE; else echo "$((n+1))k"; fi;; *)echo "";; esac; }

RANK=$(cat "$STATE" 2>/dev/null || echo 8d)
if [ "$RANK" = DONE ]; then echo "LADDER COMPLETE — all rungs 8d..1d, 1k..20k done."; exit 0; fi
STR=$(stronger "$RANK")
[ -z "$STR" ] && { echo "ERROR: unknown rank '$RANK'"; exit 2; }
PROFILE=preaz_$RANK
BASELINE=$TUNE/tunebase_human${STR}_ane.cfg
[ -f "$BASELINE" ] || { echo "ERROR: baseline $BASELINE missing (need tuned ${STR} first)"; exit 2; }

DEC=$(python3 "$TUNE/tune_decide.py" "$TUNE/jpn${RANK}_ane_L*.samples")
echo "[$(date '+%H:%M:%S')] rank=$RANK baseline=$STR  ->  $DEC"
ACTION=$(printf '%s' "$DEC" | sed -n 's/.*ACTION=\([A-Z]*\).*/\1/p')
LAMBDA=$(printf '%s' "$DEC" | sed -n 's/.*LAMBDA=\([0-9.]*\).*/\1/p')

case "$ACTION" in
  GRIND)
    if [ -z "$LAMBDA" ] || [ "$LAMBDA" = NA ]; then
      # fresh rank, no data: seed λ by EXTRAPOLATING the last λ-step (the steps grow for weaker
      # ranks). seed = stronger λ + (stronger λ - grandparent λ)*1.15; fall back to +0.04.
      SLAM=$(grep -E '^humanSLChosenMovePiklLambda' "$CONFIGS/gtp_human${STR}.cfg" | awk '{print $3}')
      GP=$(stronger "$STR")
      GLAM=$(grep -E '^humanSLChosenMovePiklLambda' "$CONFIGS/gtp_human${GP}.cfg" 2>/dev/null | awk '{print $3}')
      LAMBDA=$(python3 -c "s=float('${SLAM:-0.05}'); g='${GLAM}'; seed=(s+float(g))/2 if g else s+0.04; print(f'{max(0.001, seed):.5f}')")
      echo "  seeding fresh rank $RANK at λ=$LAMBDA (extrapolated from ${STR}=${SLAM}, ${GP}=${GLAM:-NA})"
    fi
    LAMTAG=$(printf '%s' "$LAMBDA" | sed 's/^0\.//; s/0*$//')   # 0.08680 -> 0868, 0.08677 -> 08677
    for p in $(ps aux | grep "[k]atago tunehuman" | awk '{print $2}'); do kill -9 "$p" 2>/dev/null; done
    sleep 1
    BASELINE_CFG=$BASELINE CAND_PROFILE=$PROFILE PIKL=$LAMBDA V_LO=400 V_HI=400 \
      KOMI=0.5 CAND_COLOR=black TARGET_ELO=0 ELO_TOL=8 \
      GAMES_PER_ROUND=4 GAME_THREADS=4 TIMEOUT=$TIMEOUT \
      TAG=jpn${RANK}_ane_L${LAMTAG} \
      RESUME=$TUNE/jpn${RANK}_ane_L${LAMTAG}.samples \
      LOG=$TUNE/jpn${RANK}_ane_L${LAMTAG}.log \
      OUT=$TUNE/jpn${RANK}_ane_L${LAMTAG}.out.cfg \
      "$ROOT/tune_maxvisits.sh"
    ;;
  LOCK|STOP)
    WR=$(printf '%s' "$DEC" | sed -n 's/.*WR=\([0-9.]*\).*/\1/p')
    CI=$(printf '%s' "$DEC" | sed -n 's/.*CI=\([0-9.,]*\).*/\1/p')
    N=$(printf '%s' "$DEC" | sed -n 's/.*N=\([0-9]*\).*/\1/p')
    DST=$CONFIGS/gtp_human${RANK}.cfg
    SRC=$DST; [ -f "$DST" ] || SRC=$CONFIGS/gtp_human${STR}.cfg   # update-in-place if it exists
    CALC="# CALIBRATED (Japanese, komi-0.5 handicap, ANE): preaz_${RANK} (Black) vs gtp_human${STR}.cfg (White) = ${WR}% [${CI}] over ${N} games. λ=${LAMBDA}. ${ACTION}."
    sed -e "s/^humanSLProfile *=.*/humanSLProfile = ${PROFILE}/" \
        -e "s/^humanSLChosenMovePiklLambda *=.*/humanSLChosenMovePiklLambda = ${LAMBDA}/" \
        "$SRC" > "$DST.tmp"
    { echo "$CALC"; cat "$DST.tmp"; } > "$DST"; rm -f "$DST.tmp"
    echo "WROTE $DST   λ=${LAMBDA}  ${ACTION}  ${WR}% [${CI}] ${N}g"
    echo "$(date '+%F %T')  ${RANK}  λ=${LAMBDA}  ${WR}% CI[${CI}] ${N}g  ${ACTION}" >> "$LOCKLOG"
    # build the ANE tuning baseline for the next (weaker) rung
    NB=$TUNE/tunebase_human${RANK}_ane.cfg
    sed -e 's/^nnCacheSizePowerOfTwo *=.*/nnCacheSizePowerOfTwo = 18/' \
        -e 's/^nnMutexPoolSizePowerOfTwo *=.*/nnMutexPoolSizePowerOfTwo = 12/' "$DST" > "$NB"
    { echo ""; echo "# ANE-mux tuning baseline (GPU thread0 + ANE thread1); cache lowered — no play effect.";
      echo "numNNServerThreadsPerModel = 2"; echo "deviceToUseThread0 = 0"; echo "deviceToUseThread1 = 100"; } >> "$NB"
    NEXT=$(weaker "$RANK")
    echo "$NEXT" > "$STATE"
    echo "ADVANCED $RANK -> $NEXT   (next baseline: $NB)"
    ;;
  *) echo "ERROR: could not parse action from: $DEC"; exit 3 ;;
esac
