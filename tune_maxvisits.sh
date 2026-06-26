#!/usr/bin/env bash
#
# tune_maxvisits.sh — tune a human-SL candidate's maxVisits to a target ELO vs a fixed baseline,
# within a 1-sigma CI (default ±25 ELO), using `katago tunehuman`.
#
# Default job: tune preaz_9d's maxVisits to reach 0 ELO against rank_9d @ 8 visits.
#
# WHY a script: this environment kills any long process after ~30-45 min, and binomial game noise
# means ±25 ELO needs ~150-200 games near the crossing (several minutes/chunk). So the calibration
# CHECKPOINTS every round to a samples file and RESUMES: just run this script repeatedly until it
# prints "CONVERGED". Each invocation is time-bounded (TIMEOUT, below the kill cap) so it exits
# cleanly, checkpoints, and reports status; the next run picks up where it left off.
#
# Mechanism: the candidate maxVisits is swept via the strength dial's segment C (x in [2,3]), which
# ramps maxVisits from -search-visits (low) to -max-visits-cap (high) at a FIXED piklLambda
# (=-pikl-floor). All other params (profile, humanSLRootExploreProb, temperature, ...) come from the
# baseline config, so the candidate == "preaz_9d at the probe settings" with only maxVisits varying.
#
set -u

# ---- knobs (override via env) -------------------------------------------------------------------
ROOT=/Users/chinchangyang/Code/KataGo-MLX/cpp
KATAGO=${KATAGO:-$ROOT/build_mlx/katago}
MODEL=${MODEL:-$ROOT/models/lionffen_b24c64_3x3_v3_12300.bin.gz}
HUMAN=${HUMAN:-$ROOT/models/b18c384nbt-humanv0.bin.gz}
EXAMPLE=${EXAMPLE:-$ROOT/configs/gtp_human9d_search_example.cfg}   # template (ships preaz_9d)

BASE_PROFILE=${BASE_PROFILE:-rank_9d}     # baseline (reference) human-SL profile
BASE_VISITS=${BASE_VISITS:-8}             # baseline maxVisits  (rank_9d @ 8v)
CAND_PROFILE=${CAND_PROFILE:-preaz_9d}    # candidate profile to tune
TARGET_ELO=${TARGET_ELO:-0}              # candidate - baseline ELO target
ELO_TOL=${ELO_TOL:-25}                    # stop when 1-sigma CI half-width <= this
PIKL=${PIKL:-0.08}                        # FIXED candidate piklLambda (segment C floor)
V_LO=${V_LO:-8}                           # low end of candidate maxVisits sweep
V_HI=${V_HI:-64}                          # high end of candidate maxVisits sweep
GAMES_PER_ROUND=${GAMES_PER_ROUND:-12}
GAME_THREADS=${GAME_THREADS:-10}
TIMEOUT=${TIMEOUT:-1400}                  # seconds per invocation (< the ~30-min process cap)
KOMI=${KOMI:-7.5}
CAND_COLOR=${CAND_COLOR:-auto}

TAG=${TAG:-${CAND_PROFILE}_vs_${BASE_PROFILE}${BASE_VISITS}v}
BASELINE_CFG=${BASELINE_CFG:-/tmp/baseline_${BASE_PROFILE}_${BASE_VISITS}v.cfg}
RESUME=${RESUME:-/tmp/tune_${TAG}.samples}
OUT=${OUT:-/tmp/${CAND_PROFILE}_tuned_${TAG}.cfg}
LOG=${LOG:-/tmp/tune_${TAG}.log}

# ---- build the baseline config once (from the repo example) -------------------------------------
if [ ! -f "$BASELINE_CFG" ]; then
  sed -e "s/^humanSLProfile *= *preaz_9d.*/humanSLProfile = ${BASE_PROFILE}/" \
      -e "s/^maxVisits *= *[0-9]*.*/maxVisits = ${BASE_VISITS}/" \
      "$EXAMPLE" > "$BASELINE_CFG" || { echo "ERROR: could not build baseline"; exit 2; }
  echo "Built baseline $BASELINE_CFG ($BASE_PROFILE @ ${BASE_VISITS}v)"
fi

# ---- one resumable, time-bounded calibration chunk ----------------------------------------------
echo "=== chunk: tuning $CAND_PROFILE maxVisits in [$V_LO,$V_HI] -> ${TARGET_ELO} ELO vs ${BASE_PROFILE}@${BASE_VISITS}v (tol ±${ELO_TOL}) ==="
timeout "$TIMEOUT" "$KATAGO" tunehuman \
  -model "$MODEL" -human-model "$HUMAN" \
  -baseline-config "$BASELINE_CFG" \
  -profile "$CAND_PROFILE" -target-elo "$TARGET_ELO" -elo-tol "$ELO_TOL" \
  -search-visits "$V_LO" -max-visits-cap "$V_HI" -pikl-floor "$PIKL" \
  -komi "$KOMI" -cand-color "$CAND_COLOR" \
  -x-lo 2.0 -x-hi 3.0 \
  -games-per-round "$GAMES_PER_ROUND" -max-rounds 400 \
  -num-game-threads "$GAME_THREADS" \
  -resume-file "$RESUME" -output-config "$OUT" \
  -seed "tune-${TAG}" >> "$LOG" 2>&1

# ---- report status ------------------------------------------------------------------------------
GAMES=$(grep -vcE '^#' "$RESUME" 2>/dev/null || echo 0)   # rounds checkpointed
LAST=$(grep -E "Round [0-9]+:" "$LOG" 2>/dev/null | tail -1)
# direct aggregate winrate -> ELO (cross-check of the logistic fit)
AGG=$(awk 'NR>1{w+=$2;g+=$3} END{if(g>0){wr=w/g; e=400*log(wr/(1-wr))/log(10);
           s=sqrt(wr*(1-wr)/g)*400/log(10)/(wr*(1-wr));
           printf "aggregate %d games: %+.0f ELO ±%.0f",g,e,s}}' "$RESUME" 2>/dev/null)
echo "rounds=$GAMES   $AGG"
echo "$LAST"

if grep -q "converged=yes" "$LOG" 2>/dev/null && [ -f "$OUT" ]; then
  echo "CONVERGED -> $OUT"
  grep -E '^# (achieved|dial)' "$OUT"
  exit 0
else
  echo "NOT-CONVERGED — re-run this script to resume (checkpoint: $RESUME)"
  exit 1
fi
