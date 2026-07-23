#!/usr/bin/env bash
#
# tune_lambda.sh — find the piklLambda that makes a Human-SL candidate reach a target ELO vs a fixed
# baseline, AT A FIXED maxVisits, using `katago tunehuman`'s strength-dial segment B (x in [1,2]).
#
# Segment B holds maxVisits = -search-visits and log-interpolates piklLambda from -pikl-max (at x=1,
# most human / weakest) down to -pikl-floor (at x=2, strongest). The calibration searches x to hit the
# target winrate, so the converged x* -> the parity-piklLambda at that visit count.
#
# Use: sweep this over several CAND_VISITS to chart "parity-lambda vs visits" and see whether more
# visits lets the candidate stay at parity with a HIGHER (more human-like) lambda.
#
# Default job: parity-lambda of preaz_9d @ 200v vs rank_9d @ 50v, lambda search range [0.02, 0.08].
#
set -u

ROOT=/Users/chinchangyang/Code/KataGo-MLX/cpp
KATAGO=${KATAGO:-$ROOT/build_mlx/katago}
MODEL=${MODEL:-$ROOT/models/lionffen_b24c64_3x3_v3_12300.bin.gz}
HUMAN=${HUMAN:-$ROOT/models/b18c384nbt-humanv0.bin.gz}
EXAMPLE=${EXAMPLE:-$ROOT/configs/gtp_human9d_search_example.cfg}

BASE_PROFILE=${BASE_PROFILE:-rank_9d}     # baseline (reference) profile
BASE_VISITS=${BASE_VISITS:-50}            # baseline maxVisits (rank_9d@50v ~= rank_9d@400v by visit-inertness)
CAND_PROFILE=${CAND_PROFILE:-preaz_9d}    # candidate profile
CAND_VISITS=${CAND_VISITS:-200}           # FIXED candidate maxVisits (segment-B searchVisits)
TARGET_ELO=${TARGET_ELO:-0}
ELO_TOL=${ELO_TOL:-25}
PIKL_FLOOR=${PIKL_FLOOR:-0.02}            # strongest lambda end (x=2)
PIKL_MAX=${PIKL_MAX:-0.08}               # most-human lambda end (x=1)
GAMES_PER_ROUND=${GAMES_PER_ROUND:-12}
GAME_THREADS=${GAME_THREADS:-8}
TIMEOUT=${TIMEOUT:-1300}
KOMI=${KOMI:-7.5}                         # 0.5 for a KGS 1-rank handicap match
CAND_COLOR=${CAND_COLOR:-auto}            # auto|black|white; black+KOMI=0.5 => weaker candidate gets the 1-rank handicap

TAG=${TAG:-lam_${CAND_PROFILE}_${CAND_VISITS}v_vs_${BASE_PROFILE}${BASE_VISITS}v}
BASELINE_CFG=${BASELINE_CFG:-$HOME/.katago_tune/base_${BASE_PROFILE}_${BASE_VISITS}v.cfg}
RESUME=${RESUME:-$HOME/.katago_tune/${TAG}.samples}
OUT=${OUT:-$HOME/.katago_tune/${TAG}.cfg}
LOG=${LOG:-$HOME/.katago_tune/${TAG}.log}

mkdir -p "$HOME/.katago_tune"
if [ ! -f "$BASELINE_CFG" ]; then
  sed -e "s/^humanSLProfile *= *preaz_9d.*/humanSLProfile = ${BASE_PROFILE}/" \
      -e "s/^maxVisits *= *[0-9]*.*/maxVisits = ${BASE_VISITS}/" \
      "$EXAMPLE" > "$BASELINE_CFG" || { echo "ERROR: could not build baseline"; exit 2; }
  echo "Built baseline $BASELINE_CFG ($BASE_PROFILE @ ${BASE_VISITS}v)"
fi

echo "=== chunk: parity-lambda of $CAND_PROFILE @ ${CAND_VISITS}v -> ${TARGET_ELO} ELO vs ${BASE_PROFILE}@${BASE_VISITS}v (lambda in [$PIKL_FLOOR,$PIKL_MAX], tol +/-${ELO_TOL}) ==="
timeout "$TIMEOUT" "$KATAGO" tunehuman \
  -model "$MODEL" -human-model "$HUMAN" \
  -baseline-config "$BASELINE_CFG" \
  -profile "$CAND_PROFILE" -target-elo "$TARGET_ELO" -elo-tol "$ELO_TOL" \
  -search-visits "$CAND_VISITS" -max-visits-cap "$CAND_VISITS" \
  -pikl-floor "$PIKL_FLOOR" -pikl-max "$PIKL_MAX" \
  -komi "$KOMI" -cand-color "$CAND_COLOR" \
  -x-lo 1.0 -x-hi 2.0 \
  -games-per-round "$GAMES_PER_ROUND" -max-rounds 400 \
  -num-game-threads "$GAME_THREADS" \
  -resume-file "$RESUME" -output-config "$OUT" \
  -seed "lam-${TAG}" >> "$LOG" 2>&1

GAMES=$(grep -vcE '^#' "$RESUME" 2>/dev/null || echo 0)
AGG=$(awk 'NR>1{w+=$2;g+=$3} END{if(g>0){wr=w/g; e=400*log(wr/(1-wr))/log(10);
           s=sqrt(wr*(1-wr)/g)*400/log(10)/(wr*(1-wr));
           printf "aggregate %d games: %+.0f ELO +/-%.0f",g,e,s}}' "$RESUME" 2>/dev/null)
LAST=$(grep -E "Round [0-9]+:" "$LOG" 2>/dev/null | tail -1)
echo "rounds=$GAMES   $AGG"
echo "$LAST"

if grep -q "converged=yes" "$LOG" 2>/dev/null && [ -f "$OUT" ]; then
  echo "CONVERGED -> $OUT"
  grep -E '^# (achieved|dial)' "$OUT"
  exit 0
else
  echo "NOT-CONVERGED — re-run to resume (checkpoint: $RESUME)"
  exit 1
fi
