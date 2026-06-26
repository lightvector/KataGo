# Human-SL KGS-Rank Ladder

A set of GTP configs that make KataGo (with the Human-SL net) play at a chosen amateur
rank, from **9d (top)** down to **20k**, where **each consecutive rank is exactly 1 KGS rank
(1 handicap stone) apart**. The ladder is anchored at 9d; every weaker rank is tuned to be
exactly 1 rank below the rank above it.

> **Note on history:** an earlier version of this ladder targeted a fixed **−100 ELO** per rung.
> That was **arbitrary** — the Human-SL profiles *are* KGS ranks (see below), so the correct
> spacing is **1 rank = 1 stone**, which for these bots is worth ~150–200 ELO, not 100. The
> configs and method here use the rank/handicap calibration. See *Why rank-spacing, not ELO*.

## Why rank-spacing, not ELO

The Human-SL net is **conditioned on KGS rank**. In `cpp/neuralnet/sgfmetadata.cpp`,
`makeBasicRankProfile` sets `source = SOURCE_KGS` with the comment *"KGS rating system is pretty
reasonable, so let's use KGS as the source,"* and the rank→index map is:

```
9d→1  8d→2  7d→3  6d→4  5d→5  4d→6  3d→7  2d→8  1d→9
1k→10 2k→11 … 5k→14 … 10k→19 … 20k→29
```

So `preaz_9d` and `preaz_8d` are **exactly 1 KGS rank apart** by construction (`preaz_` = the
pre-AlphaZero era, game date 2016-09). The right thing to calibrate is the **1-rank (1-stone)
gap**, not an ELO number.

**The KGS 1-rank rule (= 1 stone):** a 1-rank difference is *not* a placed stone — it is an even
game where the **stronger player (White) gets no komi compensation** (~0.5 instead of the
territory even-komi 6.5). The
weaker player keeps the first-move advantage (~7 points ≈ the value of the first handicap stone).
So two configs are exactly 1 rank apart when:

> **weaker rank as Black** vs **stronger rank as White**, **komi 0.5** → **even game (50%)**.

This is the calibration target for every rung below 9d. It is also **far more numerically stable**
than the old even-game ELO target: a 50% winrate is bounded and well-conditioned, whereas an
even-game ELO target can sit on a steep, unpinnable cliff (the abandoned 7d even-game attempt
burned ~640 games and never converged).

## The 9d anchor (special case)

9d is the **top** of the ladder and has no "rank above" it, so it is anchored differently: the
**`preaz_9d`** (pre-AZ 9d) candidate is calibrated to **even (0 ELO) parity** against the modern
**`rank_9d`** reference — a *different* Human-SL profile — both at 400 visits, in a normal even
game (komi 6.5, territory). The `rank_9d` reference is built from the repo's `gtp_human9d_search_example.cfg`
template by changing `humanSLProfile = rank_9d` (that example ships `preaz_9d`, a different
profile). Result: `gtp_human9d.cfg` = `preaz_9d @ 400v, λ0.045`.

## File naming

One config per rank: **`gtp_human<rank>.cfg`** — e.g. `gtp_human9d.cfg`, `gtp_human8d.cfg`, …,
`gtp_human20k.cfg`. The upstream examples `gtp_human5k_example.cfg` and
`gtp_human9d_search_example.cfg` are left untouched (the latter is used only to seed the
`rank_9d` reference for the 9d anchor).

## Method

Configs are produced by the **`katago tunehuman`** subcommand, which plays in-process
candidate-vs-baseline games and tunes `humanSLChosenMovePiklLambda` (the strength dial) to hit a
target winrate.

### Ruleset — matches the deployed configs

The tuning games are played under the **same ruleset as the deployed config** being calibrated
against: `tunehuman` reads the baseline config's `rules =` line and scores the in-process games with
it (falling back to Japanese). All `gtp_human<rank>.cfg` declare `rules = japanese`, so calibration
runs under **Japanese / territory** scoring — exactly how the bots are scored in real play, and the
ruleset the Human-SL net's KGS-rank conditioning was learned from. (Earlier results were measured
under Chinese / area scoring and are being re-measured under Japanese.) Because the even-game komi
differs by ruleset, the 9d anchor uses the territory-fair **komi 6.5** (not 7.5); the 1-rank handicap
stays **komi 0.5** under either ruleset.

### The lever — `humanSLChosenMovePiklLambda`

Strength is controlled almost entirely by **λ = `humanSLChosenMovePiklLambda`** at a fixed
`maxVisits`:

- **high λ** → closer to raw human policy → **weaker** / more human;
- **low λ** → trusts KataGo's search more → **stronger**.

`maxVisits` is a *weak* lever near the top of the ladder (the strong anchor saturates it), so all
the upper-dan rungs run at the anchor's **400 visits** and differ only in λ. (`tunehuman` also has
a 3-segment "strength dial" over `x∈[0,3]` — temperature at x<1, λ over [1,2], visits over [2,3] —
but for this ladder we pin visits and sweep λ directly via a fixed-λ grind.)

### Handicap calibration (rungs below 9d)

For each rung, tune λ so the **weaker rank (Black) vs the prior rung (White), komi 0.5** is **50%**:

```
-target-elo 0        # 0 ELO offset == 50% winrate target
-komi 0.5            # KGS 1-rank handicap: White (stronger) gets no compensation
-cand-color black    # weaker candidate always plays Black (the handicap is color-bound)
```

`-komi` and `-cand-color` were added to `tunehuman` for this; the harness pins `komiStdev=0` and
`komiAllowIntegerProb=0` so komi is applied **exactly**. With these flags, color is **not**
alternated (the handicap asymmetry is the point), so the measured winrate is the candidate's
winrate as Black-with-the-handicap.

The 9d **anchor** instead uses even games (`-komi 6.5`, alternating colors) targeting 50% parity
vs `rank_9d`.

### Pinning a value (avoid small-sample noise)

For these saturating bots the λ→winrate curve is **steep**, so small samples mislead badly: 20-game
reads can swing ±20% and even flip the apparent λ-ordering. Grind each λ to **~80–200 games** before
trusting it. Pin the crossing with a **direct fixed-λ grind** (set `maxVisits` = the rung's visits
on both dial ends and accumulate games), reading the raw winrate — this avoids a logistic fit that
is biased by the winrate ceiling.

### Resumable checkpointing

Each round's `(x, wins, games)` is appended to a `-resume-file` (default `<output-config>.samples`)
with a config-signature header. Re-running the same command reloads the samples and continues, so a
run interrupted by the environment's process-runtime cap resumes from its last completed round.
Partial/corrupt final lines from a hard kill are skipped; a signature mismatch fails loud. The
helper scripts `tune_lambda.sh` (λ sweep) and `tune_maxvisits.sh` (fixed-λ / visit sweep) wrap this
with a per-chunk `timeout` + a winrate/CI readout, and accept `KOMI=` / `CAND_COLOR=` env vars.

### Nets used

- **Main net (`-model`)**: `lionffen_b24c64_3x3_v3_12300.bin.gz` (b24c64).
- **Human-SL net (`-human-model`)**: `b18c384nbt-humanv0.bin.gz`.
- **Profile**: each config sets `humanSLProfile = preaz_<rank>` (pre-AlphaZero KGS-rank profiles).

## Reproduction

**9d anchor → even-game parity vs the modern `rank_9d` reference:**

```bash
# build the rank_9d (modern 9d) reference from the preaz_9d example template
sed 's/^humanSLProfile = preaz_9d/humanSLProfile = rank_9d/' \
    configs/gtp_human9d_search_example.cfg > baseline_rank9d_400.cfg

# preaz_9d @ 400v, fixed-λ grind to 50% parity (even game)
katago tunehuman \
  -model lionffen_b24c64_3x3_v3_12300.bin.gz \
  -human-model b18c384nbt-humanv0.bin.gz \
  -baseline-config baseline_rank9d_400.cfg \
  -profile preaz_9d -target-elo 0 -elo-tol 8 \
  -search-visits 400 -max-visits-cap 400 -pikl-floor 0.045 -x-lo 2.0 -x-hi 3.0 \
  -komi 6.5 -cand-color auto \
  -games-per-round 10 -num-game-threads 10 \
  -resume-file gtp_human9d.samples -output-config gtp_human9d.cfg
```

**Each weaker rank → 1 KGS rank below the prior rung, via the komi-0.5 handicap.** Example, 8d
(candidate `preaz_8d` as Black vs the tuned `gtp_human9d.cfg` as White), tuning λ to 50%:

```bash
# fixed-λ grind at the chosen λ (here 0.0865), handicap match, target 50%
katago tunehuman \
  -model lionffen_b24c64_3x3_v3_12300.bin.gz \
  -human-model b18c384nbt-humanv0.bin.gz \
  -baseline-config gtp_human9d.cfg \
  -profile preaz_8d -target-elo 0 -elo-tol 8 \
  -search-visits 400 -max-visits-cap 400 -pikl-floor 0.0865 -x-lo 2.0 -x-hi 3.0 \
  -komi 0.5 -cand-color black \
  -games-per-round 5 -num-game-threads 5 \
  -resume-file gtp_human8d.samples -output-config gtp_human8d.cfg
```

To find the right λ first, sweep a bracket (e.g. `pikl-floor 0.02 pikl-max 0.10`, `-x-lo 1 -x-hi 2`)
to locate the ~50% crossing, then fixed-λ-grind that value to ~80+ games. Re-run any command to
resume from its checkpoint. The next rung (7d) chains off `gtp_human8d.cfg` the same way.

## Results

Winrates are direct candidate-vs-baseline results with a **95%** Wilson-score CI. The 9d anchor
target is even-game 50% parity; every rung below is the komi-0.5 handicap 50% (= exactly 1 rank).

> **Tuning (Japanese rules, automated) — COMPLETE.** All **28 rungs** below the 9d anchor
> (8d…1d, 1k…20k) are tuned and locked; with the anchor that is the full **9d→20k** ladder (29 configs).
> Rungs are tuned by a sequential root-finder (`tune_decide.py` + `ladder_step.sh`): for each rank it
> pools all (λ, winrate, games) data, weighted-isotonic-fits the 50% crossing, grinds at that λ until
> the **95% CI ⊂ [40%, 60%]**, writes `gtp_human<rank>.cfg`, builds the next baseline, and chains to the
> next rung — resuming per-round to survive the environment's process-kill cap. Every rung landed at
> **46–52%** with its 95% CI inside [40, 60]. Backend: MLX (Apple-Silicon GPU+ANE); tuned λ are
> backend-independent. (Earlier area-scoring numbers were superseded by Japanese.)

### Findings

- **λ progression** (strength dial vs rank): rises smoothly through the dan rungs (9d **0.045** →
  1d **0.509**), is noisy-but-roughly-flat through the mid-kyu (1k–6k ≈ **0.47–0.51**), then **climbs
  steeply** through the deep kyu: 7k 0.534, 10k 0.590, 14k 0.616, 17k 0.741, 18k 0.782, 19k 0.898,
  **20k 1.223**. λ is *not* globally monotone (each rung is calibrated independently to its own
  baseline), but the deep-kyu trend is a clear, accelerating rise.
- **Deep-kyu rungs (7k+) are "flat-strong plateau → steep cliff."** Their winrate sits well above 50%
  across a wide λ band, then drops through 50% over a narrow λ window. These rungs cost the most games
  (≈ 500–1000 each) and needed the root-finder's noise handling (concentrate-near-50% vs grind-the-
  crossing, drop misleading snaps, a total-games safety cap, and occasional manual concentration of the
  best-sampled point).
- **20k needs near-pure-human play (λ ≈ 1.22).** The human-SL net's `preaz_19k` and `preaz_20k`
  profiles are **less than 1 KGS rank apart** (rank conditioning compresses at the weakest end), so the
  komi-0.5 handicap nearly outweighs the tiny profile gap; only at λ>1 (almost no search) does
  preaz_20k+handicap come down to even vs the tuned 19k. It still pins cleanly at λ1.2227 = 50.0%.
- **No rung hit stop-condition #2.** Every rank reached a 95% CI ⊂ [40, 60]; none was left as a
  best-effort. The reproduction below shows the per-rung command; the table lists every locked value.

| Config | Profile | Baseline (White) | Spacing target | Measured | maxVisits | piklLambda |
|--------|---------|------------------|----------------|----------|----------:|-----------:|
| `gtp_human9d.cfg` | preaz_9d | rank_9d @ 400v λ0.08 | even-game parity (50%) | 49.0% = −7 ELO [39%, 59%], 100 g (Japanese) ✅ | 400 | **0.045** |
| `gtp_human8d.cfg` | preaz_8d | gtp_human9d.cfg | 1 KGS rank (komi-0.5 = 50%) | 47.0% [40.5%, 53.6%], 219 g ✅ | 400 | **0.0868** |
| `gtp_human7d.cfg` | preaz_7d | gtp_human8d.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.6% [40.6%, 56.7%], 144 g ✅ | 400 | **0.1267** |
| `gtp_human6d.cfg` | preaz_6d | gtp_human7d.cfg | 1 KGS rank (komi-0.5 = 50%) | 52.0% [44.1%, 59.8%], 152 g ✅ | 400 | **0.1983** |
| `gtp_human5d.cfg` | preaz_5d | gtp_human6d.cfg | 1 KGS rank (komi-0.5 = 50%) | 51.2% [43.6%, 58.8%], 164 g ✅ | 400 | **0.28064** |
| `gtp_human4d.cfg` | preaz_4d | gtp_human5d.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [43.9%, 56.1%], 256 g ✅ | 400 | **0.373** |
| `gtp_human3d.cfg` | preaz_3d | gtp_human4d.cfg | 1 KGS rank (komi-0.5 = 50%) | 51.5% [43.1%, 59.7%], 136 g ✅ | 400 | **0.45556** |
| `gtp_human2d.cfg` | preaz_2d | gtp_human3d.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [41.9%, 58.1%], 144 g ✅ | 400 | **0.51330** |
| `gtp_human1d.cfg` | preaz_1d | gtp_human2d.cfg | 1 KGS rank (komi-0.5 = 50%) | 49.1% [42.5%, 55.7%], 216 g ✅ | 400 | **0.50930** |
| `gtp_human1k.cfg` | preaz_1k | gtp_human1d.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.7% [42.5%, 58.9%], 140 g ✅ | 400 | **0.48988** |
| `gtp_human2k.cfg` | preaz_2k | gtp_human1k.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.2% [40.8%, 55.7%], 168 g ✅ | 400 | **0.46755** |
| `gtp_human3k.cfg` | preaz_3k | gtp_human2k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [41.5%, 58.5%], 128 g ✅ | 400 | **0.49173** |
| `gtp_human4k.cfg` | preaz_4k | gtp_human3k.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.1% [40.5%, 55.8%], 160 g ✅ | 400 | **0.47130** |
| `gtp_human5k.cfg` | preaz_5k | gtp_human4k.cfg | 1 KGS rank (komi-0.5 = 50%) | 51.2% [43.6%, 58.9%], 160 g ✅ | 400 | **0.50720** |
| `gtp_human6k.cfg` | preaz_6k | gtp_human5k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.8% [42.0%, 59.6%], 120 g ✅ | 400 | **0.48925** |
| `gtp_human7k.cfg` | preaz_7k | gtp_human6k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.9% [41.8%, 60.0%], 112 g ✅ | 400 | **0.53370** |
| `gtp_human8k.cfg` | preaz_8k | gtp_human7k.cfg | 1 KGS rank (komi-0.5 = 50%) | 49.1% [40.2%, 58.1%], 116 g ✅ | 400 | **0.50640** |
| `gtp_human9k.cfg` | preaz_9k | gtp_human8k.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.0% [41.3%, 54.9%], 204 g ✅ | 400 | **0.53880** |
| `gtp_human10k.cfg` | preaz_10k | gtp_human9k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [42.0%, 58.0%], 148 g ✅ | 400 | **0.59036** |
| `gtp_human11k.cfg` | preaz_11k | gtp_human10k.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.1% [40.5%, 55.8%], 160 g ✅ | 400 | **0.56458** |
| `gtp_human12k.cfg` | preaz_12k | gtp_human11k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.8% [42.2%, 59.3%], 128 g ✅ | 400 | **0.54297** |
| `gtp_human13k.cfg` | preaz_13k | gtp_human12k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.8% [42.1%, 59.4%], 124 g ✅ | 400 | **0.58977** |
| `gtp_human14k.cfg` | preaz_14k | gtp_human13k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [41.3%, 58.7%], 124 g ✅ | 400 | **0.61625** |
| `gtp_human15k.cfg` | preaz_15k | gtp_human14k.cfg | 1 KGS rank (komi-0.5 = 50%) | 49.1% [40.2%, 58.1%], 116 g ✅ | 400 | **0.61839** |
| `gtp_human16k.cfg` | preaz_16k | gtp_human15k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [42.1%, 57.9%], 152 g ✅ | 400 | **0.67050** |
| `gtp_human17k.cfg` | preaz_17k | gtp_human16k.cfg | 1 KGS rank (komi-0.5 = 50%) | 48.3% [40.9%, 55.7%], 172 g ✅ | 400 | **0.74130** |
| `gtp_human18k.cfg` | preaz_18k | gtp_human17k.cfg | 1 KGS rank (komi-0.5 = 50%) | 46.3% [40.4%, 52.2%], 268 g ✅ (steep-λ-cliff rung) | 400 | **0.78210** |
| `gtp_human19k.cfg` | preaz_19k | gtp_human18k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [41.0%, 59.0%], 116 g ✅ | 400 | **0.89820** |
| `gtp_human20k.cfg` | preaz_20k | gtp_human19k.cfg | 1 KGS rank (komi-0.5 = 50%) | 50.0% [40.6%, 59.4%], 104 g ✅ | 400 | **1.22270** |

_(Remaining ranks 7d…20k to be appended as tuned, each 1 KGS rank below the prior via the handicap method.)_

### 8d rung — what was measured (2026-06)

- **8d = 1 KGS rank below 9d at λ0.0865.** `preaz_8d@400v λ0.0865` (Black) vs `gtp_human9d.cfg`
  (White, komi 0.5) = **60/120 = 50.0%, 95% CI [41.2%, 58.8%]** (⊂ [40%,60%]) → exactly 1 rank confirmed.
- **preaz_8d needs the SAME 400 visits as the 9d anchor** — at 200v it is far too weak (visits are
  a weak lever near the strong anchor). The rung is reached by **raising λ** (9d's 0.045 → 0.0865,
  i.e. more human / weaker move-selection).
- **1 stone ≈ 150–200 ELO here, not 100.** The KGS-correct 8d (λ0.0865) is meaningfully weaker than
  the earlier −100-ELO attempt (λ0.0575, now superseded), because the komi-0.5 handicap is a large
  advantage for these bots.
- **The handicap (50%-winrate) calibration is well-behaved** — bounded target, gentle slope near
  50% — so it pins cleanly, unlike the even-game ELO target which hit an unpinnable λ-cliff at 7d.

### 9d anchor — what was measured (2026-06)

- **piklLambda is the dominant strength lever; visits are nearly inert at high λ.** Versus the strong
  `rank_9d@400v` reference, raising the candidate's visits at λ0.08 barely moved the result (it loses
  ~−190 ELO at 400v and never reaches parity within [400, 1600] visits) — at high λ the extra search
  is spent exploring human-policy (weaker) moves. **Lowering λ is what reaches parity.**
- **Parity sits at λ ≈ 0.045** (not 0.08): `preaz_9d@400v λ0.045` = **201/383 = 52.5% = +17 ELO,
  95% CI [−18, +52]** vs `rank_9d@400v λ0.08` — statistically at parity, stable across the whole
  94→383-game grind (the point estimate never left [−7, +23] ELO).
- **The λ→ELO response is shallow near parity (~23 ELO per 0.01 λ)** because preaz_9d's winrate
  *saturates* (ceiling ~67%, not →100%) against the strong reference — `humanSLRootExploreProb
  = 0.8` caps how much search can sharpen play. So a logistic auto-fit is biased high near parity;
  the anchor was pinned with a direct fixed-λ grind, and λ ∈ [0.04, 0.05] all sit within ~±25 ELO of
  parity (finer precision than ±0.01 λ is unnecessary, and would cost ~30 h of 400v games).

### Cost & practical notes

- A rung confirmed to ~±10% winrate (≈1 rank) takes ~80–200 games at 400v (~hours at half
  resources). The full 20k→9d ladder is a large compute project; lower (weaker) ranks may chain off
  cheaper, lower-visit baselines once away from the strong 9d/8d anchors.
- Run **one** GPU job at a time (~5 game-threads) — concurrent `katago` processes trigger
  memory-pressure (jetsam) kills. Keep run artifacts in a persistent dir, not `/tmp`.
- Always report a tuned winrate/ELO **with its 95% CI and sample count** — small (5–30 game)
  samples are deceptive on these steep curves.

---
_Generated by the `tunehuman` workflow. Configs and this doc are local artifacts (not the
upstream KataGo examples)._
