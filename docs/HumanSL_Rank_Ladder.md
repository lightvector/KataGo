# Human-SL Even-Game ELO Ladder

A set of GTP configs that make KataGo (with the Human-SL net) play at a chosen amateur strength,
from **8d (top anchor)** down to **25k**. The ladder is anchored at 8d and tuned in **two regimes**:

- **7d → 14k — a fixed 100-ELO staircase.** Each consecutive rung is tuned (via
  `humanSLChosenMovePiklLambda`) so that, in a **normal even game** (komi 6.5, colours alternated),
  its stronger neighbour beats it by **100 ELO**, certified to a **95% CI within ±30** ([70, 130]) —
  with two honestly-documented exceptions (4d +76, 3d +112; see the Status note below).
- **15k → 25k — a pure-human tail.** At this depth the even-game gap between *adjacent* Human-SL
  ranks is **non-monotonic in λ and peaks below 100 ELO** — adjacent deep-kyu profiles are
  near-tied, so a full 100-ELO step is **not reachable** by the λ lever. These rungs therefore ship
  at **`humanSLChosenMovePiklLambda = 1e8` (pure-human imitation)** and their **natural** even-game
  gap vs the stronger neighbour is *measured and documented* (to a 95% CI half-width ≤30, before
  integer rounding of printed endpoints), not forced to 100. See the [deep-kyu finding](#findings).

> **Status (2026-08-06): both regimes tuned.** **7d→14k are all locked** — 21 rungs: 17 certified at
> +100, plus the four honestly re-measured dan rungs below. **15k→25k
> are all shipped at λ=1e8** (11 of 11 pure-human rungs), each with its natural even-game gap measured
> to a 95% CI half-width ≤30 (before integer rounding of the table endpoints):
> **+29, +24, +44, +70, +32, +8, +23, +29, −1, +55, +86** (15k→25k) —
> non-monotonic and all **below +100** (most well below it), consistent with the deep-kyu finding.
> Four dan rungs
> (**6d, 5d, 4d, 3d**) that had locked early via a since-removed optimistic pooled estimator were
> **re-measured honestly** at their exact shipped λ (φ=1 — 6d/5d on the single concentrated cell,
> 4d/3d pooling the two nearest-λ cells — to a 95% CI half-width ≤~30) —
> **no λ re-tuning** (out of scope): **6d +100 [71,130], 5d +91 [70,111], 4d +76 [57,96], 3d +112
> [89,134]**. 6d/5d land in-band;
> the honest 4d (**+76**) and 3d (**+112**) gaps sit just outside [70,130], revealing that the original
> pooled estimator set those two λ imprecisely — 4d is under-spaced (~+76 below 5d, not +100) and 3d is
> over-spaced (~+112). These are **documented as measured, not corrected — accepted as the final result**
> (project decision 2026-08-06: measure honestly, do not λ-re-tune, and accept the honest 4d/3d gaps
> rather than force them into [70,130]). See [Results](#results) and the
> [deep-kyu finding](#findings).

## What "the gap" means here

Each rung is calibrated on a **normal even game** — territory-fair **komi 6.5**, colours alternated
— against its already-tuned stronger neighbour. Two configs are exactly one rung apart when:

> **weaker rank** vs **stronger rank**, **even game (komi 6.5, alternating colours)** → the stronger
> side wins **64.0%**, i.e. the weaker side wins **36.0%** = `1/(1+10^(100/400))` (a **100-ELO** gap).

In the staircase regime (7d→14k) the gap is held to a **95% CI within [70, 130] ELO** (i.e. 100 ± 30)
— a *directly measured*, per-adjacent-pair precision target — except the honestly re-measured 4d/3d
(documented in Status/Results); the deep-kyu tail (15k→25k) is *measured to ±30 precision* rather than
held to a target. (Because the ladder is a chain anchored at 8d, the *cumulative*
ELO-vs-8d of a far rung compounds across hops and has a wider CI; the ±30 applies to each **adjacent**
gap, which is what is calibrated.)

### Uniform 100 ELO/rung (goal history)

The gap target was set to **100 ELO for every rung** on 2026-07-17. An earlier version of this
re-tune used 100 ELO for the dan rungs and 50 ELO for the kyu rungs; that split was dropped in favour
of a **uniform 100** target, because the low-dan λ values come out small (strong play) and a moderate
100/rung let the run observe how λ climbs into the kyu range. The anticipated deep-kyu saturation then
occurred: from 15k down, λ ran to pure-human policy without reaching +100, and the uniform target gave
way to the two-regime outcome above (100-ELO staircase 7d→14k; measured pure-human tail 15k→25k).

An even-earlier calibration (preserved in the author's fork's git history) spaced the ladder by **1 KGS rank** using a
**komi-0.5 handicap game tuned to 50% winrate**; a subsequent even-game evaluation showed those
rank-spaced rungs had **highly variable** even-game gaps (dan steps +119/+127/+128, kyu steps
scattered, some tied/inverted). **This re-tune replaces that** with fixed even-game ELO spacing.

## Anchors and scope

- **8d anchor** (hand-set): `preaz_8d @ maxVisits 40, humanSLChosenMovePiklLambda 0.06,
  winLossUtilityFactor 0`. A deliberately weak/fast top rung; every rung 7d→25k calibrates
  down the chain from it.
- **9d (legacy)** is a separate, stronger 400-visit reference (`preaz_9d @ 400v, λ0.045,
  winLossUtilityFactor 1.0`) and is **not** part of this 40-visit ladder — it is not shipped with
  it (the legacy config remains in the author's fork).
- **Ladder stops at 25k** — the Human-SL net's rank input encoding saturates at inverse-rank 34 = 25k
  (`RANK_LEN_PER_PLA = 34` in `cpp/neuralnet/sgfmetadata.cpp`), so ranks weaker than 25k encode
  identically to 25k. Profile names below 20k (`preaz_21k`…`preaz_25k`) are parsed by the small
  `SGFMetadata::getProfile` extension shipped alongside these configs.

## File naming

One config per rank: **`gtp_human<rank>.cfg`** — `gtp_human8d.cfg` … `gtp_human25k.cfg`. The upstream
examples `gtp_human5k_example.cfg` and `gtp_human9d_search_example.cfg` are left untouched.

## Method

### Nets (calibration is net-specific)

- **Main net (`-model`)**: **`kata1-b28c512nbt-s8326494464-d4628051565.bin.gz`** (b28c512nbt, the
  strong modern net). At low λ this strong main net **overrides** the human profile so search
  dominates and the dan rungs separate cleanly. **A different main net invalidates the calibration.**
- **Human-SL net (`-human-model`)**: `b18c384nbt-humanv0.bin.gz`. Supplies the human-imitation policy
  blended in by `humanSLChosenMovePiklLambda`. It is evaluated on essentially every move
  (`humanSLRootExploreProbWeightless = 0.8`), so it is the **throughput bottleneck** despite being
  the smaller file.
- **Profile**: each config sets `humanSLProfile = preaz_<rank>` (the pre-AlphaZero KGS-rank profile).

### Fixed, even-game settings (must match between tuning and deployment)

- **`maxVisits = 40`** — fast; the human-SL policy dominates move selection at these λ.
- **`winLossUtilityFactor = 0.0`** — pure score-margin utility, which keeps the winrate response to λ
  smooth (and, through the dan/low-kyu regime, monotone; the deep-kyu regime is non-monotonic
  regardless — see [Findings](#findings)). (Accepted trade-off: score-maximizing endgame may grind
  points after the game is decided; resignation is unaffected — it uses raw win-probability.)
- **komi 6.5** (territory-fair even komi; `komiStdev = 0`, `komiAllowIntegerProb = 0` so it is exact),
  **colours alternated** to remove first-move bias. Rules = **Japanese** (SIMPLE/TERRITORY/SEKI),
  matching real play and the ruleset the net's KGS-rank conditioning was learned from.

### The lever — `humanSLChosenMovePiklLambda`

- **low λ** → trusts KataGo's search → **stronger**;
- **high λ** → closer to raw human policy → **weaker**.

`maxVisits` is a weak lever (the human net is evaluated every move regardless), so all rungs run at a
fixed 40 visits and differ only in λ.

### The engine — `katago tunehuman` (calibration harness, fork-only)

The `tunehuman` subcommand is maintained in the author's fork
([ChinChangYang/KataGo](https://github.com/ChinChangYang/KataGo), branch `tunehuman-mlx`) and is not
part of upstream KataGo — the shipped configs are plain GTP configs and need no engine change beyond
the small 21k–25k profile-name parsing extension shipped with them. It
plays in-process candidate-vs-baseline games and supports every
even-game knob: **`-komi 6.5 -cand-color auto`** (auto alternates colours → unbiased even-game
winrate) and **`-target-elo -100`** (maps to the 36.0% candidate-winrate target via
`winrate = 1/(1+10^(-elo/400))`). It checkpoints each round's `(x, wins, games)` to a `-resume-file`
so a run survives the environment's process-kill cap and resumes. Games hitting the move cap
(undetermined) are **discarded**, not scored 0.5.

### The calibrator — `tune_elo.py`

A pure-Python decision brain that reads the accumulated per-λ `(wins, games)` samples for a rung and
decides **GRIND** (which λ to sample next) / **LOCK** / **STOP**. Key design points:

- **Locate the crossing.** Fit a Bayesian monotone-logistic curve of winrate vs `ln(λ)` (reused from
  `tune_fit.py`) to estimate λ\* where the candidate winrate = 36.0% (a 100-ELO gap). The crossing is
  refined by:
  - **local bracket** (interpolating the two well-sampled, ≥120-game cells straddling the target) for
    clean, steep rungs — most precise;
  - **coarse-binned crossing** (pooling λ into ~7% log-bins that average out per-cell noise) when the
    rung is **overdispersed** (φ > 1.3) **or has a poorly-determined, wide crossing CI**
    (`width_ratio > 2.5`) — the signature of a **flat/noisy** rung, where per-cell reads scatter and
    would otherwise make the concentration wander.
- **Concentrate at the crossing, and stay there (sticky).** Games pile on a stable 1% relative-λ grid
  cell at the crossing. On a flat/noisy rung the crossing *estimate* wobbles chunk-to-chunk, so the
  chosen cell is **persisted** (`elo<rank>_conc.txt`) and re-used every chunk; it is only re-picked when
  that cell is well-sampled yet its point gap is clearly off-target (crossing mislocated). Without this,
  a gently-sloped rung's concentration wanders across dozens of λ and never accumulates the games one
  cell needs to lock — the deep-dan failure mode (2d originally needed four manual fixed-λ passes).
- **LOCK = a single concentrated cell's binomial (φ=1) Wilson gap-CI ⊂ [70, 130] at the shipped λ.**
  Within a *fixed* λ the games are i.i.d. Bernoulli, so a concentrated cell's Wilson interval is exact
  at φ=1; the cross-cell overdispersion (φ>1) is logistic *misfit*, not within-cell noise, and must not
  inflate a single-cell lock. (Empirically ~530–560 games at φ=1 for a centred rung; a flat or
  off-centre rung needs more, ~1000–2500.) **This single-cell gate is the *sole* lock criterion** — an
  earlier φ-inflated *pooled-window* fallback was removed (2026-07-18) after an audit found it locked
  the steeper dan rungs optimistically: it pooled a λ-gradient into a biased, too-tight CI (see
  Findings). Sticky makes a fallback unnecessary — games always reach the honest single-cell count at
  the shipped λ.
- **Slope prior.** The logistic-slope prior is seeded from the slope already measured on the last few
  locked rungs (same nets/40v → similar slope), reducing cold-start exploration; a moderate prior SD
  lets each rung's own data dominate (verified not to shift any locked crossing).

### Driver, throughput, and housekeeping

- `elo_ladder_step.sh` runs one resumable chunk and, on LOCK, writes `gtp_human<rank>.cfg` (forcing
  `maxVisits=40`, `winLossUtilityFactor=0`, the tuned λ, and a fresh even-game-calibrated header),
  rebuilds the next rung's ANE-mux tuning baseline, and advances. `elo_ladder_loop.sh` self-continues
  across rung advances and is resumable across the ~25–45 min process-kill cap.
- **Throughput/housekeeping (the author's fork calibration setup, Apple Silicon — none of this is
  needed to *use* the configs):** `GAMES_PER_ROUND = 8` (amortizes the per-round barrier so the
  GPU+ANE mux stays filled); the tuning baseline uses the GPU+ANE mux (`numNNServerThreadsPerModel =
  2`, `deviceToUseThread0 = 0`, `deviceToUseThread1 = 100`) with 4 game-threads × 8 search-threads.
  The ANE compiled-bundle cache (`~/Library/Caches/katago/com.apple.e5rt.e5bundlecache`) is pruned
  between chunks (each chunk is a cache-miss that would otherwise accumulate ~10 GB/day). Backend:
  the fork's MLX backend (Apple-Silicon GPU + ANE, not part of upstream KataGo); tuned λ are
  **backend-independent**.

## Results

Even-game gaps are direct candidate-vs-baseline results (weaker rank vs its stronger neighbour, komi
6.5, alternating colours, Japanese, b28c512 main net, 40v, winLossUtilityFactor 0) with a 95% CI.
Certified rungs sit at **100 ELO ± (95% CI ⊂ [70, 130])**; the honestly re-measured **4d** and **3d**
are the two documented exceptions (see the note below the table).

| Config | Profile | Baseline (stronger) | Even-game gap (95% CI) | Games | maxVisits | piklLambda |
|--------|---------|---------------------|------------------------|------:|----------:|-----------:|
| `gtp_human8d.cfg` | preaz_8d | — (hand-set anchor) | anchor, not calibrated | — | 40 | **0.06** |
| `gtp_human7d.cfg` | preaz_7d | gtp_human8d.cfg | **+100** [72, 128] ✅ certified | 644 | 40 | **0.07760** |
| `gtp_human6d.cfg` | preaz_6d | gtp_human7d.cfg | **+100** [71, 130] ✅ measured in-band | 576 | 40 | 0.09940 |
| `gtp_human5d.cfg` | preaz_5d | gtp_human6d.cfg | **+91** [70, 111] ✅ measured in-band | 1208 | 40 | 0.13240 |
| `gtp_human4d.cfg` | preaz_4d | gtp_human5d.cfg | **+76** [57, 96] ⚠ measured (below +100) | 1256 | 40 | 0.15750 |
| `gtp_human3d.cfg` | preaz_3d | gtp_human4d.cfg | **+112** [89, 134] ⚠ measured (above +100) | 1039 | 40 | 0.18960 |
| `gtp_human2d.cfg` | preaz_2d | gtp_human3d.cfg | **+101** [74, 127] ✅ certified | 719 | 40 | **0.21300** |
| `gtp_human1d.cfg` | preaz_1d | gtp_human2d.cfg | **+106** [85, 128] ✅ certified | 1087 | 40 | **0.19170** |
| `gtp_human1k.cfg` | preaz_1k | gtp_human1d.cfg | **+105** [82, 129] ✅ certified | 896 | 40 | **0.20150** |
| `gtp_human2k.cfg` | preaz_2k | gtp_human1k.cfg | **+96** [71, 121] ✅ certified | 800 | 40 | **0.19950** |
| `gtp_human3k.cfg` | preaz_3k | gtp_human2k.cfg | **+100** [72, 127] ✅ certified | 664 | 40 | **0.20760** |
| `gtp_human4k.cfg` | preaz_4k | gtp_human3k.cfg | **+93** [71, 114] ✅ certified | 1088 | 40 | **0.21180** |
| `gtp_human5k.cfg` | preaz_5k | gtp_human4k.cfg | **+108** [86, 129] ✅ certified | 1104 | 40 | **0.21600** |
| `gtp_human6k.cfg` | preaz_6k | gtp_human5k.cfg | **+93** [71, 115] ✅ certified | 1000 | 40 | **0.22480** |
| `gtp_human7k.cfg` | preaz_7k | gtp_human6k.cfg | **+100** [71, 129] ✅ certified | 608 | 40 | **0.24590** |
| `gtp_human8k.cfg` | preaz_8k | gtp_human7k.cfg | **+102** [75, 129] ✅ certified | 688 | 40 | **0.25840** |
| `gtp_human9k.cfg` | preaz_9k | gtp_human8k.cfg | **+107** [85, 129] ✅ certified | 1032 | 40 | **0.30620** |
| `gtp_human10k.cfg` | preaz_10k | gtp_human9k.cfg | **+101** [74, 128] ✅ certified | 688 | 40 | **0.37250** |
| `gtp_human11k.cfg` | preaz_11k | gtp_human10k.cfg | **+103** [76, 129] ✅ certified | 704 | 40 | **0.40810** |
| `gtp_human12k.cfg` | preaz_12k | gtp_human11k.cfg | **+100** [71, 129] ✅ certified | 608 | 40 | **0.46300** |
| `gtp_human13k.cfg` | preaz_13k | gtp_human12k.cfg | **+104** [79, 129] ✅ certified | 800 | 40 | **0.83000** |
| `gtp_human14k.cfg` | preaz_14k | gtp_human13k.cfg | **+100** [71, 129] ✅ certified | 592 | 40 | **3.40040** |

_The deep-kyu tail **15k → 25k** is tuned in the pure-human regime (λ=1e8, gap measured not forced) —
see [Deep-kyu pure-human tail](#deep-kyu-pure-human-tail-15k--25k) below._

> **13k note (deep-kyu compression):** the 12k↔13k even-game gap is a **flat, noisy
> plateau ~+58 ELO for λ ∈ [0.60, 0.80]**, then climbs through a **steep, narrow
> transition across [0.80, 0.85]** (+58 → ~+140) to a **~+140 plateau** for λ ≥ 0.85.
> So 13k required **λ=0.83 — far above its neighbors** (12k=0.463) — and a long
> concentration (~800 games) to certify the +100 gap. Deep-kyu rungs were expected to
> need high λ and extended CERT grinds for this reason — borne out at 14k (λ=3.40), while
> from 15k down even λ→∞ no longer reaches +100 and the ladder switches to the measured
> pure-human tail (below).

> **14k note (high-λ crossing):** the 14k↔13k even-game gap is **flat ~+9 ELO for λ ≲ 1.5**
> (candidate ≈ tied with 13k), then rises steeply, crossing **+100 at λ≈3.4** and plateauing
> ~+170 by very large λ. So 14k required **λ=3.40 — an order of magnitude above the low-kyu
> regime** (adjacent deep-kyu profiles barely separate until strong pikl weighting). The gap
> reads are ordinary binomial noise (φ=1.00, ±56 ELO SE at 40 games), so the crossing was
> located by the **pooled logistic fit** across fixed-λ cells, not per-cell reads; certified at
> **+100 [71, 129] over 592 games**.

The gap/CI shown is the **honest Wilson measurement at the shipped λ** (φ=1; 6d/5d single-cell, 4d/3d
pooled over the two nearest-λ cells). ✅ *measured in-band* = that CI ⊂ [70, 130]. ⚠ *measured
(below/above +100)* = **4d (+76)** and **3d (+112)**, whose honest gaps fall just outside [70, 130].
These four dan rungs had originally locked via a since-removed optimistic pooled estimator; re-measured
honestly they read 6d +100, 5d +91, 4d +76, 3d +112 — i.e. the pooled estimator had set 4d's and 3d's λ
imprecisely. Per the project decision, these are **measured and documented as-is, not λ-re-tuned**
(retuning is out of scope). Games column = games at the shipped-λ cell(s).

### Deep-kyu pure-human tail (15k → 25k)

At this depth adjacent Human-SL ranks are **near-tied**: the even-game gap is **non-monotonic in λ and
peaks below 100 ELO** (see [Findings](#findings)), so a full 100-ELO step is **not reachable** by the λ
lever. These rungs therefore ship at **`humanSLChosenMovePiklLambda = 1e8`** (pure-human imitation) and
their **natural** even-game gap vs the stronger neighbour is **measured** (95% CI half-width ≤30;
the table's endpoints are rounded to integers, so several rows' implied half-width comes out to
30.5), not forced to 100. Same even-game protocol (komi 6.5, alternating colours, Japanese, b28c512 main net, 40v,
winLossUtilityFactor 0).

| Config | Profile | Baseline (stronger) | Even-game gap (95% CI) | Games | maxVisits | piklLambda |
|--------|---------|---------------------|------------------------|------:|----------:|-----------:|
| `gtp_human15k.cfg` | preaz_15k | gtp_human14k.cfg | **+29** [-1, 59] measured | 528 | 40 | **1e8** |
| `gtp_human16k.cfg` | preaz_16k | gtp_human15k.cfg | **+24** [-6, 55] measured | 512 | 40 | **1e8** |
| `gtp_human17k.cfg` | preaz_17k | gtp_human16k.cfg | **+44** [14, 75] measured | 504 | 40 | **1e8** |
| `gtp_human18k.cfg` | preaz_18k | gtp_human17k.cfg | **+70** [40, 101] measured | 520 | 40 | **1e8** |
| `gtp_human19k.cfg` | preaz_19k | gtp_human18k.cfg | **+32** [2, 62] measured | 520 | 40 | **1e8** |
| `gtp_human20k.cfg` | preaz_20k | gtp_human19k.cfg | **+8** [-22, 39] measured | 504 | 40 | **1e8** |
| `gtp_human21k.cfg` | preaz_21k | gtp_human20k.cfg | **+23** [-7, 54] measured | 504 | 40 | **1e8** |
| `gtp_human22k.cfg` | preaz_22k | gtp_human21k.cfg | **+29** [-1, 59] measured | 504 | 40 | **1e8** |
| `gtp_human23k.cfg` | preaz_23k | gtp_human22k.cfg | **−1** [-32, 29] measured | 496 | 40 | **1e8** |
| `gtp_human24k.cfg` | preaz_24k | gtp_human23k.cfg | **+55** [25, 85] measured | 536 | 40 | **1e8** |
| `gtp_human25k.cfg` | preaz_25k | gtp_human24k.cfg | **+86** [55, 116] measured | 528 | 40 | **1e8** |

> The measured deep-kyu gaps are **small** (15k is only ~29 ELO below 14k) — a direct, honest
> consequence of adjacent deep-kyu Human-SL ranks being nearly the same strength. This is documented,
> not a defect: the ladder's *fine-grained* separation lives in the dan/low-kyu region, while the
> deep-kyu rungs mark the correct **rank input** to the net with whatever natural strength gap the
> pure-human policy provides.

<!-- COMPLETE: all 32 rungs 7d→25k tuned. Dan 6d/5d/4d/3d re-measured honestly (measure-only, no λ
     re-tune); deep-kyu 15k→25k all shipped @λ=1e8 with measured gaps. -->

### Findings

- **λ climbs through the dan rungs, then dips at the 2d→1d step:** 8d 0.06 → 7d 0.078 → 6d 0.099 →
  5d 0.132 → 4d 0.158 → 3d 0.190 → 2d 0.213 → **1d 0.192**. It rises monotonically 7d→2d (each weaker
  rung needs more human-imitation), but 1d's λ (0.192) sits *below* 2d's (0.213). That is expected, not
  a regression: λ is **not** directly comparable across rungs because each uses a different (weaker)
  `preaz_<rank>` profile. 1d's profile is already weaker than 2d's, so it needs less λ-blending to land
  100 ELO below 2d. 1d certified honestly through the sticky + single-cell pipeline (+106 [85,128], 1087g).
- **The deep-dan rungs (2d, and likely 1d) are flat and noisy.** As λ enters the mid-dan range the
  even-game gap becomes only weakly sensitive to λ (~0.7 ELO per 1% λ at 2d, vs ~6 for 7d) and the
  per-cell winrate scatters widely (φ-misfit while the fit's φ stays ~1). This makes the crossing hard
  to pin: 2d needed ~2700 games and a manual fixed-λ concentration before the calibrator's
  wide-crossing → coarse-bin logic was added to handle it automatically.
- **Per-rung cost varies** from ~600 games (clean, steep rungs) to ~2500 (flat/noisy). Clean rungs
  lock near the ~530–560-game φ=1 minimum; flat rungs cost more because the CI must be pinned through
  the noise.
- **The pooled-window estimator was optimistic on steep rungs (audit 2026-07-18).** Re-checking every
  locked rung at its *exact shipped λ* with a single-cell φ=1 CI found only **7d and 2d** honestly
  ⊂[70,130]; **6d/5d/4d/3d** had locked via a pooled ±3% λ-window that averaged a gradient into a
  biased, too-tight CI. On a steep dan rung a 3% λ window spans ~15 ELO of gap — enough to pull the
  pooled mean toward the target while the shipped λ's *own* gap sat elsewhere. Fix: **sticky
  concentration** (so one cell reaches the honest game count) + **deletion of the pooled fallback**
  (single-cell φ=1 is now the sole gate). **Honest re-measurement (2026-08-06, measure-only, no λ
  re-tune):** at their shipped λ (4d/3d: pooling the two adjacent 1%-grid cells around it) the four
  rungs measure **6d +100 [71,130], 5d +91 [70,111],
  4d +76 [57,96], 3d +112 [89,134]** (6d/5d single-cell; 4d/3d pooled over the two nearest-λ cells,
  ~1256/1039 games, CI half-width ≤~23). So the pooled estimator was not just optimistically *tight* but
  also *off-centre* on 4d and 3d: **4d is under-spaced (~+76 below 5d, not +100) and 3d over-spaced
  (~+112)**, while 6d/5d land right.
  Because λ re-tuning is out of scope, these are documented as measured. Lesson: measure each rung on
  games at **one** λ — or, when a single cell is short of games, pool only the two adjacent 1%-grid
  cells (as for 4d/3d), never a wide λ-gradient window — and that honest on-λ gap is the number to
  trust.

#### Deep-kyu regime (15k → 25k): the 100-ELO step is not reachable by λ

- **λ climbs steeply through the deep-kyu rungs, then the lever saturates:** 11k 0.408 → 12k 0.463 →
  13k **0.830** → 14k **3.40**. Each deep-kyu rung needs an order-of-magnitude more human-imitation than
  the low-kyu rungs, because adjacent `preaz_<rank>` profiles at this depth are *very close in strength*.
- **The adjacent even-game gap is non-monotonic in λ and peaks below 100 ELO.** Mapping 15k↔14k across
  λ showed the gap **rises to a peak (~+85–98) around λ≈10, then declines** back toward a small value as
  λ→∞ (pure human): probes read λ20→+89, λ44→+17, λ1e8→+17…+29. So beyond the peak, *more* human-imitation
  makes the candidate no weaker — the two ranks are simply near-tied. A **monotone** logistic crossing
  estimator misfits this shape (it extrapolates the +100 crossing to ever-larger λ); a peaked/empirical
  search is required to even find the maximum gap.
- **Consequence (feasibility):** for these rungs the *maximum achievable* adjacent gap appears to be
  **< 100 ELO** (mapped in detail at 15k↔14k; consistent with all 11 measured pure-human gaps < +100),
  so the uniform-100 target is **infeasible via the λ lever alone**. Rather than ship an absurd,
  peak-hunted λ for a gap that is still short of 100, the ladder ships each deep-kyu rung at
  **λ=1e8 (pure human)** and **documents the natural gap** it produces (15k: **+29 [-1, 59]** over 528
  games). A full 100-ELO deep-kyu staircase would need a *different* weakening lever (e.g. fewer visits,
  softmax temperature, or a weaker human-model rank) — noted as possible future work.
- This matches the earlier komi-6.5 evaluation of the old rank-spaced ladder, which already found several
  deep-kyu adjacent pairs **tied or inverted** (e.g. 15k–16k ≈ 0, 9k–10k ≈ −26): adjacent deep-kyu
  Human-SL ranks genuinely encode nearly the same playing strength.

## Reproduction

Deployed run command (per config header):

```bash
./katago gtp -config gtp_human<rank>.cfg \
  -model kata1-b28c512nbt-s8326494464-d4628051565.bin.gz \
  -human-model b18c384nbt-humanv0.bin.gz
```

Tuning one rung (e.g. 7d vs the 8d anchor), driven automatically by the calibrator + loop:

```bash
# One even-game chunk (fixed λ, resumable):
BASELINE_CFG=~/.katago_tune/tunebase_human8d_ane.cfg CAND_PROFILE=preaz_7d PIKL=<λ> V_LO=40 V_HI=40 \
  KOMI=6.5 CAND_COLOR=auto HANDICAP=0 TARGET_ELO=-100 ELO_TOL=1 \
  GAMES_PER_ROUND=8 GAME_THREADS=4 TAG=elo7d_L<λ> RESUME=~/.katago_tune/elo7d_L<λ>.samples \
  bash tune_maxvisits.sh

# The full autonomous chain (self-continues 7d→25k, then auto-runs the certification/top-up phase):
CHUNK_TIMEOUT=1200 MAX_CHUNKS=400 bash ./elo_ladder_loop.sh   # driven by tune_elo.py; on DONE it calls
                                                              # elo_topup.sh until every certifiable rung
                                                              # certifies (re-measured dan rungs and the
                                                              # deep-kyu tail are instead measured to a
                                                              # CI half-width <= 30)

# Certify an under-certified rung standalone (linear grind must be stopped — one katago at a time):
DRY_RUN=1 bash ./elo_topup.sh     # show which rung/λ it would top up
bash ./elo_topup.sh               # play one resumable chunk at that rung's exact shipped λ

# Status table + tuned-λ recap + ETA (GOAL_MET re-verifies each rung honestly):
python3 elo_ladder_report.py
```

The calibrator (`tune_elo.py`, with fit helpers imported from `tune_fit.py`), drivers
(`elo_ladder_step.sh`, `elo_ladder_loop.sh`), fixed-λ chunk runner (`tune_maxvisits.sh` — despite the
name, here it plays even-game chunks at a fixed 40 visits), top-up pass (`elo_topup.sh`), and reporter
(`elo_ladder_report.py`) live in the author's fork alongside `tunehuman`; none of them are part of
upstream KataGo.

---
_Generated by the `tunehuman` + `tune_elo.py` even-game ELO-ladder workflow (calibration harness in
the author's fork). All 32 rungs (7d→25k) are final — 17 certified at +100, four dan rungs honestly
re-measured, 11 deep-kyu rungs measured at λ=1e8; **this document is final.**_
