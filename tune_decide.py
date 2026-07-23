#!/usr/bin/env python3
"""
tune_decide.py — decision brain for the Human-SL rank-ladder λ calibration.

Pools ALL (λ, wins, games) data for one rank (across every λ ever tried), enforces a
monotonic winrate-vs-λ curve (weighted isotonic / PAVA — robust to noise & non-monotone
fluctuations), estimates the 50%-winrate crossing λ*, and decides the next action:

  ACTION=LOCK   LAMBDA=<λ>  WR=<%>  CI=<lo,hi>  N=<games>   # some λ's 95% CI ⊂ [40,60]
  ACTION=GRIND  LAMBDA=<λ>                                  # accumulate games at λ* (grid 1e-4)
  ACTION=STOP   LAMBDA=<λ>  WR=<%>  CI=<lo,hi>  N=<games>   # >MAXGAMES, no lock: best-effort

Winrate DECREASES with λ (higher λ = more human = weaker candidate). Candidate is the
weaker rank as Black with the komi-0.5 handicap; target = 50%.

Usage: tune_decide.py <samples-glob>   e.g.  tune_decide.py '~/.katago_tune/jpn8d_ane_L*.samples'
"""
import sys, glob, os, re, math

CI_LO, CI_HI = 0.40, 0.60   # target band for the 95% CI
GRID = 1e-4                 # λ rounding grid (so games pool at grid points)
MAXGAMES = 500             # per-rank budget before stop-condition #2
Z = 1.96

def wilson(w, n):
    if n == 0: return (0.0, 1.0)
    p = w / n
    den = 1 + Z*Z/n
    cen = (p + Z*Z/(2*n)) / den
    mar = (Z/den) * math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (cen - mar, cen + mar)

def parse(path):
    """Return (lambda, wins, games) for one fixed-λ samples file."""
    lam = None; w = 0; g = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                m = re.search(r'piklFloor=([0-9.]+)', line)
                if m: lam = float(m.group(1))
                continue
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    w += float(parts[1]); g += int(parts[2])
                except ValueError:
                    pass
    return (lam, int(w), g)

def pava_decreasing(pts):
    """Weighted pool-adjacent-violators for a NON-INCREASING fit.
    pts: list of (lambda, wins, games) sorted by lambda ascending.
    Returns list of blocks: (lambda_min, lambda_max, pooled_winrate, games)."""
    blocks = [[lam, lam, w, g] for (lam, w, g) in pts]  # [lmin,lmax,wins,games]
    i = 0
    while i < len(blocks) - 1:
        r_i = blocks[i][2] / blocks[i][3]
        r_j = blocks[i+1][2] / blocks[i+1][3]
        if r_i < r_j:  # violation of non-increasing: pool
            blocks[i][1] = blocks[i+1][1]
            blocks[i][2] += blocks[i+1][2]
            blocks[i][3] += blocks[i+1][3]
            del blocks[i+1]
            if i > 0: i -= 1
        else:
            i += 1
    return [(b[0], b[1], b[2]/b[3], b[3]) for b in blocks]

def main():
    pattern = os.path.expanduser(sys.argv[1])
    files = sorted(glob.glob(pattern))
    # aggregate by λ (multiple files at same λ pool together)
    agg = {}
    for f in files:
        lam, w, g = parse(f)
        if lam is None or g == 0: continue
        a = agg.setdefault(round(lam, 6), [0, 0])
        a[0] += w; a[1] += g
    pts = sorted((lam, w, g) for lam, (w, g) in agg.items())
    total = sum(g for _, _, g in pts)
    if not pts:
        print("ACTION=GRIND LAMBDA=NA NOTE=no-data"); return

    # ---- LOCK check: any λ whose 95% CI ⊂ [40,60]? pick the best-centered ----
    lockable = []
    for lam, w, g in pts:
        lo, hi = wilson(w, g)
        if lo >= CI_LO and hi <= CI_HI:
            lockable.append((abs(w/g - 0.5), lam, w/g, lo, hi, g))
    if lockable:
        lockable.sort()
        _, lam, p, lo, hi, g = lockable[0]
        print(f"ACTION=LOCK LAMBDA={lam:.5f} WR={100*p:.1f} CI={100*lo:.1f},{100*hi:.1f} N={g}")
        return

    # ---- decide next λ ----
    fit = pava_decreasing(pts)
    lams = [lam for lam, _, _ in pts]
    lo_lam, hi_lam = min(lams), max(lams)
    bracketed = fit[0][2] >= 0.5 >= fit[-1][2]
    best = min(pts, key=lambda t: abs(t[1]/t[2]-0.5))   # λ whose winrate is closest to 50%
    blo, bhi = wilson(best[1], best[2])
    bp = best[1]/best[2]

    # stop-condition #2: best λ ground past budget, OR total games across all λ exceeds a hard cap
    # (a genuinely noisy/cliff rung that spreads games without any single λ pinning), OR λ blew up.
    if best[2] > MAXGAMES or total > 1300 or best[0] > 1e8:
        print(f"ACTION=STOP LAMBDA={best[0]:.5f} WR={100*bp:.1f} "
              f"CI={100*blo:.1f},{100*bhi:.1f} N={best[2]} NOTE=best-{best[2]}g-total{total}g-cannot-pin")
        return

    # BRACKETED (points on both sides of 50%): grind the INTERPOLATED crossing — the most central
    # λ — rather than concentrating a slightly-off best point (which would need far more games to
    # lock). The isotonic fit tames noise; the crossing estimate refines each chunk.
    if bracketed:
        # Concentrate a WELL-SAMPLED λ ONLY when it's genuinely near 50% (±3%, >=50 games) -> it locks
        # fast on its own. Do NOT concentrate an EDGE point (e.g. 54% or 46%) even under heavy sampling:
        # that just spins without ever tightening into [40,60]. With no near-50% point, fall through and
        # grind the interpolated crossing (which targets 50% and, via the snap below, still concentrates).
        central = sorted((abs(w/g-0.5), l, g) for (l, w, g) in pts if g >= 50 and abs(w/g-0.5) <= 0.03)
        if central:
            print(f"ACTION=GRIND LAMBDA={round(central[0][1],5):.5f} NOTE=concentrate-central-{central[0][2]}g")
            return
        lam_star = None
        for k in range(len(fit)-1):
            a_lam, b_lam = fit[k][1], fit[k+1][0]
            a_wr, b_wr = fit[k][2], fit[k+1][2]
            if a_wr >= 0.5 >= b_wr and a_wr != b_wr:
                frac = (a_wr - 0.5) / (a_wr - b_wr)
                lam_star = a_lam + frac * (b_lam - a_lam); break
        if lam_star is None:
            lam_star = 0.5*(lo_lam+hi_lam)
        # Grind the interpolated crossing directly (targets 50%). Consolidation near 50% is handled by
        # the central-concentrate check above (±3%); snapping the crossing onto a spatially-near but
        # slightly-off λ (e.g. 54%) only stalls — it grinds the wrong side and never tightens to lock.
        lam_grid = round(round(lam_star/GRID)*GRID, 5)
        print(f"ACTION=GRIND LAMBDA={lam_grid:.5f} NOTE=bracketed-cross~{lam_star:.5f}-total{total}g")
        return

    # NOT bracketed: concentrate the best λ ONLY if it's VERY central (CI includes 50% AND within
    # ±3%) — a point 3-5% off is an edge point that locks slowly; better to EXPAND and find the real
    # 50% crossing (where the lock comes fast) than to grind ~340 games at a persistently-55% λ.
    if blo <= 0.5 <= bhi and abs(bp - 0.5) <= 0.03:
        print(f"ACTION=GRIND LAMBDA={best[0]:.5f} NOTE=concentrate-{best[2]}g-CI[{100*blo:.0f},{100*bhi:.0f}]")
        return
    # ...otherwise EXPAND from the extreme λ toward 50% to find a bracket.
    if bp < 0.5:                            # all too weak -> stronger; expand below strongest λ
        ext = pts[0]; ewr = ext[1]/ext[2]
        step = min(0.05, max(0.01, abs(ewr-0.5)*0.30 + 0.006))
        nxt = round(max(GRID, ext[0] - step), 5)
        print(f"ACTION=GRIND LAMBDA={nxt:.5f} NOTE=all-too-weak({100*bp:.0f}%)-expand-stronger")
    else:                                   # all too strong -> weaker; expand above weakest λ
        ext = pts[-1]; ewr = ext[1]/ext[2]
        step = min(0.05, max(0.01, abs(ewr-0.5)*0.30 + 0.006))
        nxt = round(ext[0] + step, 5)
        print(f"ACTION=GRIND LAMBDA={nxt:.5f} NOTE=all-too-strong({100*bp:.0f}%)-expand-weaker")

if __name__ == "__main__":
    main()
