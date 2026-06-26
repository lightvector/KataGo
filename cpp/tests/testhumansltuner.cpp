#include "../tests/tests.h"

#include "../program/humansltuner.h"

#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

using namespace std;

static double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

void Tests::runHumanSLTunerTests() {
  cout << "Running human SL tuner tests" << endl;

  // Test 1: LogisticRS recovers known coefficients.
  {
    LogisticRS rs(0.5);
    double xsv[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for(double x : xsv) {
      double p = sigmoid(0.5 - 2.0 * x);
      rs.addSample(x, std::round(1000.0 * p), 1000.0);
    }
    rs.fit();
    testAssert(std::fabs(rs.getB0() - 0.5) < 0.1);
    testAssert(std::fabs(rs.getB1() + 2.0) < 0.1);
  }

  // Test 2: root recovers the target dial.
  {
    LogisticRS rs(0.5);
    double xsv[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    for(double x : xsv) {
      double p = sigmoid(-x); // b ~ (0, -1)
      rs.addSample(x, std::round(1000.0 * p), 1000.0);
    }
    rs.fit();
    testAssert(std::fabs(rs.root(0.36) - 0.5754) < 0.05);
    testAssert(rs.rootSeElo(0.36) < 50.0);
  }

  // Test 3: CI shrinks with more data.
  {
    LogisticRS big(0.5), small(0.5);
    double xsv[] = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    for(double x : xsv) {
      double p = sigmoid(-x);
      big.addSample(x, std::round(2000.0 * p), 2000.0);
      small.addSample(x, std::round(80.0 * p), 80.0);
    }
    big.fit();
    small.fit();
    testAssert(big.rootSeElo(0.36) < small.rootSeElo(0.36));
  }

  // Test 4: dial schedule monotonicity and continuity.
  {
    StrengthDialConfig c; // defaults
    double prevDtauA = 1e18;
    double prevPiklB = 1e18;
    int prevVisC = -1;
    for(int i = 0; i <= 60; i++) {
      double x = i * 0.05;
      StrengthDialParams p = strengthDialToParams(x, c);
      if(x < 1.0) {
        testAssert(p.maxVisits == 1);
        testAssert(p.piklLambda == StrengthDialConfig::PIKL_INERT);
        testAssert(p.deltaTau <= prevDtauA + 1e-12);
        prevDtauA = p.deltaTau;
      } else if(x < 2.0) {
        testAssert(p.maxVisits == c.searchVisits);
        testAssert(p.deltaTau == 0.0);
        testAssert(p.piklLambda <= prevPiklB + 1e-9);
        prevPiklB = p.piklLambda;
      } else {
        testAssert(std::fabs(p.piklLambda - c.piklFloor) < 1e-12);
        testAssert(p.maxVisits >= prevVisC);
        prevVisC = p.maxVisits;
      }
    }
    // Continuity at x == 2: both sides give maxVisits == searchVisits and piklLambda == piklFloor.
    StrengthDialParams justBelow = strengthDialToParams(2.0 - 1e-9, c);
    StrengthDialParams at2 = strengthDialToParams(2.0, c);
    testAssert(at2.maxVisits == c.searchVisits);
    testAssert(justBelow.maxVisits == c.searchVisits);
    testAssert(std::fabs(at2.piklLambda - c.piklFloor) < 1e-9);
    testAssert(std::fabs(justBelow.piklLambda - c.piklFloor) < 1e-6);
  }

  // Test 5: calibrateToTarget is unbiased with an honest CI. Deterministic (fixed seeds).
  {
    auto winrateOfElo = [](double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); };

    auto runScenario = [&](const std::function<double(double)>& eloFn) {
      const int numSeeds = 100;
      double sumErr = 0.0, sumSqErr = 0.0;
      int cover1 = 0, cover2 = 0;
      for(int s = 0; s < numSeeds; s++) {
        std::mt19937_64 playRng((uint64_t)(1000 + s));
        auto playAt = [&](double x) -> std::pair<double,int> {
          double wr = winrateOfElo(eloFn(x));
          int games = 20;
          std::binomial_distribution<int> binom(games, wr);
          int wins = binom(playRng);
          return std::make_pair((double)wins, games);
        };
        CalibrationResult res = calibrateToTarget(
          playAt, 0.0, 1.0, 0.36, 20, 30, 25.0, (uint64_t)(s + 1), 0.5, nullptr);
        double err = eloFn(res.xStar) + 100.0; // true target ELO is -100
        sumErr += err;
        sumSqErr += err * err;
        if(std::fabs(err) <= res.eloSe) cover1++;
        if(std::fabs(err) <= 2.0 * res.eloSe) cover2++;
      }
      double meanErr = sumErr / numSeeds;
      double rmse = std::sqrt(sumSqErr / numSeeds);
      double cov1 = (double)cover1 / numSeeds;
      double cov2 = (double)cover2 / numSeeds;
      testAssert(std::fabs(meanErr) < 15.0); // unbiased
      testAssert(rmse < 45.0);
      testAssert(cov1 >= 0.55 && cov1 <= 0.90); // honest, not overconfident
      testAssert(cov2 >= 0.88);
    };

    runScenario([](double x) { return -100.0 + 300.0 * (x - 0.5); });
    runScenario([](double x) { double d = x - 0.5; return -100.0 + 250.0 * d + 500.0 * d * d * d; });
  }

  // Test 6: overrideConfigText replaces existing keys, ignores comments, appends new keys.
  {
    std::string input = "a = 1\nb=2\n# c = 3\n";
    std::vector<std::pair<std::string,std::string>> ov = {{"b", "9"}, {"d", "4"}};
    std::string out = overrideConfigText(input, ov);
    testAssert(out == "a = 1\nb = 9\n# c = 3\nd = 4\n");
  }

  // Test 7: an unreachable target pins x* to the boundary, never reports "converged",
  // and keeps the reported CI NaN-safe. Exercises the degenerate extrapolation regime
  // (candidate far stronger than the target across the whole dial range).
  {
    auto winrateOfElo = [](double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); };
    std::mt19937_64 playRng(12345);
    auto playAt = [&](double x) -> std::pair<double,int> {
      double elo = 150.0 + 100.0 * x; // always >= +150 ELO; the 0.36 (-100 ELO) root lies below xLo
      double wr = winrateOfElo(elo);
      int games = 20;
      std::binomial_distribution<int> binom(games, wr);
      return std::make_pair((double)binom(playRng), games);
    };
    CalibrationResult res = calibrateToTarget(
      playAt, 0.0, 1.0, 0.36, 20, 30, 25.0, (uint64_t)7, 0.5, nullptr);
    testAssert(res.converged == false);
    testAssert(std::fabs(res.xStar - 0.0) < 1e-6); // pinned to xLo
    testAssert(!std::isnan(res.eloSe));            // honest CI: large/inf allowed, NaN never
    testAssert(res.eloSe >= 0.0);
    testAssert(res.totalGames > 0);
  }

  // Test 8: LogisticRS stays NaN-safe under near-degenerate data.
  {
    // (a) Perfectly separable data (all losses below 0, all wins above). The MLE slope
    // diverges; L2 must keep coefficients finite and the reported CI non-NaN.
    LogisticRS sep(0.5);
    sep.addSample(-1.0, 0.0, 50.0);
    sep.addSample(-1.0, 0.0, 50.0);
    sep.addSample( 1.0, 50.0, 50.0);
    sep.addSample( 1.0, 50.0, 50.0);
    sep.fit();
    testAssert(std::isfinite(sep.getB0()));
    testAssert(std::isfinite(sep.getB1()));
    testAssert(!std::isnan(sep.rootSeElo(0.36)));

    // (b) No spread in x: the slope is unidentified. root() must be NaN (not +-inf) and
    // rootSeElo() must be a non-NaN sentinel (+inf), with no crash.
    LogisticRS flat(0.5);
    for(int i = 0; i < 5; i++) flat.addSample(0.5, 25.0, 50.0);
    flat.fit();
    testAssert(std::isfinite(flat.getB0()));
    testAssert(std::isfinite(flat.getB1()));
    double r = flat.root(0.36);
    double se = flat.rootSeElo(0.36);
    testAssert(std::isnan(r) || std::isfinite(r)); // defined-or-NaN, never an inf trap
    testAssert(!std::isnan(se));
  }

  // Test 9: convergence is structurally impossible with fewer than 4 rounds, even on a
  // clean low-noise reachable surface (it requires >= 4 distinct dial samples). This pins
  // down the invariant that motivates the CLI's max-rounds warning.
  {
    auto winrateOfElo = [](double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); };
    std::mt19937_64 playRng(999);
    auto playAt = [&](double x) -> std::pair<double,int> {
      double wr = winrateOfElo(-100.0 + 300.0 * (x - 0.5)); // reachable; -100 ELO at x=0.5
      int games = 200;
      std::binomial_distribution<int> binom(games, wr);
      return std::make_pair((double)binom(playRng), games);
    };
    for(int mr = 1; mr <= 3; mr++) {
      CalibrationResult res = calibrateToTarget(
        playAt, 0.0, 1.0, 0.36, 200, mr, 25.0, (uint64_t)(100 + mr), 0.5, nullptr);
      testAssert(res.converged == false);
      testAssert(res.rounds == mr);
    }
  }

  // Test 10: resolveVisitBudget auto+auto anchors mid==cap==baseline and never raises (B in {2,12,400}).
  // This is the headline requirement: with both knobs auto, the candidate's visit budget collapses
  // onto the baseline, so segment C is flat and visits can never exceed the baseline.
  {
    int Bs[] = {2, 12, 400};
    for(int B : Bs) {
      VisitBudget vb = resolveVisitBudget((int64_t)B, -1, -1);
      testAssert(vb.midVisits == B);
      testAssert(vb.maxVisitsCap == B);
      testAssert(vb.raisesAboveBaseline == false);
      testAssert(vb.flooredFromBelow2 == false);
      testAssert(vb.midVisits >= 2);
      testAssert(vb.maxVisitsCap >= vb.midVisits);
    }
  }

  // Test 11: explicit -max-visits-cap above baseline is honored and flags raisesAboveBaseline
  // (the only intended way visits exceed the baseline -- so the CLI can warn).
  {
    VisitBudget vb = resolveVisitBudget((int64_t)12, -1, 400);
    testAssert(vb.midVisits == 12);
    testAssert(vb.maxVisitsCap == 400);
    testAssert(vb.raisesAboveBaseline == true);
    testAssert(vb.maxVisitsCap >= vb.midVisits);
  }

  // Test 12: explicit -max-visits-cap below baseline is clamped UP to mid (segment C never drops below
  // segment B, else the log2 interpolation would run downward and break monotonicity); no raise.
  {
    VisitBudget vb = resolveVisitBudget((int64_t)12, -1, 5);
    testAssert(vb.midVisits == 12);
    testAssert(vb.maxVisitsCap == 12);
    testAssert(vb.maxVisitsCap >= vb.midVisits);
    testAssert(vb.raisesAboveBaseline == false);
  }

  // Test 13: explicit -search-visits below 2 is floored to 2 (piklLambda needs >1 visit), cap auto-anchors.
  {
    VisitBudget vb = resolveVisitBudget((int64_t)12, 1, -1);
    testAssert(vb.midVisits == 2);
    testAssert(vb.maxVisitsCap == 12);
    testAssert(vb.flooredFromBelow2 == true);
    testAssert(vb.raisesAboveBaseline == false);
    VisitBudget vb0 = resolveVisitBudget((int64_t)12, 0, -1);
    testAssert(vb0.midVisits == 2);
    testAssert(vb0.maxVisitsCap == 12);
    testAssert(vb0.flooredFromBelow2 == true);
    testAssert(vb0.raisesAboveBaseline == false);
  }

  // Test 14: baseline maxVisits==1 edge. piklLambda is inert at 1 visit, so segment B must run at 2;
  // that unavoidably raises above the degenerate 1-visit baseline, and the helper must report it (the
  // CLI then emits the *soft* floor warning, gated on flooredFromBelow2, not the loud over-baseline one).
  // The adjacent non-degenerate baseline B==2 must NOT raise.
  {
    VisitBudget vb = resolveVisitBudget((int64_t)1, -1, -1);
    testAssert(vb.midVisits == 2);
    testAssert(vb.maxVisitsCap == 2);
    testAssert(vb.raisesAboveBaseline == true);
    testAssert(vb.flooredFromBelow2 == true);
    VisitBudget vb2 = resolveVisitBudget((int64_t)2, -1, -1);
    testAssert(vb2.midVisits == 2);
    testAssert(vb2.maxVisitsCap == 2);
    testAssert(vb2.raisesAboveBaseline == false);
    testAssert(vb2.flooredFromBelow2 == false);
  }

  // Test 15: no-cap sentinel. A baseline that omits maxVisits gets SearchParams' ctor default 1<<50;
  // the helper must treat that as "no real cap" and anchor to the legacy 100, NEVER to 2^50 (which an
  // int signature would have truncated/exploded). B==0 is likewise treated as no-cap.
  {
    VisitBudget vb = resolveVisitBudget(((int64_t)1) << 50, -1, -1);
    testAssert(vb.midVisits == 100);
    testAssert(vb.maxVisitsCap == 100);
    testAssert(vb.raisesAboveBaseline == false);
    testAssert(vb.baselineHasCap == false);
    testAssert(vb.effectiveBaseline == 100);
    VisitBudget vb0 = resolveVisitBudget((int64_t)0, -1, -1);
    testAssert(vb0.midVisits == 100);
    testAssert(vb0.maxVisitsCap == 100);
    testAssert(vb0.raisesAboveBaseline == false);
    testAssert(vb0.baselineHasCap == false);

    // Finite-but-absurd baseline in (1e6, 1<<50): the int64->int anchor clamp must hold at 1e6, NOT
    // truncate. Without the ABS_MAX clamp, (int)(1<<40) == 0 and midVisits would collapse to 2.
    VisitBudget vbBig = resolveVisitBudget(((int64_t)1) << 40, -1, -1);
    testAssert(vbBig.midVisits == 1000000);
    testAssert(vbBig.maxVisitsCap == 1000000);
    testAssert(vbBig.effectiveBaseline == 1000000);
    testAssert(vbBig.raisesAboveBaseline == false);
    VisitBudget vbCtl = resolveVisitBudget((int64_t)5000000, -1, -1); // just inside the clamp window
    testAssert(vbCtl.midVisits == 1000000);
    testAssert(vbCtl.maxVisitsCap == 1000000);

    // No-cap baseline + explicit override: raisesAboveBaseline must stay false (no finite baseline to
    // exceed), guarding the baselineHasCap gate; the explicit cap is still honored.
    VisitBudget vbNoCapExplicit = resolveVisitBudget(((int64_t)1) << 50, -1, 9999);
    testAssert(vbNoCapExplicit.midVisits == 100);
    testAssert(vbNoCapExplicit.maxVisitsCap == 9999);
    testAssert(vbNoCapExplicit.raisesAboveBaseline == false);
    testAssert(vbNoCapExplicit.baselineHasCap == false);
    testAssert(vbNoCapExplicit.effectiveBaseline == 100);
  }

  // Test 16: dial invariant under auto -- a StrengthDialConfig built from resolveVisitBudget keeps
  // strengthDialToParams' maxVisits <= baseline for ALL x (B>=2), and segment C is flat at B on [2,3].
  // This binds the "visits never increase under auto" requirement to the actual dial output.
  {
    int Bs[] = {2, 12, 400};
    for(int B : Bs) {
      VisitBudget vb = resolveVisitBudget((int64_t)B, -1, -1);
      testAssert(vb.raisesAboveBaseline == false);
      StrengthDialConfig c; // defaults for pikl*/dtau
      c.searchVisits = vb.midVisits;
      c.maxVisitsCap = vb.maxVisitsCap;
      for(int i = 0; i <= 300; i++) {
        double x = i * 0.01; // 0.00 .. 3.00
        StrengthDialParams p = strengthDialToParams(x, c);
        testAssert(p.maxVisits <= B);
        testAssert(p.maxVisits >= 1);
        testAssert(p.maxVisits <= c.maxVisitsCap);
        if(x >= 2.0)
          testAssert(p.maxVisits == B); // segment C flat at baseline
      }
    }
  }

  // Test 17: positive control -- when the user explicitly raises the cap, segment C DOES climb above
  // baseline, confirming Test 16's invariant is gated on auto and not vacuously true.
  {
    int B = 12;
    VisitBudget vb = resolveVisitBudget((int64_t)B, -1, 400);
    testAssert(vb.raisesAboveBaseline == true);
    StrengthDialConfig c;
    c.searchVisits = vb.midVisits;
    c.maxVisitsCap = vb.maxVisitsCap;
    StrengthDialParams strong = strengthDialToParams(3.0, c);
    StrengthDialParams mid = strengthDialToParams(1.5, c);
    StrengthDialParams weak = strengthDialToParams(0.0, c);
    testAssert(strong.maxVisits == 400);
    testAssert(strong.maxVisits > B);
    testAssert(mid.maxVisits == 12);
    testAssert(weak.maxVisits == 1);
  }

  // Test 18: explicit -search-visits >= 2 passes through unchanged (SC-3), and an explicit mid above a
  // finite baseline raises via the mid lever with flooredFromBelow2==false (SC-5) -- the exact precondition
  // for the loud over-baseline warning to fire through the mid lever (no current test reached this).
  {
    VisitBudget pass = resolveVisitBudget((int64_t)400, 50, -1);
    testAssert(pass.midVisits == 50);          // explicit mid honored, not anchored to 400
    testAssert(pass.flooredFromBelow2 == false);
    testAssert(pass.maxVisitsCap == 400);      // auto cap anchors to baseline >= mid
    testAssert(pass.raisesAboveBaseline == false); // 50 < 400, 400 == 400

    VisitBudget midRaise = resolveVisitBudget((int64_t)12, 50, -1);
    testAssert(midRaise.midVisits == 50);
    testAssert(midRaise.maxVisitsCap == 50);   // cap auto = max(mid=50, anchor=12) = 50
    testAssert(midRaise.flooredFromBelow2 == false);
    testAssert(midRaise.raisesAboveBaseline == true); // mid 50 > baseline 12
    testAssert(midRaise.effectiveBaseline == 12);

    // The MF-1 scenario: a floored mid (-search-visits 1 -> 2) AND an explicit cap far above baseline.
    // flooredFromBelow2 must NOT suppress the cap-driven over-baseline signal: the CLI's loud warning
    // gates on (maxVisitsCap != -1 && maxVisitsCap > effectiveBaseline), which is true here.
    VisitBudget capRaiseFloored = resolveVisitBudget((int64_t)12, 1, 400);
    testAssert(capRaiseFloored.midVisits == 2);
    testAssert(capRaiseFloored.maxVisitsCap == 400);
    testAssert(capRaiseFloored.flooredFromBelow2 == true);
    testAssert(capRaiseFloored.raisesAboveBaseline == true);
    testAssert(capRaiseFloored.effectiveBaseline == 12);
    testAssert(capRaiseFloored.maxVisitsCap > capRaiseFloored.effectiveBaseline); // -> loud warning fires
  }

  // Test 19: effectiveXHi shrinks the calibration range to 2.0 only when segment C is flat (cap==mid,
  // the auto outcome) AND the range straddles x=2; otherwise it returns xHi unchanged.
  {
    VisitBudget flat = resolveVisitBudget((int64_t)12, -1, -1);   // cap==mid==12 -> flat segment C
    testAssert(effectiveXHi(flat, 0.0, 3.0) == 2.0);             // auto/flat, straddles 2 -> shrink
    testAssert(effectiveXHi(flat, 0.0, 1.5) == 1.5);             // xHi already <= 2 -> no shrink
    testAssert(effectiveXHi(flat, 2.5, 3.0) == 3.0);             // xLo >= 2 (all-in-plateau) -> not shrunk here
    VisitBudget raised = resolveVisitBudget((int64_t)12, -1, 400); // cap 400 != mid 12 -> non-flat
    testAssert(effectiveXHi(raised, 0.0, 3.0) == 3.0);           // real visit gradient -> keep full range
  }

  // Test 20: resume. A calibration can be checkpointed per-round and continued across process restarts
  // (the `tunehuman` command persists each round so an environment runtime cap can't lose progress).
  // (a) onSampleCollected fires exactly once per NEW round. (b) Resuming with an already-converged sample
  // set returns converged WITHOUT playing more games and reproduces the same fit. (c) A split run
  // (chunk1, then resume) continues the round/game counts and still converges accurately.
  {
    auto winrateOfElo = [](double elo) { return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0)); };
    auto eloFn = [](double x) { return -100.0 + 300.0 * (x - 0.5); }; // reachable; -100 ELO at x=0.5

    // (a) one onSampleCollected call per round, each carrying that round's games.
    {
      std::mt19937_64 playRng(2024);
      auto playAt = [&](double x) -> std::pair<double,int> {
        int games = 50;
        std::binomial_distribution<int> binom(games, winrateOfElo(eloFn(x)));
        return std::make_pair((double)binom(playRng), games);
      };
      std::vector<CalibrationSample> collected;
      auto onSample = [&](double x, double wins, double games) {
        collected.push_back(CalibrationSample{x, wins, games});
      };
      CalibrationResult res = calibrateToTarget(
        playAt, 0.0, 1.0, 0.36, 50, 6, 25.0, (uint64_t)42, 0.5, nullptr,
        std::vector<CalibrationSample>(), onSample);
      testAssert((int)collected.size() == res.rounds);
      double sumGames = 0.0;
      for(const CalibrationSample& s : collected) { testAssert(s.games == 50.0); sumGames += s.games; }
      testAssert((int)sumGames == res.totalGames);
    }

    // (b) resuming with a converged sample set short-circuits: playAt never called, fit reproduced exactly.
    {
      std::mt19937_64 playRng(7);
      auto playAt = [&](double x) -> std::pair<double,int> {
        int games = 200;
        std::binomial_distribution<int> binom(games, winrateOfElo(eloFn(x)));
        return std::make_pair((double)binom(playRng), games);
      };
      std::vector<CalibrationSample> samples;
      auto cap = [&](double x, double wins, double games) {
        samples.push_back(CalibrationSample{x, wins, games});
      };
      CalibrationResult full = calibrateToTarget(
        playAt, 0.0, 1.0, 0.36, 200, 30, 25.0, (uint64_t)123, 0.5, nullptr,
        std::vector<CalibrationSample>(), cap);
      testAssert(full.converged);
      testAssert((int)samples.size() == full.rounds);

      bool playCalled = false;
      auto noPlay = [&](double x) -> std::pair<double,int> {
        (void)x; playCalled = true; return std::make_pair(0.0, 0);
      };
      CalibrationResult resumed = calibrateToTarget(
        noPlay, 0.0, 1.0, 0.36, 200, 30, 25.0, (uint64_t)123, 0.5, nullptr, samples, nullptr);
      testAssert(!playCalled);                                    // no new games played
      testAssert(resumed.converged);
      testAssert(resumed.rounds == (int)samples.size());
      testAssert(std::fabs(resumed.xStar - full.xStar) < 1e-9);   // identical fit from identical samples
      testAssert(resumed.totalGames == full.totalGames);
    }

    // (c) split run: chunk1 (3 rounds, not yet converged) then resume continues and converges.
    {
      std::mt19937_64 playRng(555);
      auto playAt = [&](double x) -> std::pair<double,int> {
        int games = 200;
        std::binomial_distribution<int> binom(games, winrateOfElo(eloFn(x)));
        return std::make_pair((double)binom(playRng), games);
      };
      std::vector<CalibrationSample> s;
      auto cap = [&](double x, double wins, double games) { s.push_back(CalibrationSample{x, wins, games}); };
      CalibrationResult chunk1 = calibrateToTarget(
        playAt, 0.0, 1.0, 0.36, 200, 3, 25.0, (uint64_t)900, 0.5, nullptr,
        std::vector<CalibrationSample>(), cap);
      testAssert(chunk1.converged == false);
      testAssert((int)s.size() == 3);

      std::vector<CalibrationSample> seed = s; // resume from the 3 checkpointed rounds
      CalibrationResult chunk2 = calibrateToTarget(
        playAt, 0.0, 1.0, 0.36, 200, 30, 25.0, (uint64_t)900, 0.5, nullptr, seed, cap);
      testAssert(chunk2.rounds > 3);                  // continued past the checkpoint
      testAssert(chunk2.rounds == (int)s.size());     // every NEW round was checkpointed too
      double allGames = 0.0; for(const CalibrationSample& cs : s) allGames += cs.games;
      testAssert((int)allGames == chunk2.totalGames); // total accounts for prior + new games
      testAssert(chunk2.converged);
      testAssert(std::fabs(eloFn(chunk2.xStar) + 100.0) < 60.0); // accurate near -100 ELO
    }
  }

  cout << "Done human SL tuner tests" << endl;
}
