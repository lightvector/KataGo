#include "../program/humansltuner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

static double clipd(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static double lerp(double a, double b, double t) { return a + (b - a) * t; }

LogisticRS::LogisticRS(double l2_)
  : l2(l2_), xs(), ws(), ns(), b0(0.0), b1(0.0), fitted(false) {
  cov[0][0] = 0.0; cov[0][1] = 0.0; cov[1][0] = 0.0; cov[1][1] = 0.0;
}

void LogisticRS::addSample(double x, double wins, double games) {
  xs.push_back(x);
  ws.push_back(wins);
  ns.push_back(games);
}

LogisticRS& LogisticRS::fit(int iters) {
  for(int iter = 0; iter < iters; iter++) {
    double g0 = l2 * b0;
    double g1 = l2 * b1;
    double S0 = l2, S1 = 0.0, S2 = l2;
    for(size_t i = 0; i < xs.size(); i++) {
      double z = clipd(b0 + b1 * xs[i], -30.0, 30.0);
      double p = 1.0 / (1.0 + std::exp(-z));
      double resid = ns[i] * p - ws[i];
      g0 += resid;
      g1 += xs[i] * resid;
      double w = clipd(ns[i] * p * (1.0 - p), 1e-9, std::numeric_limits<double>::infinity());
      S0 += w;
      S1 += w * xs[i];
      S2 += w * xs[i] * xs[i];
    }
    double H00 = S0, H01 = S1, H10 = S1, H11 = S2;
    double det = H00 * H11 - H01 * H10;
    if(std::fabs(det) < 1e-12)
      continue;
    double step0 = clipd((H11 * g0 - H01 * g1) / det, -10.0, 10.0);
    double step1 = clipd((-H10 * g0 + H00 * g1) / det, -10.0, 10.0);
    b0 -= step0;
    b1 -= step1;
  }

  // Recompute covariance = (X^T W X + l2 I)^-1 at the final coefficients.
  double S0 = l2, S1 = 0.0, S2 = l2;
  for(size_t i = 0; i < xs.size(); i++) {
    double z = clipd(b0 + b1 * xs[i], -30.0, 30.0);
    double p = 1.0 / (1.0 + std::exp(-z));
    double w = clipd(ns[i] * p * (1.0 - p), 1e-9, std::numeric_limits<double>::infinity());
    S0 += w;
    S1 += w * xs[i];
    S2 += w * xs[i] * xs[i];
  }
  double det = S0 * S2 - S1 * S1;
  if(std::fabs(det) < 1e-12) {
    cov[0][0] = cov[0][1] = cov[1][0] = cov[1][1] = 0.0;
  } else {
    cov[0][0] = S2 / det;
    cov[0][1] = -S1 / det;
    cov[1][0] = -S1 / det;
    cov[1][1] = S0 / det;
  }
  fitted = true;
  return *this;
}

double LogisticRS::predict(double x) const {
  double z = clipd(b0 + b1 * x, -30.0, 30.0);
  return 1.0 / (1.0 + std::exp(-z));
}

double LogisticRS::root(double targetWinrate) const {
  if(std::fabs(b1) < 1e-9)
    return std::nan("");
  double logitT = std::log(targetWinrate / (1.0 - targetWinrate));
  return (logitT - b0) / b1;
}

double LogisticRS::rootSeElo(double targetWinrate) const {
  if(!fitted || std::fabs(b1) < 1e-9)
    return std::numeric_limits<double>::infinity();
  double logitT = std::log(targetWinrate / (1.0 - targetWinrate));
  double dx_db0 = -1.0 / b1;
  double dx_db1 = -(logitT - b0) / (b1 * b1);
  double varX = dx_db0 * dx_db0 * cov[0][0]
              + 2.0 * dx_db0 * dx_db1 * cov[0][1]
              + dx_db1 * dx_db1 * cov[1][1];
  double eloPerX = std::fabs(b1) * LogisticRS::ELO_PER_LOGIT;
  return eloPerX * std::sqrt(std::max(varX, 0.0));
}

int LogisticRS::distinctXCount(double eps) const {
  std::vector<double> sorted = xs;
  std::sort(sorted.begin(), sorted.end());
  int count = 0;
  for(size_t i = 0; i < sorted.size(); i++) {
    if(i == 0 || sorted[i] - sorted[i - 1] > eps)
      count++;
  }
  return count;
}

VisitBudget resolveVisitBudget(int64_t baselineMaxVisits, int userSearchVisits, int userMaxVisitsCap) {
  const int64_t NO_REAL_CAP   = (int64_t)1 << 50; // == SearchParams ctor default (search bounded elsewhere)
  const int     LEGACY_ANCHOR = 100;              // fallback anchor when the baseline has no finite cap
  const int     ABS_MAX       = 1000000;          // hard ceiling so a finite-but-huge int64 baseline can't overflow int

  bool baselineHasCap = (baselineMaxVisits > 0 && baselineMaxVisits < NO_REAL_CAP);
  int  anchor = baselineHasCap
                  ? (int)std::min<int64_t>(baselineMaxVisits, (int64_t)ABS_MAX)
                  : LEGACY_ANCHOR;

  // Segment B depth: piklLambda is inert below 2 visits, so floor at 2.
  int  rawMid = (userSearchVisits == -1) ? anchor : userSearchVisits;
  int  midVisits = std::max(2, rawMid);
  bool flooredFromBelow2 = (rawMid < 2);

  // Segment C strong end: auto climbs back to the baseline anchor (never above it); explicit is honored
  // but never below mid, so the segment-C log2 interpolation never runs downward.
  int  rawCap = (userMaxVisitsCap == -1) ? std::max(midVisits, anchor) : userMaxVisitsCap;
  int  maxVisitsCap = std::max(midVisits, rawCap);

  bool raisesAboveBaseline = baselineHasCap &&
      ((int64_t)midVisits > baselineMaxVisits || (int64_t)maxVisitsCap > baselineMaxVisits);

  return VisitBudget{midVisits, maxVisitsCap, raisesAboveBaseline, flooredFromBelow2, anchor, baselineHasCap};
}

double effectiveXHi(const VisitBudget& vb, double xLo, double xHi) {
  if(vb.maxVisitsCap == vb.midVisits && xHi > 2.0 && xLo < 2.0)
    return 2.0;
  return xHi;
}

StrengthDialParams strengthDialToParams(double x, const StrengthDialConfig& c) {
  x = clipd(x, 0.0, 3.0);
  StrengthDialParams out;
  if(x < 1.0) {
    // Segment A (weak): temperature lever at 1 visit (piklLambda is inert at 1 visit).
    out.maxVisits = 1;
    out.piklLambda = StrengthDialConfig::PIKL_INERT;
    out.deltaTau = c.dtauMax * (1.0 - x);
  } else if(x < 2.0) {
    // Segment B (mid): piklLambda lever with search on.
    out.maxVisits = c.searchVisits;
    double lg = lerp(std::log10(c.piklMax), std::log10(c.piklFloor), x - 1.0);
    out.piklLambda = std::pow(10.0, lg);
    out.deltaTau = 0.0;
  } else {
    // Segment C (strong): visits lever, piklLambda fully trusted.
    double lg = lerp(std::log2((double)c.searchVisits), std::log2((double)c.maxVisitsCap), x - 2.0);
    out.maxVisits = (int)std::lround(std::pow(2.0, lg));
    out.piklLambda = c.piklFloor;
    out.deltaTau = 0.0;
  }
  return out;
}

CalibrationResult calibrateToTarget(
  const std::function<std::pair<double,int>(double)>& playAt,
  double xLo, double xHi, double targetWinrate,
  int gamesPerRound, int maxRounds, double eloTol,
  uint64_t rngSeed, double l2,
  const std::function<void(int,double,double,int,int)>& onRound,
  const std::vector<CalibrationSample>& initialSamples,
  const std::function<void(double,double,double)>& onSampleCollected
) {
  (void)gamesPerRound; // games count comes from playAt's return value
  LogisticRS rs(l2);
  double xStar = 0.5 * (xLo + xHi);
  double se = std::numeric_limits<double>::infinity();
  int totalGames = 0;
  bool converged = false;

  // Resume: seed the fit with any prior rounds' samples so an interrupted calibration continues instead
  // of restarting. The round loop then begins at initialSamples.size().
  for(const CalibrationSample& s : initialSamples) {
    rs.addSample(s.x, s.wins, s.games);
    totalGames += (int)s.games;
  }
  const int startRound = (int)initialSamples.size();
  int roundsRun = startRound;
  if(startRound > 0) {
    rs.fit();
    double r0 = rs.root(targetWinrate);
    if(std::isfinite(r0))
      xStar = clipd(r0, xLo, xHi);
    se = rs.rootSeElo(targetWinrate);
    // If the reloaded samples already satisfy convergence, finish without playing any more games.
    if(startRound >= 4 && rs.distinctXCount() >= 4 && se <= eloTol)
      converged = true;
  }

  // Perturbing the seed by startRound keeps each resumed chunk exploring fresh offsets; for the
  // from-scratch path (startRound == 0) this is exactly rngSeed, so that path is byte-identical to before.
  std::mt19937_64 rng(rngSeed + 0x9e3779b97f4a7c15ULL * (uint64_t)startRound);
  std::uniform_real_distribution<double> uniform(xLo, xHi);

  for(int rnd = startRound; !converged && rnd < maxRounds; rnd++) {
    roundsRun = rnd + 1;
    double x;
    if(rnd < 2) {
      x = uniform(rng); // explore uniformly the first 2 rounds
    } else {
      double sigma = std::max(0.05, 0.5 * (xHi - xLo) * std::pow(0.85, (double)rnd));
      std::normal_distribution<double> gaussian(0.0, sigma);
      x = clipd(xStar + gaussian(rng), xLo, xHi);
    }
    std::pair<double,int> res = playAt(x);
    double wins = res.first;
    int games = res.second;
    rs.addSample(x, wins, (double)games);
    rs.fit();
    double r = rs.root(targetWinrate);
    if(std::isfinite(r))
      xStar = clipd(r, xLo, xHi);
    se = rs.rootSeElo(targetWinrate);
    totalGames += games;
    if(onSampleCollected)
      onSampleCollected(x, wins, (double)games);
    if(onRound)
      onRound(rnd, xStar, se, rs.distinctXCount(), totalGames);
    if(rnd >= 3 && rs.distinctXCount() >= 4 && se <= eloTol) {
      converged = true;
      break;
    }
  }

  CalibrationResult result;
  result.xStar = xStar;
  result.eloSe = se;
  result.totalGames = totalGames;
  result.rounds = roundsRun;
  result.converged = converged;
  result.model = rs;
  return result;
}

std::string overrideConfigText(
  const std::string& baselineText,
  const std::vector<std::pair<std::string,std::string>>& overrides
) {
  // Split into lines (dropping CR), remembering content; we re-join with '\n'.
  std::vector<std::string> lines;
  {
    std::string cur;
    for(char ch : baselineText) {
      if(ch == '\n') { lines.push_back(cur); cur.clear(); }
      else if(ch == '\r') { /* drop */ }
      else cur.push_back(ch);
    }
    lines.push_back(cur);
  }

  std::vector<bool> applied(overrides.size(), false);

  for(std::string& line : lines) {
    size_t start = 0;
    while(start < line.size() && (line[start] == ' ' || line[start] == '\t'))
      start++;
    if(start >= line.size() || line[start] == '#')
      continue;
    size_t eq = line.find('=', start);
    if(eq == std::string::npos)
      continue;
    size_t keyEnd = start;
    while(keyEnd < eq && line[keyEnd] != ' ' && line[keyEnd] != '\t')
      keyEnd++;
    std::string key = line.substr(start, keyEnd - start);
    for(size_t k = 0; k < overrides.size(); k++) {
      if(!applied[k] && key == overrides[k].first) {
        line = line.substr(0, start) + key + " = " + overrides[k].second;
        applied[k] = true;
        break;
      }
    }
  }

  std::string out;
  for(size_t i = 0; i < lines.size(); i++) {
    out += lines[i];
    if(i + 1 < lines.size())
      out += "\n";
  }
  for(size_t k = 0; k < overrides.size(); k++) {
    if(!applied[k]) {
      if(!out.empty() && out.back() != '\n')
        out += "\n";
      out += overrides[k].first + " = " + overrides[k].second + "\n";
    }
  }
  return out;
}
