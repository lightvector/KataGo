#ifndef PROGRAM_HUMANSLTUNER_H_
#define PROGRAM_HUMANSLTUNER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Pure math + round loop for the `tunehuman` subcommand.
// NO KataGo NN/search dependencies — fully unit-testable without model files.

// Binomial logistic regression winrate(x) = sigmoid(b0 + b1*x), fit by
// L2-regularized Newton-MAP. Linear logit because the strength coordinate is monotone.
class LogisticRS {
 public:
  // 400 / ln(10): converts a logit difference to ELO.
  static constexpr double ELO_PER_LOGIT = 400.0 / 2.302585092994046; // ~173.7178

  explicit LogisticRS(double l2_ = 0.5);

  void addSample(double x, double wins, double games); // wins may be fractional (draws = 0.5)
  LogisticRS& fit(int iters = 50);
  double predict(double x) const;               // sigmoid(b0 + b1 x)
  double root(double targetWinrate) const;      // x* with predict(x*) == target; NaN if degenerate
  double rootSeElo(double targetWinrate) const; // delta-method SE of x*, in ELO units; +inf if degenerate
  int distinctXCount(double eps = 1e-6) const;  // number of distinct sampled x values

  double getB0() const { return b0; }
  double getB1() const { return b1; }

 private:
  double l2;
  std::vector<double> xs;
  std::vector<double> ws;
  std::vector<double> ns;
  double b0;
  double b1;
  double cov[2][2]; // covariance of (b0,b1); valid after fit()
  bool fitted;
};

struct StrengthDialParams {
  double piklLambda;
  int    maxVisits;
  double deltaTau;
};

struct StrengthDialConfig {
  double piklFloor = 0.02;
  double piklMax = 1.0e4;
  int    searchVisits = 100; // must be >= 2
  int    maxVisitsCap = 400;
  double dtauMax = 0.6;
  static constexpr double PIKL_INERT = 1.0e9; // KataGo default; "off"
};

// Resolved per-run visit budget, anchored to the baseline config's own maxVisits so the
// candidate never spends MORE compute than the baseline unless the operator explicitly opts in.
struct VisitBudget {
  int  midVisits;            // -> StrengthDialConfig.searchVisits (segment B depth / segment-C low anchor)
  int  maxVisitsCap;         // -> StrengthDialConfig.maxVisitsCap (segment C strong end); always >= midVisits
  bool raisesAboveBaseline;  // true iff baseline has a finite cap and (midVisits > B || maxVisitsCap > B)
  bool flooredFromBelow2;    // true iff a sub-2 mid (incl. a B<2 auto baseline) was bumped up to 2
  int  effectiveBaseline;    // the anchor: baseline cap (clamped to 1e6) when finite, else the legacy 100;
                             // this is the value an explicit -search-visits/-max-visits-cap is judged against
  bool baselineHasCap;       // false when the baseline omits maxVisits (search bounded by time/playouts)
};

// Pure, NN-free. baselineMaxVisits is SearchParams.maxVisits (int64_t; the ctor default 1<<50 means
// "no real cap" -- search is bounded by time/playouts instead). userSearchVisits / userMaxVisitsCap
// use -1 as the "auto" sentinel (anchor to the baseline); any other value is the explicit operator
// override. midVisits is floored to 2 (piklLambda is inert below 2 visits) and maxVisitsCap is clamped
// up to midVisits (so segment C's log2 interpolation never runs downward). A finite-but-absurd baseline
// (> 1e6 yet < 1<<50) is clamped to 1e6 so the int dial fields cannot overflow.
VisitBudget resolveVisitBudget(int64_t baselineMaxVisits, int userSearchVisits, int userMaxVisitsCap);

// Returns the strength-coordinate upper bound to actually calibrate over. When segment C is flat
// (maxVisitsCap == midVisits, the auto outcome), the strong third [2,3] collapses to a single point,
// so calibrating there is meaningless: shrink to 2.0 (only when the original range straddles it,
// i.e. xHi > 2.0 && xLo < 2.0). Otherwise returns xHi unchanged. Pure, NN-free.
double effectiveXHi(const VisitBudget& vb, double xLo, double xHi);

// Maps a scalar strength coordinate x in [0,3] (low=weak, high=strong) to the three dials,
// globally monotone in strength. Clamps x to [0,3].
StrengthDialParams strengthDialToParams(double x, const StrengthDialConfig& c);

struct CalibrationResult {
  double xStar;
  double eloSe;     // 1-sigma CI half-width in ELO at xStar
  int    totalGames;
  int    rounds;
  bool   converged;
  LogisticRS model; // final fitted surface (for reporting fitted ELO)
};

// One played round's outcome: a dial coordinate and its {wins, games} tally. Persisted to disk by the
// `tunehuman` command (one per round) so a calibration that is interrupted -- e.g. the process is killed
// by an environment runtime cap -- can be resumed from where it left off instead of restarting.
struct CalibrationSample {
  double x;
  double wins;  // may be fractional (draws = 0.5)
  double games;
};

// playAt(x) plays a batch at dial x and returns {wins, games}; wins may be fractional.
// onRound(round, xStar, eloSe, distinctXs, totalGames) is optional progress logging.
// initialSamples seeds the fit with prior rounds (for resume): the round loop starts at
// initialSamples.size() and, if those samples already satisfy convergence, returns without playing more.
// onSampleCollected(x, wins, games) fires once per NEWLY played round (not for initialSamples), so the
// caller can durably append each round's outcome to a checkpoint file.
CalibrationResult calibrateToTarget(
  const std::function<std::pair<double,int>(double)>& playAt,
  double xLo, double xHi, double targetWinrate,
  int gamesPerRound, int maxRounds, double eloTol,
  uint64_t rngSeed, double l2 = 0.5,
  const std::function<void(int,double,double,int,int)>& onRound = nullptr,
  const std::vector<CalibrationSample>& initialSamples = std::vector<CalibrationSample>(),
  const std::function<void(double,double,double)>& onSampleCollected = nullptr);

// Rewrites baselineText, replacing the value of each override key on its existing
// non-comment "key = value" line (preserving the key spelling), or appending
// "key = value" at the end if absent. All other lines/comments are left intact.
std::string overrideConfigText(
  const std::string& baselineText,
  const std::vector<std::pair<std::string,std::string>>& overrides);

#endif // PROGRAM_HUMANSLTUNER_H_
