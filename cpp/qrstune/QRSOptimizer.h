// qrstune/QRSOptimizer.h

#ifndef QRSTUNE_QRSOPTIMIZER_H_
#define QRSTUNE_QRSOPTIMIZER_H_

#include <cstdint>
#include <random>
#include <vector>

namespace QRSTune {

// ============================================================
// Feature map phi(x):
//   [1,  x_0..x_{D-1},  x_0^2..x_{D-1}^2,  x_i*x_j for i<j]
//   Total features F = 1 + D + D*(D+1)/2
// ============================================================

int numFeatures(int D);

// Fill phi[0..F-1] given x[0..D-1].
void computeFeatures(int D, const double* x, double* phi);

double sigmoid(double z);

// Solve Ax = b in-place (A is F x F, b is length F) via partial-pivot
// Gaussian elimination. Returns false if singular. Overwrites A and b.
bool gaussianSolve(int F, std::vector<std::vector<double>>& A, std::vector<double>& b);

// ============================================================
// QRSModel: quadratic logistic regression with L2 regularization.
// Provides MAP estimation and win-probability prediction.
// ============================================================
class QRSModel {
  int D_, F_;
  std::vector<double> beta_;   // F coefficients (intercept, linear, quad, cross)
  double l2_;                  // L2 regularization strength

  // Build the negative Hessian (Fisher info + L2 prior) at current beta.
  void buildNegHessian(const std::vector<std::vector<double>>& xs,
                       std::vector<std::vector<double>>& negH) const;

  // Build the D x D quadratic Hessian M and rhs for the system M*x = -linearCoeffs.
  void buildQuadHessian(std::vector<std::vector<double>>& M,
                        std::vector<double>& rhs) const;

 public:
  QRSModel();
  QRSModel(int D, double l2_reg = 0.1);

  // Newton-Raphson MAP estimation.
  // xs: sample coordinates; ys: outcomes in {0.0, 0.5, 1.0}
  void fit(const std::vector<std::vector<double>>& xs,
           const std::vector<double>& ys,
           int max_iter = 30);

  // Win probability at x[0..D-1]
  double predict(const double* x) const;

  // Linear score phi(x)^T beta (used for MAP maximization)
  double score(const double* x) const;

  // Find x in [-1,+1]^D that maximizes score(x) = phi(x)^T beta.
  // For a quadratic, the unconstrained stationary point satisfies:
  //   M x = -b_lin
  // where M[i][i] = 2*beta_quad[i], M[i][j]=M[j][i] = beta_cross[i,j],
  //       b_lin[i] = beta_linear[i].
  // The solution is clamped to [-1,+1]^D.
  void mapOptimum(double* out_x) const;

  int dims()     const { return D_; }
  int features() const { return F_; }

  // Compute standard errors of the MAP optimum x* via the delta method.
  // Rebuilds the Fisher information matrix from xs at current beta, inverts
  // it to get Cov(beta), then propagates through the beta -> x* mapping.
  // On success, fills se[0..D-1] with SEs in normalized [-1,+1] coords and
  // sets clamped[0..D-1] to true for dims where x* was clamped to boundary.
  // Returns false if the Hessian is singular (CIs unavailable).
  bool computeOptimumSE(const std::vector<std::vector<double>>& xs,
                        double* se,
                        bool* clamped) const;
};

// ============================================================
// QRSBuffer: sample storage with confidence-based pruning.
// Samples whose predicted win rate is far below the current
// MAP estimate are dropped to keep the model locally focused.
// ============================================================
class QRSBuffer {
  std::vector<std::vector<double>> xs_;
  std::vector<double> ys_;
  int min_keep_;        // never prune below this count
  double prune_margin_; // drop samples where p_pred < p_best - margin

 public:
  QRSBuffer(int min_keep = 30, double prune_margin = 0.25);

  void add(const std::vector<double>& x, double y);

  // Remove samples significantly below the current MAP win estimate.
  // Samples are ranked by predicted quality so that min_keep_ retains the
  // best samples rather than the oldest (which are typically from early
  // uniform random exploration).
  void prune(const QRSModel& model);

  const std::vector<std::vector<double>>& xs() const { return xs_; }
  const std::vector<double>&              ys() const { return ys_; }
  int size() const { return (int)xs_.size(); }
};

// ============================================================
// QRSTuner: top-level interface
//
// Usage:
//   QRSTuner tuner(D, seed, numTrials);
//   for each trial:
//     auto x = tuner.nextSample();
//     ... run game, get win=1.0 / loss=0.0 / draw=0.5 ...
//     tuner.addResult(x, outcome);
//   auto best = tuner.bestCoords();
// ============================================================
class QRSTuner {
  int D_;
  QRSModel  model_;
  QRSBuffer buffer_;
  std::mt19937_64 rng_;
  int trial_count_;
  int total_trials_;
  int refit_every_;    // refit model after every N trials
  int prune_every_;    // prune once per this many refits

  // Exploration noise std dev: decays linearly from initial to final
  double sigma_initial_;
  double sigma_final_;

 public:
  // D            : number of dimensions
  // seed         : RNG seed for reproducibility
  // total_trials : expected total number of trials (for scheduling)
  // l2_reg       : L2 regularization for QRSModel (default 0.1)
  // refit_every  : how often to refit model (default 10 trials)
  // prune_every  : prune every N-th refit (default 5)
  QRSTuner(int D, uint64_t seed, int total_trials,
           double l2_reg     = 0.1,
           int refit_every   = 10,
           int prune_every   = 5,
           double sigma_init = 0.40,
           double sigma_fin  = 0.05);

  // Propose next point to evaluate.
  // During early exploration (< F samples) returns a random point.
  // Afterwards: MAP optimum + decaying Gaussian noise clamped to [-1,+1]^D.
  std::vector<double> nextSample();

  // Record the outcome of a trial.
  // y: 1.0 = win, 0.0 = loss, 0.5 = draw
  void addResult(const std::vector<double>& x, double y);

  // Return current MAP optimum in [-1,+1]^D
  std::vector<double> bestCoords() const;

  // Estimated win probability at the MAP optimum
  double bestWinProb() const;

  int trialCount()   const { return trial_count_; }
  int dims()         const { return D_; }
  const QRSModel& model() const { return model_; }
  const QRSBuffer& buffer() const { return buffer_; }
};

void runTests();

}  // namespace QRSTune

#endif  // QRSTUNE_QRSOPTIMIZER_H_
