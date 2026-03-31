// qrstune/QRSOptimizer.h

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace QRSTune {

// ============================================================
// Feature map phi(x):
//   [1,  x_0..x_{D-1},  x_0^2..x_{D-1}^2,  x_i*x_j for i<j]
//   Total features F = 1 + D + D*(D+1)/2
// ============================================================

static inline int numFeatures(int D) {
  return 1 + D + D * (D + 1) / 2;
}

// Fill phi[0..F-1] given x[0..D-1].
static inline void computeFeatures(int D, const double* x, double* phi) {
  int k = 0;
  phi[k++] = 1.0;
  for(int i = 0; i < D; i++) phi[k++] = x[i];
  for(int i = 0; i < D; i++) phi[k++] = x[i] * x[i];
  for(int i = 0; i < D; i++)
    for(int j = i + 1; j < D; j++)
      phi[k++] = x[i] * x[j];
}

static inline double sigmoid(double z) {
  if(z > 40.0) return 1.0;
  if(z < -40.0) return 0.0;
  return 1.0 / (1.0 + std::exp(-z));
}

// Solve Ax = b in-place (A is F x F, b is length F) via partial-pivot
// Gaussian elimination. Returns false if singular. Overwrites A and b.
static bool gaussianSolve(int F, std::vector<std::vector<double>>& A, std::vector<double>& b) {
  for(int col = 0; col < F; col++) {
    int piv = col;
    for(int r = col + 1; r < F; r++)
      if(std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
    std::swap(A[col], A[piv]);
    std::swap(b[col], b[piv]);
    if(std::fabs(A[col][col]) < 1e-12) return false;
    double inv = 1.0 / A[col][col];
    for(int r = col + 1; r < F; r++) {
      double f = A[r][col] * inv;
      for(int c = col; c < F; c++) A[r][c] -= f * A[col][c];
      b[r] -= f * b[col];
    }
  }
  for(int r = F - 1; r >= 0; r--) {
    for(int c = r + 1; c < F; c++) b[r] -= A[r][c] * b[c];
    b[r] /= A[r][r];
  }
  return true;
}

// ============================================================
// QRSModel: quadratic logistic regression with L2 regularization.
// Provides MAP estimation and win-probability prediction.
// ============================================================
class QRSModel {
  int D_, F_;
  std::vector<double> beta_;   // F coefficients (intercept, linear, quad, cross)
  double l2_;                  // L2 regularization strength

 public:
  QRSModel() : D_(0), F_(0), l2_(0.1) {}
  QRSModel(int D, double l2_reg = 0.1)
    : D_(D), F_(numFeatures(D)), beta_(numFeatures(D), 0.0), l2_(l2_reg) {}

  // Newton-Raphson MAP estimation.
  // xs: sample coordinates; ys: outcomes in {0.0, 0.5, 1.0}
  void fit(const std::vector<std::vector<double>>& xs,
           const std::vector<double>& ys,
           int max_iter = 30) {
    int N = (int)xs.size();
    if(N < F_) return;  // underdetermined; keep prior beta = 0

    std::vector<double> phi(F_);

    for(int iter = 0; iter < max_iter; iter++) {
      // Gradient and (negative) Hessian from L2 prior
      std::vector<double> grad(F_, 0.0);
      std::vector<std::vector<double>> negH(F_, std::vector<double>(F_, 0.0));
      for(int f = 0; f < F_; f++) {
        grad[f] = -l2_ * beta_[f];
        negH[f][f] = l2_;
      }

      // Data contribution
      for(int n = 0; n < N; n++) {
        computeFeatures(D_, xs[n].data(), phi.data());
        double z = 0.0;
        for(int f = 0; f < F_; f++) z += beta_[f] * phi[f];
        double p = sigmoid(z);
        double w = p * (1.0 - p);
        double resid = ys[n] - p;
        for(int f = 0; f < F_; f++) {
          grad[f] += resid * phi[f];
          for(int g = f; g < F_; g++)
            negH[f][g] += w * phi[f] * phi[g];
        }
      }
      // Symmetrize negH
      for(int f = 0; f < F_; f++)
        for(int g = f + 1; g < F_; g++)
          negH[g][f] = negH[f][g];

      // Solve negH * delta = grad  =>  beta += delta
      if(!gaussianSolve(F_, negH, grad)) break;
      double maxd = 0.0;
      for(int f = 0; f < F_; f++) {
        beta_[f] += grad[f];
        maxd = std::max(maxd, std::fabs(grad[f]));
      }
      if(maxd < 1e-7) break;
    }
  }

  // Win probability at x[0..D-1]
  double predict(const double* x) const {
    std::vector<double> phi(F_);
    computeFeatures(D_, x, phi.data());
    double z = 0.0;
    for(int f = 0; f < F_; f++) z += beta_[f] * phi[f];
    return sigmoid(z);
  }

  // Linear score phi(x)^T beta (used for MAP maximization)
  double score(const double* x) const {
    std::vector<double> phi(F_);
    computeFeatures(D_, x, phi.data());
    double z = 0.0;
    for(int f = 0; f < F_; f++) z += beta_[f] * phi[f];
    return z;
  }

  // Find x in [-1,+1]^D that maximizes score(x) = phi(x)^T beta.
  // For a quadratic, the unconstrained stationary point satisfies:
  //   M x = -b_lin
  // where M[i][i] = 2*beta_quad[i], M[i][j]=M[j][i] = beta_cross[i,j],
  //       b_lin[i] = beta_linear[i].
  // The solution is clamped to [-1,+1]^D.
  void mapOptimum(double* out_x) const {
    // Beta layout: [intercept, linear[0..D-1], quad[0..D-1], cross by (i<j)]
    const double* b_lin  = beta_.data() + 1;
    const double* b_quad = beta_.data() + 1 + D_;
    const double* b_cross = beta_.data() + 1 + 2 * D_;

    std::vector<std::vector<double>> M(D_, std::vector<double>(D_, 0.0));
    std::vector<double> rhs(D_);

    for(int k = 0; k < D_; k++) {
      M[k][k] = 2.0 * b_quad[k];
      rhs[k]  = -b_lin[k];
    }
    int idx = 0;
    for(int i = 0; i < D_; i++)
      for(int j = i + 1; j < D_; j++) {
        M[i][j] += b_cross[idx];
        M[j][i] += b_cross[idx];
        idx++;
      }

    if(!gaussianSolve(D_, M, rhs)) {
      for(int i = 0; i < D_; i++) out_x[i] = 0.0;
      return;
    }
    for(int i = 0; i < D_; i++)
      out_x[i] = std::max(-1.0, std::min(1.0, rhs[i]));
  }

  int dims()     const { return D_; }
  int features() const { return F_; }
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
  QRSBuffer(int min_keep = 30, double prune_margin = 0.25)
    : min_keep_(min_keep), prune_margin_(prune_margin) {}

  void add(const std::vector<double>& x, double y) {
    xs_.push_back(x);
    ys_.push_back(y);
  }

  // Remove samples significantly below the current MAP win estimate.
  void prune(const QRSModel& model) {
    int N = (int)xs_.size();
    if(N <= min_keep_ * 2) return;

    // Best predicted win rate across all stored samples
    double p_best = 0.0;
    for(int i = 0; i < N; i++) {
      double p = model.predict(xs_[i].data());
      if(p > p_best) p_best = p;
    }
    double threshold = p_best - prune_margin_;

    std::vector<std::vector<double>> nx;
    std::vector<double> ny;
    for(int i = 0; i < N; i++) {
      double p = model.predict(xs_[i].data());
      if(p >= threshold || (int)nx.size() < min_keep_) {
        nx.push_back(xs_[i]);
        ny.push_back(ys_[i]);
      }
    }
    xs_ = std::move(nx);
    ys_ = std::move(ny);
  }

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
           double sigma_fin  = 0.05)
    : D_(D),
      model_(D, l2_reg),
      buffer_(/*min_keep=*/std::max(20, total_trials / 50),
              /*prune_margin=*/0.25),
      rng_(seed),
      trial_count_(0),
      total_trials_(total_trials),
      refit_every_(refit_every),
      prune_every_(prune_every),
      sigma_initial_(sigma_init),
      sigma_final_(sigma_fin) {}

  // Propose next point to evaluate.
  // During early exploration (< F samples) returns a random point.
  // Afterwards: MAP optimum + decaying Gaussian noise clamped to [-1,+1]^D.
  std::vector<double> nextSample() {
    std::vector<double> x(D_);
    int F = model_.features();

    if(buffer_.size() < F + 1) {
      // Insufficient data for reliable fit — explore uniformly
      std::uniform_real_distribution<double> uni(-1.0, 1.0);
      for(int i = 0; i < D_; i++) x[i] = uni(rng_);
      return x;
    }

    // Base: MAP optimum
    model_.mapOptimum(x.data());

    // Decaying exploration noise
    double progress = (double)trial_count_ / std::max(1, total_trials_ - 1);
    double sigma = sigma_initial_ + progress * (sigma_final_ - sigma_initial_);
    std::normal_distribution<double> noise(0.0, sigma);
    for(int i = 0; i < D_; i++)
      x[i] = std::max(-1.0, std::min(1.0, x[i] + noise(rng_)));

    return x;
  }

  // Record the outcome of a trial.
  // y: 1.0 = win, 0.0 = loss, 0.5 = draw
  void addResult(const std::vector<double>& x, double y) {
    buffer_.add(x, y);
    trial_count_++;

    if(trial_count_ % refit_every_ == 0 && buffer_.size() >= model_.features() + 1) {
      model_.fit(buffer_.xs(), buffer_.ys());
      int refit_count = trial_count_ / refit_every_;
      if(refit_count % prune_every_ == 0)
        buffer_.prune(model_);
    }
  }

  // Return current MAP optimum in [-1,+1]^D
  std::vector<double> bestCoords() const {
    std::vector<double> best(D_);
    model_.mapOptimum(best.data());
    return best;
  }

  // Estimated win probability at the MAP optimum
  double bestWinProb() const {
    auto best = bestCoords();
    return model_.predict(best.data());
  }

  int trialCount()   const { return trial_count_; }
  int dims()         const { return D_; }
  const QRSModel& model() const { return model_; }
};

}  // namespace QRSTune
