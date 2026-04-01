// qrstune/QRSOptimizer.cpp
//
// QRS-Tune: Quadratic Response Surface optimizer for binary-outcome tuning.
//
// Models win probability as sigmoid(phi(x)^T * beta) where phi(x) is a
// quadratic feature map: [1, x_i, x_i^2, x_i*x_j]. The model is fit via
// Newton-Raphson MAP estimation with L2 regularization. The MAP optimum
// of the fitted quadratic surface is used as the next evaluation point,
// with decaying Gaussian noise for exploration.

#include "../qrstune/QRSOptimizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "../core/test.h"

using namespace std;

// Pivot values smaller than this are treated as singular
static const double SINGULAR_THRESHOLD = 1e-12;

// Newton-Raphson stops when the largest coefficient change is below this
static const double CONVERGENCE_THRESHOLD = 1e-7;

// Sigmoid is clamped to 0 or 1 beyond this magnitude to avoid overflow
static const double SIGMOID_CLAMP = 40.0;

// ============================================================
// Free functions
// ============================================================

int QRSTune::numFeatures(int D) {
  return 1 + D + D * (D + 1) / 2;
}

// Feature layout for D dimensions:
//   [intercept(1), linear(D), quadratic(D), cross-terms(D*(D-1)/2)]
// Example for D=2, x=[a,b]: phi = [1, a, b, a^2, b^2, a*b]
void QRSTune::computeFeatures(int D, const double* x, double* phi) {
  int k = 0;
  phi[k++] = 1.0;
  for(int i = 0; i < D; i++) phi[k++] = x[i];
  for(int i = 0; i < D; i++) phi[k++] = x[i] * x[i];
  for(int i = 0; i < D; i++)
    for(int j = i + 1; j < D; j++)
      phi[k++] = x[i] * x[j];
}

double QRSTune::sigmoid(double z) {
  if(z > SIGMOID_CLAMP) return 1.0;
  if(z < -SIGMOID_CLAMP) return 0.0;
  return 1.0 / (1.0 + exp(-z));
}

// Partial-pivot Gaussian elimination: solves Ax = b in-place.
// On return, b contains the solution x. A is destroyed.
// Returns false if A is singular (pivot below SINGULAR_THRESHOLD).
bool QRSTune::gaussianSolve(int F, vector<vector<double>>& A, vector<double>& b) {
  // Forward elimination with partial pivoting
  for(int col = 0; col < F; col++) {
    int piv = col;
    for(int r = col + 1; r < F; r++)
      if(fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
    swap(A[col], A[piv]);
    swap(b[col], b[piv]);
    if(fabs(A[col][col]) < SINGULAR_THRESHOLD) return false;
    double inv = 1.0 / A[col][col];
    for(int r = col + 1; r < F; r++) {
      double mult = A[r][col] * inv;
      for(int c = col; c < F; c++) A[r][c] -= mult * A[col][c];
      b[r] -= mult * b[col];
    }
  }
  // Back substitution
  for(int r = F - 1; r >= 0; r--) {
    for(int c = r + 1; c < F; c++) b[r] -= A[r][c] * b[c];
    b[r] /= A[r][r];
  }
  return true;
}

// ============================================================
// QRSModel
// ============================================================

QRSTune::QRSModel::QRSModel()
  :D_(0),
   F_(0),
   l2_(0.1)
{}

QRSTune::QRSModel::QRSModel(int D, double l2_reg)
  :D_(D),
   F_(numFeatures(D)),
   beta_(numFeatures(D), 0.0),
   l2_(l2_reg)
{}

// Newton-Raphson MAP estimation for L2-regularized quadratic logistic regression.
//
// Maximizes: sum_n [ y_n * log(p_n) + (1-y_n) * log(1-p_n) ] - (l2/2) * ||beta||^2
// where p_n = sigmoid(phi(x_n)^T * beta).
//
// Each iteration:
//   1. Compute gradient and negative Hessian (including L2 prior)
//   2. Solve the Newton system: negH * delta = grad
//   3. Update: beta += delta
//   4. Stop when max |delta_f| < CONVERGENCE_THRESHOLD
void QRSTune::QRSModel::fit(const vector<vector<double>>& xs,
                             const vector<double>& ys,
                             int max_iter) {
  int N = (int)xs.size();
  if(N < F_) return;  // underdetermined; keep prior beta = 0

  vector<double> phi(F_);
  vector<double> grad(F_);
  vector<vector<double>> negH(F_, vector<double>(F_));

  for(int iter = 0; iter < max_iter; iter++) {
    // Initialize with L2 prior contribution: grad = -l2*beta, negH = l2*I
    fill(grad.begin(), grad.end(), 0.0);
    for(int f = 0; f < F_; f++) fill(negH[f].begin(), negH[f].end(), 0.0);
    for(int f = 0; f < F_; f++) {
      grad[f] = -l2_ * beta_[f];
      negH[f][f] = l2_;
    }

    // Accumulate data likelihood: grad += (y-p)*phi, negH += p*(1-p)*phi*phi^T
    for(int n = 0; n < N; n++) {
      computeFeatures(D_, xs[n].data(), phi.data());
      double logit = 0.0;
      for(int f = 0; f < F_; f++) logit += beta_[f] * phi[f];
      double p = sigmoid(logit);
      double hessianWeight = p * (1.0 - p);
      double residual = ys[n] - p;
      for(int f = 0; f < F_; f++) {
        grad[f] += residual * phi[f];
        for(int g = f; g < F_; g++)
          negH[f][g] += hessianWeight * phi[f] * phi[g];
      }
    }
    // Symmetrize: negH is only filled for g >= f above
    for(int f = 0; f < F_; f++)
      for(int g = f + 1; g < F_; g++)
        negH[g][f] = negH[f][g];

    // Solve Newton step: negH * delta = grad (grad is overwritten with delta)
    if(!gaussianSolve(F_, negH, grad)) break;

    // Apply step and check convergence
    double maxStep = 0.0;
    for(int f = 0; f < F_; f++) {
      beta_[f] += grad[f];
      maxStep = max(maxStep, fabs(grad[f]));
    }
    if(maxStep < CONVERGENCE_THRESHOLD) break;
  }
}

double QRSTune::QRSModel::predict(const double* x) const {
  return sigmoid(score(x));
}

double QRSTune::QRSModel::score(const double* x) const {
  vector<double> phi(F_);
  computeFeatures(D_, x, phi.data());
  double logit = 0.0;
  for(int f = 0; f < F_; f++) logit += beta_[f] * phi[f];
  return logit;
}

// Find the unconstrained optimum of the quadratic score surface, then clamp to [-1,+1]^D.
//
// Beta layout: [intercept, linear[0..D-1], quadratic[0..D-1], cross[i<j]]
// The quadratic surface gradient is: M*x + linearCoeffs = 0
// where M[i][i] = 2*quadCoeffs[i], M[i][j] = crossCoeffs[pair(i,j)]
void QRSTune::QRSModel::mapOptimum(double* out_x) const {
  const double* linearCoeffs = beta_.data() + 1;
  const double* quadCoeffs   = beta_.data() + 1 + D_;
  const double* crossCoeffs  = beta_.data() + 1 + 2 * D_;

  // Build the Hessian matrix M and right-hand side for M*x = -linearCoeffs
  vector<vector<double>> M(D_, vector<double>(D_, 0.0));
  vector<double> rhs(D_);

  for(int k = 0; k < D_; k++) {
    M[k][k] = 2.0 * quadCoeffs[k];
    rhs[k]  = -linearCoeffs[k];
  }
  int idx = 0;
  for(int i = 0; i < D_; i++)
    for(int j = i + 1; j < D_; j++) {
      M[i][j] += crossCoeffs[idx];
      M[j][i] += crossCoeffs[idx];
      idx++;
    }

  if(!gaussianSolve(D_, M, rhs)) {
    for(int i = 0; i < D_; i++) out_x[i] = 0.0;
    return;
  }
  // Clamp to the normalized coordinate range [-1, +1]
  for(int i = 0; i < D_; i++)
    out_x[i] = max(-1.0, min(1.0, rhs[i]));
}

// ============================================================
// QRSBuffer
// ============================================================

QRSTune::QRSBuffer::QRSBuffer(int min_keep, double prune_margin)
  :min_keep_(min_keep),
   prune_margin_(prune_margin)
{}

void QRSTune::QRSBuffer::add(const vector<double>& x, double y) {
  xs_.push_back(x);
  ys_.push_back(y);
}

// Confidence-based pruning: drop samples whose predicted win rate
// is more than prune_margin_ below the best predicted win rate.
// Samples are ranked so that min_keep_ retains the highest-quality
// samples (not just the oldest).
void QRSTune::QRSBuffer::prune(const QRSModel& model) {
  int N = (int)xs_.size();
  if(N <= min_keep_ * 2) return;

  // Score all samples and find best predicted win rate
  vector<pair<double, int>> scored(N);  // (predicted winrate, original index)
  double bestPrediction = 0.0;
  for(int i = 0; i < N; i++) {
    double p = model.predict(xs_[i].data());
    scored[i] = {p, i};
    if(p > bestPrediction) bestPrediction = p;
  }
  double threshold = bestPrediction - prune_margin_;

  // Sort by descending predicted quality so min_keep_ retains the best
  sort(scored.begin(), scored.end(),
    [](const pair<double,int>& a, const pair<double,int>& b) {
      return a.first > b.first;
    });

  // Mark samples to keep: above threshold, or among top min_keep_
  vector<bool> keep(N, false);
  int kept = 0;
  for(auto& entry : scored) {
    if(entry.first >= threshold || kept < min_keep_) {
      keep[entry.second] = true;
      kept++;
    }
  }

  // Rebuild in original order to preserve temporal structure
  vector<vector<double>> newXs;
  vector<double> newYs;
  for(int i = 0; i < N; i++) {
    if(keep[i]) {
      newXs.push_back(std::move(xs_[i]));
      newYs.push_back(ys_[i]);
    }
  }
  xs_ = std::move(newXs);
  ys_ = std::move(newYs);
}

// ============================================================
// QRSTuner
// ============================================================

QRSTune::QRSTuner::QRSTuner(int D, uint64_t seed, int total_trials,
                             double l2_reg, int refit_every, int prune_every,
                             double sigma_init, double sigma_fin)
  :D_(D),
   model_(D, l2_reg),
   buffer_(max(20, total_trials / 50), 0.25),
   rng_(seed),
   trial_count_(0),
   total_trials_(total_trials),
   refit_every_(refit_every),
   prune_every_(prune_every),
   sigma_initial_(sigma_init),
   sigma_final_(sigma_fin)
{}

vector<double> QRSTune::QRSTuner::nextSample() {
  vector<double> x(D_);
  int F = model_.features();

  if(buffer_.size() < F + 1) {
    // Insufficient data for reliable fit — explore uniformly
    uniform_real_distribution<double> uni(-1.0, 1.0);
    for(int i = 0; i < D_; i++) x[i] = uni(rng_);
    return x;
  }

  // Start from MAP optimum, add decaying Gaussian noise for exploration
  model_.mapOptimum(x.data());
  double progress = (double)trial_count_ / max(1, total_trials_ - 1);
  double sigma = sigma_initial_ + progress * (sigma_final_ - sigma_initial_);
  normal_distribution<double> noise(0.0, sigma);
  for(int i = 0; i < D_; i++)
    x[i] = max(-1.0, min(1.0, x[i] + noise(rng_)));

  return x;
}

void QRSTune::QRSTuner::addResult(const vector<double>& x, double y) {
  buffer_.add(x, y);
  trial_count_++;

  if(trial_count_ % refit_every_ == 0 && buffer_.size() >= model_.features() + 1) {
    model_.fit(buffer_.xs(), buffer_.ys());
    int refit_count = trial_count_ / refit_every_;
    if(refit_count % prune_every_ == 0)
      buffer_.prune(model_);
  }
}

vector<double> QRSTune::QRSTuner::bestCoords() const {
  vector<double> best(D_);
  model_.mapOptimum(best.data());
  return best;
}

double QRSTune::QRSTuner::bestWinProb() const {
  auto best = bestCoords();
  return model_.predict(best.data());
}

// ============================================================
// Tests
// ============================================================

static bool approxEqual(double x, double y, double tolerance) {
  return fabs(x - y) < tolerance;
}

void QRSTune::runTests() {
  cout << "Running QRSTune tests" << endl;

  // Test numFeatures: F = 1 + D + D*(D+1)/2
  // D=0: 1, D=1: 3, D=2: 6, D=3: 10
  {
    testAssert(numFeatures(0) == 1);
    testAssert(numFeatures(1) == 3);
    testAssert(numFeatures(2) == 6);
    testAssert(numFeatures(3) == 10);
  }

  // Test computeFeatures: D=2, x=[0.5, -0.3]
  // Expected: [1.0, 0.5, -0.3, 0.25, 0.09, -0.15]
  {
    double x[2] = {0.5, -0.3};
    double phi[6];
    computeFeatures(2, x, phi);
    testAssert(approxEqual(phi[0], 1.0, 1e-15));
    testAssert(approxEqual(phi[1], 0.5, 1e-15));
    testAssert(approxEqual(phi[2], -0.3, 1e-15));
    testAssert(approxEqual(phi[3], 0.25, 1e-15));
    testAssert(approxEqual(phi[4], 0.09, 1e-15));
    testAssert(approxEqual(phi[5], -0.15, 1e-15));
  }

  // Test sigmoid
  {
    testAssert(approxEqual(sigmoid(0.0), 0.5, 1e-15));
    testAssert(sigmoid(50.0) == 1.0);
    testAssert(sigmoid(-50.0) == 0.0);
    testAssert(approxEqual(sigmoid(1.0), 1.0 / (1.0 + exp(-1.0)), 1e-12));
    // Beyond clamp threshold: exactly 0 or 1
    testAssert(sigmoid(SIGMOID_CLAMP + 1.0) == 1.0);
    testAssert(sigmoid(-SIGMOID_CLAMP - 1.0) == 0.0);
    // Moderate values: still fractional
    testAssert(sigmoid(5.0) < 1.0);
    testAssert(sigmoid(-5.0) > 0.0);
  }

  // Test gaussianSolve: 2x2 system [[2,1],[1,3]] * x = [5,7] => x = [8/5, 9/5]
  {
    vector<vector<double>> A = {{2.0, 1.0}, {1.0, 3.0}};
    vector<double> b = {5.0, 7.0};
    bool ok = gaussianSolve(2, A, b);
    testAssert(ok);
    testAssert(approxEqual(b[0], 8.0 / 5.0, 1e-12));
    testAssert(approxEqual(b[1], 9.0 / 5.0, 1e-12));
  }

  // Test gaussianSolve: 3x3 identity system
  {
    vector<vector<double>> A = {{1,0,0},{0,1,0},{0,0,1}};
    vector<double> b = {3.0, -1.0, 7.0};
    bool ok = gaussianSolve(3, A, b);
    testAssert(ok);
    testAssert(approxEqual(b[0], 3.0, 1e-15));
    testAssert(approxEqual(b[1], -1.0, 1e-15));
    testAssert(approxEqual(b[2], 7.0, 1e-15));
  }

  // Test gaussianSolve: singular matrix returns false
  {
    vector<vector<double>> A = {{1.0, 2.0}, {2.0, 4.0}};
    vector<double> b = {3.0, 6.0};
    bool ok = gaussianSolve(2, A, b);
    testAssert(!ok);
  }

  // Test QRSModel fit + predict: 1D separable data
  // All samples at x=+0.8 win, all at x=-0.8 lose.
  // After fitting, predict(+0.8) should be high and predict(-0.8) should be low.
  {
    QRSModel model(1, 0.1);
    vector<vector<double>> xs;
    vector<double> ys;
    for(int i = 0; i < 20; i++) {
      xs.push_back({0.8});
      ys.push_back(1.0);
      xs.push_back({-0.8});
      ys.push_back(0.0);
    }
    model.fit(xs, ys);
    double xWin[] = {0.8};
    double xLose[] = {-0.8};
    double xMid[] = {0.0};
    double pWin = model.predict(xWin);
    double pLose = model.predict(xLose);
    testAssert(pWin > 0.7);
    testAssert(pLose < 0.3);
    // Midpoint should be near 0.5
    double pMid = model.predict(xMid);
    testAssert(approxEqual(pMid, 0.5, 0.15));
  }

  // Test QRSModel mapOptimum: after fitting 1D win-at-positive data,
  // the MAP optimum should be in the positive region (clamped to [−1,+1]).
  {
    QRSModel model(1, 0.1);
    vector<vector<double>> xs;
    vector<double> ys;
    for(int i = 0; i < 20; i++) {
      xs.push_back({0.8});
      ys.push_back(1.0);
      xs.push_back({-0.8});
      ys.push_back(0.0);
    }
    model.fit(xs, ys);
    double bestX;
    model.mapOptimum(&bestX);
    // The optimum should have a higher predicted win rate than the anti-optimum
    double negOne = -1.0;
    testAssert(model.predict(&bestX) > model.predict(&negOne) + 0.1);
  }

  // Test QRSModel 2D: wins cluster at (+0.5, +0.5), losses at (-0.5, -0.5)
  {
    QRSModel model(2, 0.1);
    vector<vector<double>> xs;
    vector<double> ys;
    for(int i = 0; i < 20; i++) {
      xs.push_back({0.5, 0.5});
      ys.push_back(1.0);
      xs.push_back({-0.5, -0.5});
      ys.push_back(0.0);
    }
    model.fit(xs, ys);
    double xWin[] = {0.5, 0.5};
    double xLose[] = {-0.5, -0.5};
    double pWin = model.predict(xWin);
    double pLose = model.predict(xLose);
    testAssert(pWin > 0.7);
    testAssert(pLose < 0.3);
  }

  // Test QRSTuner end-to-end: 1D, deterministic seed, outcome strongly correlated
  // with x > 0. After enough trials the best predicted win rate should exceed 0.5.
  {
    const int numTrials = 100;
    QRSTuner tuner(1, /*seed=*/42, numTrials,
                   /*l2_reg=*/0.1, /*refit_every=*/10, /*prune_every=*/5);
    for(int trial = 0; trial < numTrials; trial++) {
      vector<double> sample = tuner.nextSample();
      // Strong signal: win when x > 0, lose when x < 0
      double outcome = (sample[0] > 0.0) ? 1.0 : 0.0;
      tuner.addResult(sample, outcome);
    }
    testAssert(tuner.trialCount() == numTrials);
    // The fitted model should recognize that positive x is better
    testAssert(tuner.bestWinProb() > 0.5);
  }

  // Test QRSBuffer prune: verify pruning reduces buffer size
  {
    QRSModel model(1, 0.1);
    vector<vector<double>> xs;
    vector<double> ys;
    // Build data with a clear win region
    for(int i = 0; i < 30; i++) {
      xs.push_back({0.8});
      ys.push_back(1.0);
      xs.push_back({-0.8});
      ys.push_back(0.0);
    }
    model.fit(xs, ys);

    QRSBuffer buffer(5, 0.10);  // tight margin, keep at least 5
    for(int i = 0; i < 60; i++) {
      buffer.add(xs[i], ys[i]);
    }
    testAssert(buffer.size() == 60);
    buffer.prune(model);
    // Should have pruned some low-quality samples
    testAssert(buffer.size() < 60);
    testAssert(buffer.size() >= 5);  // min_keep
  }
}
