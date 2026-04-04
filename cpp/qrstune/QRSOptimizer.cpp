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

#include "../core/global.h"
#include "../core/logger.h"
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

// Build the negative Hessian (Fisher info + L2 prior) at current beta.
// negH must be pre-sized to F_ x F_; contents are overwritten.
void QRSTune::QRSModel::buildNegHessian(const vector<vector<double>>& xs,
                                         vector<vector<double>>& negH) const {
  int N = (int)xs.size();
  for(int f = 0; f < F_; f++) fill(negH[f].begin(), negH[f].end(), 0.0);
  for(int f = 0; f < F_; f++)
    negH[f][f] = l2_;

  vector<double> phi(F_);
  for(int n = 0; n < N; n++) {
    computeFeatures(D_, xs[n].data(), phi.data());
    double logit = 0.0;
    for(int f = 0; f < F_; f++) logit += beta_[f] * phi[f];
    double p = sigmoid(logit);
    double w = p * (1.0 - p);
    for(int f = 0; f < F_; f++)
      for(int g = f; g < F_; g++)
        negH[f][g] += w * phi[f] * phi[g];
  }
  for(int f = 0; f < F_; f++)
    for(int g = f + 1; g < F_; g++)
      negH[g][f] = negH[f][g];
}

// Build the D x D quadratic Hessian M and rhs for M*x = -linearCoeffs.
// M and rhs must be pre-sized; contents are overwritten.
void QRSTune::QRSModel::buildQuadHessian(vector<vector<double>>& M,
                                          vector<double>& rhs) const {
  const double* linearCoeffs = beta_.data() + 1;
  const double* quadCoeffs   = beta_.data() + 1 + D_;
  const double* crossCoeffs  = beta_.data() + 1 + 2 * D_;

  for(int k = 0; k < D_; k++) {
    fill(M[k].begin(), M[k].end(), 0.0);
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
}

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

  // Reset to prior mean to avoid warm-start saturation cascade: a
  // previously-extreme intercept makes w = p*(1-p) ≈ 0 for all samples,
  // degenerating the Hessian to l2_*I and producing unbounded Newton steps.
  fill(beta_.begin(), beta_.end(), 0.0);

  vector<double> phi(F_);
  vector<double> grad(F_);
  vector<vector<double>> negH(F_, vector<double>(F_));

  for(int iter = 0; iter < max_iter; iter++) {
    // Build negH and compute gradient simultaneously
    fill(grad.begin(), grad.end(), 0.0);
    for(int f = 0; f < F_; f++) fill(negH[f].begin(), negH[f].end(), 0.0);
    for(int f = 0; f < F_; f++) {
      grad[f] = -l2_ * beta_[f];
      negH[f][f] = l2_;
    }

    for(int n = 0; n < N; n++) {
      computeFeatures(D_, xs[n].data(), phi.data());
      double logit = 0.0;
      for(int f = 0; f < F_; f++) logit += beta_[f] * phi[f];
      double p = sigmoid(logit);
      double w = p * (1.0 - p);
      double residual = ys[n] - p;
      for(int f = 0; f < F_; f++) {
        grad[f] += residual * phi[f];
        for(int g = f; g < F_; g++)
          negH[f][g] += w * phi[f] * phi[g];
      }
    }
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
void QRSTune::QRSModel::mapOptimum(double* out_x) const {
  // If any dimension is convex, the fit is dominated by noise in that
  // dimension.  Return the origin — a conservative, prior-centered choice.
  if(hasConvexDim()) {
    for(int i = 0; i < D_; i++) out_x[i] = 0.0;
    return;
  }

  vector<vector<double>> M(D_, vector<double>(D_, 0.0));
  vector<double> rhs(D_);
  buildQuadHessian(M, rhs);

  if(!gaussianSolve(D_, M, rhs)) {
    for(int i = 0; i < D_; i++) out_x[i] = 0.0;
    return;
  }
  for(int i = 0; i < D_; i++)
    out_x[i] = max(-1.0, min(1.0, rhs[i]));
}

bool QRSTune::QRSModel::hasConvexDim() const {
  const double* quadCoeffs = beta_.data() + 1 + D_;
  for(int d = 0; d < D_; d++)
    if(quadCoeffs[d] >= 0.0) return true;
  return false;
}

// Compute standard errors of the MAP optimum via the delta method.
//
// 1. Rebuild negH (Fisher info + L2 prior) at current beta.
// 2. Invert negH -> Cov(beta).
// 3. Compute unconstrained optimum x* and M^{-1}.
// 4. Build Jacobian J = dx*/dbeta via implicit differentiation.
// 5. Cov(x*) = J * Cov(beta) * J^T.
// 6. SE[d] = sqrt(Cov(x*)[d][d]).
bool QRSTune::QRSModel::computeOptimumSE(const vector<vector<double>>& xs,
                                          double* se,
                                          bool* clamped) const {
  int N = (int)xs.size();
  if(N < F_) return false;

  // If any dim is convex, mapOptimum returns origin — CIs for the
  // unconstrained critical point would be misleading.
  if(hasConvexDim()) return false;

  // --- Step 1: Build negH (Fisher info + L2 prior) at current beta ---
  vector<vector<double>> negH(F_, vector<double>(F_, 0.0));
  buildNegHessian(xs, negH);

  // --- Step 2: Invert negH -> Cov(beta), column by column ---
  vector<vector<double>> covBeta(F_, vector<double>(F_, 0.0));
  for(int g = 0; g < F_; g++) {
    auto negH_copy = negH;
    vector<double> e(F_, 0.0);
    e[g] = 1.0;
    if(!gaussianSolve(F_, negH_copy, e)) return false;
    for(int f = 0; f < F_; f++)
      covBeta[f][g] = e[f];
  }

  // --- Step 3: Compute unconstrained optimum x* and M^{-1} ---
  vector<vector<double>> M(D_, vector<double>(D_, 0.0));
  vector<double> rhs(D_);
  buildQuadHessian(M, rhs);

  // Save M for Jacobian computation before solve destroys it
  auto M_saved = M;
  if(!gaussianSolve(D_, M, rhs)) return false;
  vector<double> xStar(rhs);

  for(int d = 0; d < D_; d++)
    clamped[d] = (fabs(xStar[d]) >= 1.0 - 1e-9);

  // Compute M^{-1} column by column
  vector<vector<double>> Minv(D_, vector<double>(D_, 0.0));
  for(int g = 0; g < D_; g++) {
    auto M_copy = M_saved;
    vector<double> e(D_, 0.0);
    e[g] = 1.0;
    if(!gaussianSolve(D_, M_copy, e)) return false;
    for(int d = 0; d < D_; d++)
      Minv[d][g] = e[d];
  }

  // --- Step 4: Build Jacobian J (D x F) via implicit differentiation ---
  vector<vector<double>> J(D_, vector<double>(F_, 0.0));

  // Linear coefficients: J[:, 1+i] = -Minv[:, i]
  for(int i = 0; i < D_; i++)
    for(int d = 0; d < D_; d++)
      J[d][1 + i] = -Minv[d][i];

  // Quadratic diagonal coefficients: J[:, 1+D+i] = -2 x*_i Minv[:, i]
  for(int i = 0; i < D_; i++)
    for(int d = 0; d < D_; d++)
      J[d][1 + D_ + i] = -2.0 * xStar[i] * Minv[d][i];

  // Cross-term coefficients: J[:, f] = -(x*_j Minv[:, i] + x*_i Minv[:, j])
  int idx = 0;
  for(int i = 0; i < D_; i++)
    for(int j = i + 1; j < D_; j++) {
      for(int d = 0; d < D_; d++)
        J[d][1 + 2 * D_ + idx] = -(xStar[j] * Minv[d][i] + xStar[i] * Minv[d][j]);
      idx++;
    }

  // Clamped dims: x*[d] is at boundary, so dx*_d/dbeta = 0
  for(int d = 0; d < D_; d++)
    if(clamped[d])
      fill(J[d].begin(), J[d].end(), 0.0);

  // --- Step 5: Cov(x*) = J Cov(beta) J^T ---
  vector<vector<double>> temp(D_, vector<double>(F_, 0.0));
  for(int d = 0; d < D_; d++)
    for(int g = 0; g < F_; g++)
      for(int f = 0; f < F_; f++)
        temp[d][g] += J[d][f] * covBeta[f][g];

  vector<vector<double>> covX(D_, vector<double>(D_, 0.0));
  for(int d1 = 0; d1 < D_; d1++)
    for(int d2 = 0; d2 < D_; d2++)
      for(int f = 0; f < F_; f++)
        covX[d1][d2] += temp[d1][f] * J[d2][f];

  // --- Step 6: Extract SEs ---
  for(int d = 0; d < D_; d++) {
    se[d] = covX[d][d] > 0.0 ? sqrt(covX[d][d]) : 0.0;
  }

  return true;
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
// The min_keep_ guard retains the oldest samples (in insertion order),
// preserving spatial diversity from early uniform exploration.
void QRSTune::QRSBuffer::prune(const QRSModel& model) {
  int N = (int)xs_.size();
  if(N <= min_keep_ * 2) return;

  double bestPrediction = 0.0;
  vector<double> preds(N);
  for(int i = 0; i < N; i++) {
    preds[i] = model.predict(xs_[i].data());
    if(preds[i] > bestPrediction) bestPrediction = preds[i];
  }
  double threshold = bestPrediction - prune_margin_;

  vector<vector<double>> newXs;
  vector<double> newYs;
  for(int i = 0; i < N; i++) {
    if(preds[i] >= threshold || (int)newXs.size() < min_keep_) {
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
   sigma_final_(sigma_fin),
   logger_(nullptr)
{}

void QRSTune::QRSTuner::setLogger(Logger* logger) {
  logger_ = logger;
}

vector<double> QRSTune::QRSTuner::nextSample() {
  vector<double> x(D_);
  int F = model_.features();
  string sigmaStr;

  if(buffer_.size() < F + 1) {
    // Insufficient data for reliable fit — explore uniformly
    uniform_real_distribution<double> uni(-1.0, 1.0);
    for(int i = 0; i < D_; i++) x[i] = uni(rng_);
    sigmaStr = "uniform";
  } else {
    // Start from MAP optimum, add decaying Gaussian noise for exploration
    model_.mapOptimum(x.data());
    double progress = (double)trial_count_ / max(1, total_trials_ - 1);
    double sigma = sigma_initial_ + progress * (sigma_final_ - sigma_initial_);

    // When the fit has convex dimensions (noise-dominated), keep exploration
    // wide to avoid premature convergence around the unreliable origin.
    if(model_.hasConvexDim())
      sigma = sigma_initial_;

    normal_distribution<double> noise(0.0, sigma);
    for(int i = 0; i < D_; i++)
      x[i] = max(-1.0, min(1.0, x[i] + noise(rng_)));
    sigmaStr = Global::strprintf("%.4f", sigma);
  }

  if(logger_) {
    string msg = "QRS sample trial=" + to_string(trial_count_) + " sigma=" + sigmaStr + " x=[";
    for(int i = 0; i < D_; i++) {
      if(i > 0) msg += ",";
      msg += Global::strprintf("%.3f", x[i]);
    }
    msg += "]";
    pendingLogMsg_ = msg;
  }

  return x;
}

void QRSTune::QRSTuner::addResult(const vector<double>& x, double y) {
  if(!pendingLogMsg_.empty() && logger_) {
    string label;
    if(y == 1.0)      label = "exp wins";
    else if(y == 0.0) label = "exp loses";
    else               label = "draw";
    logger_->write(pendingLogMsg_ + " -> " + label);
    pendingLogMsg_.clear();
  }
  buffer_.add(x, y);
  trial_count_++;

  if(trial_count_ % refit_every_ == 0 && buffer_.size() >= model_.features() + 1) {
    int sizeBefore = buffer_.size();
    model_.fit(buffer_.xs(), buffer_.ys());
    int refit_count = trial_count_ / refit_every_;
    if(refit_count % prune_every_ == 0) {
      if(model_.hasConvexDim()) {
        if(logger_)
          logger_->write("QRS prune skipped: model has convex dims, predictions unreliable");
      } else {
        buffer_.prune(model_);
        if(logger_)
          logger_->write("QRS prune: " + to_string(sizeBefore) + " -> " + to_string(buffer_.size()) + " samples");
      }
    }
    if(logger_) {
      auto best = bestCoords();
      double winP = model_.predict(best.data());
      const auto& b = model_.beta();
      string diag = "QRS refit trial=" + to_string(trial_count_);
      diag += " buf=" + to_string(buffer_.size());
      diag += " intercept=" + Global::strprintf("%.4f", b[0]);
      diag += " quadDiag=[";
      for(int d = 0; d < D_; d++) {
        if(d > 0) diag += ",";
        diag += Global::strprintf("%.4f", b[1 + D_ + d]);
      }
      diag += "]";
      diag += " concave=" + string(model_.hasConvexDim() ? "N" : "Y");
      diag += " bestQRS=[";
      for(int d = 0; d < D_; d++) {
        if(d > 0) diag += ",";
        diag += Global::strprintf("%.3f", best[d]);
      }
      diag += "]";
      diag += " winP=" + Global::strprintf("%.4f", winP);
      logger_->write(diag);
    }
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
    double posOne = 1.0, negOne = -1.0;
    testAssert(tuner.model().predict(&posOne) > tuner.model().predict(&negOne));
  }

  // Test computeOptimumSE: 1D with a concave peak (wins near center, losses at edges).
  // This gives a negative quadratic coefficient, making M invertible.
  {
    QRSModel model(1, 0.01);
    vector<vector<double>> xs;
    vector<double> ys;
    for(int i = 0; i < 40; i++) {
      xs.push_back({0.0});  ys.push_back(1.0);   // center: wins
      xs.push_back({0.8});  ys.push_back(0.0);   // right edge: losses
      xs.push_back({-0.8}); ys.push_back(0.0);   // left edge: losses
    }
    model.fit(xs, ys);
    double se[1];
    bool clamped[1];
    bool ok = model.computeOptimumSE(xs, se, clamped);
    testAssert(ok);
    testAssert(se[0] > 0.0);
    testAssert(se[0] < 2.0);
  }

  // Test computeOptimumSE: more data gives smaller SE
  {
    auto buildData = [](int reps, vector<vector<double>>& xs, vector<double>& ys) {
      for(int i = 0; i < reps; i++) {
        xs.push_back({0.0});  ys.push_back(1.0);
        xs.push_back({0.8});  ys.push_back(0.0);
        xs.push_back({-0.8}); ys.push_back(0.0);
      }
    };

    QRSModel modelSmall(1, 0.01);
    QRSModel modelLarge(1, 0.01);
    vector<vector<double>> xsSmall, xsLarge;
    vector<double> ysSmall, ysLarge;
    buildData(15, xsSmall, ysSmall);
    buildData(150, xsLarge, ysLarge);
    modelSmall.fit(xsSmall, ysSmall);
    modelLarge.fit(xsLarge, ysLarge);
    double seSmall[1], seLarge[1];
    bool clampedSmall[1], clampedLarge[1];
    bool okSmall = modelSmall.computeOptimumSE(xsSmall, seSmall, clampedSmall);
    bool okLarge = modelLarge.computeOptimumSE(xsLarge, seLarge, clampedLarge);
    testAssert(okSmall && okLarge);
    testAssert(seLarge[0] < seSmall[0]);
  }

  // Test computeOptimumSE: insufficient data returns false
  {
    QRSModel model(1, 0.1);
    vector<vector<double>> xs = {{0.5}, {-0.5}};
    vector<double> ys = {1.0, 0.0};
    // N=2 < F=3, so fit() does nothing and beta stays zero
    model.fit(xs, ys);
    double se[1];
    bool clamped[1];
    bool ok = model.computeOptimumSE(xs, se, clamped);
    testAssert(!ok);
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

  // Test QRSTuner convergence with stochastic outcomes in a 1D quadratic
  // landscape centered at x* = 0.35.
  {
    const double trueOpt = 0.35;
    const int numTrials = 500;
    mt19937_64 outcomeRng(99);
    uniform_real_distribution<double> uni01(0.0, 1.0);

    QRSTuner tuner(1, /*seed=*/42, numTrials,
                   /*l2_reg=*/0.1, /*refit_every=*/10, /*prune_every=*/5,
                   /*sigma_init=*/0.60, /*sigma_fin=*/0.20);
    for(int trial = 0; trial < numTrials; trial++) {
      vector<double> sample = tuner.nextSample();
      double dx = sample[0] - trueOpt;
      double winProb = sigmoid(2.0 - 4.0 * dx * dx);
      double outcome = (uni01(outcomeRng) < winProb) ? 1.0 : 0.0;
      tuner.addResult(sample, outcome);
    }
    vector<double> best = tuner.bestCoords();
    testAssert(fabs(best[0] - trueOpt) < 0.15);
    testAssert(tuner.bestWinProb() > 0.7);
  }

  // Test: Nearly-flat 3D landscape exposes convex-fitting bug.
  //
  // When the true function is nearly flat and we have only ~128 stochastic
  // trials fitting 10 parameters, noise can make the fitted quadratic convex
  // (positive coefficient) in some dimensions.  mapOptimum() then returns the
  // MINIMUM in those dimensions instead of the maximum.
  {
    const int D = 3;
    const int numTrials = 128;
    const double trueOpt[3] = {0.3, -0.2, 0.4};
    const double curvature = 0.1;  // very weak — winrate spans only ~0.39-0.50

    mt19937_64 outcomeRng(0);
    uniform_real_distribution<double> uni01(0.0, 1.0);

    QRSTuner tuner(D, /*seed=*/42, numTrials,
                   /*l2_reg=*/0.1, /*refit_every=*/10, /*prune_every=*/5,
                   /*sigma_init=*/0.60, /*sigma_fin=*/0.20);

    for(int trial = 0; trial < numTrials; trial++) {
      vector<double> sample = tuner.nextSample();
      double sc = 0.0;
      for(int d = 0; d < D; d++) {
        double dx = sample[d] - trueOpt[d];
        sc -= curvature * dx * dx;
      }
      double winProb = sigmoid(sc);
      double outcome = (uni01(outcomeRng) < winProb) ? 1.0 : 0.0;
      tuner.addResult(sample, outcome);
    }

    // Probe fitted quadratic coefficients:
    //   quadCoeff_k = (score(e_k) + score(-e_k) - 2*score(0)) / 2
    const QRSModel& model = tuner.model();
    double origin[3] = {0.0, 0.0, 0.0};
    double s0 = model.score(origin);
    bool anyConvex = false;
    double probe[3] = {0.0, 0.0, 0.0};
    for(int d = 0; d < D; d++) {
      probe[d] = 1.0;
      double sp = model.score(probe);
      probe[d] = -1.0;
      double sn = model.score(probe);
      probe[d] = 0.0;
      double quadCoeff = (sp + sn - 2.0 * s0) / 2.0;
      if(quadCoeff > 0.0) {
        anyConvex = true;
        break;
      }
    }
    // With these seeds, noise overwhelms the weak signal and at least one
    // fitted dimension ends up convex (positive quadratic coefficient).
    testAssert(anyConvex);

    // Invariant: the optimizer's "best" should predict at least as well as
    // an arbitrary point like the origin.  Currently fails because
    // mapOptimum() returns the critical point of the fitted quadratic
    // without checking whether it is a maximum or minimum.
    double probAtBest = tuner.bestWinProb();
    double probAtOrigin = model.predict(origin);
    testAssert(probAtBest >= probAtOrigin);
  }

  // Regression test for Newton-Raphson intercept divergence on a flat 2D
  // landscape (true winrate = 50% everywhere, so correct intercept = 0).
  // Scans seeds until one triggers the warm-start saturation cascade.
  {
    const int D = 2;
    const int numTrials = 100;
    // Diverged intercepts are 400-500; non-diverged are < 1.
    const double DIVERGE_THRESHOLD = 50.0;

    bool diverged = false;
    uniform_real_distribution<double> uni01(0.0, 1.0);

    for(uint64_t tunerSeed = 0; tunerSeed < 20 && !diverged; tunerSeed++) {
      mt19937_64 outcomeRng(tunerSeed * 1000 + 7);

      QRSTuner tuner(D, /*seed=*/tunerSeed, numTrials,
                     /*l2_reg=*/0.1, /*refit_every=*/10, /*prune_every=*/5,
                     /*sigma_init=*/0.40, /*sigma_fin=*/0.05);

      for(int trial = 0; trial < numTrials; trial++) {
        vector<double> sample = tuner.nextSample();
        double outcome = (uni01(outcomeRng) < 0.5) ? 1.0 : 0.0;
        tuner.addResult(sample, outcome);
      }

      if(fabs(tuner.model().beta()[0]) > DIVERGE_THRESHOLD)
        diverged = true;
    }

    // Fixed: no seed triggers intercept divergence after removing warm-start.
    testAssert(!diverged);
  }
}
