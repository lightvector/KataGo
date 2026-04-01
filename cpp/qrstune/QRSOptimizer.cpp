// qrstune/QRSOptimizer.cpp

#include "../qrstune/QRSOptimizer.h"

#include <algorithm>
#include <cmath>

using namespace std;

// ============================================================
// Free functions
// ============================================================

int QRSTune::numFeatures(int D) {
  return 1 + D + D * (D + 1) / 2;
}

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
  if(z > 40.0) return 1.0;
  if(z < -40.0) return 0.0;
  return 1.0 / (1.0 + exp(-z));
}

bool QRSTune::gaussianSolve(int F, vector<vector<double>>& A, vector<double>& b) {
  for(int col = 0; col < F; col++) {
    int piv = col;
    for(int r = col + 1; r < F; r++)
      if(fabs(A[r][col]) > fabs(A[piv][col])) piv = r;
    swap(A[col], A[piv]);
    swap(b[col], b[piv]);
    if(fabs(A[col][col]) < 1e-12) return false;
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

void QRSTune::QRSModel::fit(const vector<vector<double>>& xs,
                             const vector<double>& ys,
                             int max_iter) {
  int N = (int)xs.size();
  if(N < F_) return;  // underdetermined; keep prior beta = 0

  vector<double> phi(F_);
  vector<double> grad(F_);
  vector<vector<double>> negH(F_, vector<double>(F_));

  for(int iter = 0; iter < max_iter; iter++) {
    // Gradient and (negative) Hessian from L2 prior
    fill(grad.begin(), grad.end(), 0.0);
    for(int f = 0; f < F_; f++) fill(negH[f].begin(), negH[f].end(), 0.0);
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
      maxd = max(maxd, fabs(grad[f]));
    }
    if(maxd < 1e-7) break;
  }
}

double QRSTune::QRSModel::predict(const double* x) const {
  return sigmoid(score(x));
}

double QRSTune::QRSModel::score(const double* x) const {
  vector<double> phi(F_);
  computeFeatures(D_, x, phi.data());
  double z = 0.0;
  for(int f = 0; f < F_; f++) z += beta_[f] * phi[f];
  return z;
}

void QRSTune::QRSModel::mapOptimum(double* out_x) const {
  // Beta layout: [intercept, linear[0..D-1], quad[0..D-1], cross by (i<j)]
  const double* b_lin  = beta_.data() + 1;
  const double* b_quad = beta_.data() + 1 + D_;
  const double* b_cross = beta_.data() + 1 + 2 * D_;

  vector<vector<double>> M(D_, vector<double>(D_, 0.0));
  vector<double> rhs(D_);

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

void QRSTune::QRSBuffer::prune(const QRSModel& model) {
  int N = (int)xs_.size();
  if(N <= min_keep_ * 2) return;

  // Score all samples and find best predicted win rate
  vector<pair<double, int>> scored(N);
  double p_best = 0.0;
  for(int i = 0; i < N; i++) {
    double p = model.predict(xs_[i].data());
    scored[i] = {p, i};
    if(p > p_best) p_best = p;
  }
  double threshold = p_best - prune_margin_;

  // Sort by descending predicted quality
  sort(scored.begin(), scored.end(),
    [](const pair<double,int>& a, const pair<double,int>& b) {
      return a.first > b.first;
    });

  // Keep samples above threshold, plus top-quality samples up to min_keep_
  vector<bool> keep(N, false);
  int kept = 0;
  for(auto& kv : scored) {
    if(kv.first >= threshold || kept < min_keep_) {
      keep[kv.second] = true;
      kept++;
    }
  }

  // Rebuild in original order to preserve temporal structure
  vector<vector<double>> nx;
  vector<double> ny;
  for(int i = 0; i < N; i++) {
    if(keep[i]) {
      nx.push_back(std::move(xs_[i]));
      ny.push_back(ys_[i]);
    }
  }
  xs_ = std::move(nx);
  ys_ = std::move(ny);
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

  // Base: MAP optimum
  model_.mapOptimum(x.data());

  // Decaying exploration noise
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
