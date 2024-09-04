#include "vector_util.hpp"
#include "vector2D.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

using namespace std;

namespace vecUtil {

  // Element-wise sum between two vectors
  vector<double> sum(const vector<double> &v1, const vector<double> &v2) {
    assert(v1.size() == v2.size());
    vector<double> res(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), res.begin(), plus<double>());
    return res;
  }

  // Element-wise difference between two vectors
  vector<double> diff(const vector<double> &v1, const vector<double> &v2) {
    assert(v1.size() == v2.size());
    vector<double> res(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), res.begin(), minus<double>());
    return res;
  }

  // Element-wise multiplication between two vectors
  vector<double> mult(const vector<double> &v1, const vector<double> &v2) {
    assert(v1.size() == v2.size());
    vector<double> res(v1.size());
    transform(
        v1.begin(), v1.end(), v2.begin(), res.begin(), multiplies<double>());
    return res;
  }

  // Element-wise division between two vectors
  vector<double> div(const vector<double> &v1, const vector<double> &v2) {
    assert(v1.size() == v2.size());
    vector<double> res(v1.size());
    transform(v1.begin(), v1.end(), v2.begin(), res.begin(), divides<double>());
    return res;
  }

  // Element-wise multiplication of a vector and a scalar
  vector<double> mult(const vector<double> &v, const double a) {
    vector<double> res = v;
    transform(
        res.begin(), res.end(), res.begin(), [&a](double c) { return c * a; });
    return res;
  }

  // Root square difference between two vectors
  double rms(const vector<double> &v1,
             const vector<double> &v2,
             const bool normalize) {
    const vector<double> tmp = diff(v1, v2);
    double rms = inner_product(tmp.begin(), tmp.end(), tmp.begin(), 0.0);
    if (normalize) rms /= tmp.size();
    return sqrt(rms);
  }

  // Fill vector with constant value
  void fill(vector<double> &v, const double &num) {
    std::for_each(v.begin(), v.end(), [&](double &vi) { vi = num; });
  }

  // Matrix multiplication: C = A * B
  Vector2D multiply(const Vector2D &A, const Vector2D &B) {
    assert(A.size(1) == B.size(0));
    size_t n = A.size(0);
    size_t m = B.size(1);
    size_t p = B.size(0);
    Vector2D C(n, m);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        for (size_t k = 0; k < p; ++k) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }
    return C;
  }

  // Matrix-vector multiplication: y = A * x
  vector<double> multiply(const Vector2D &A, const vector<double> &x) {
    assert(A.size(1) == x.size());
    size_t n = A.size(0);
    size_t m = x.size();
    vector<double> y(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        y[i] += A(i, j) * x[j];
      }
    }
    return y;
  }

  // Transpose of a matrix
  Vector2D transpose(const Vector2D &A) {
    size_t n = A.size(0);
    size_t m = A.size(1);
    Vector2D AT(m, n);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        AT(j, i) = A(i, j);
      }
    }
    return AT;
  }

  // Solve linear system using Gaussian elimination
  vector<double> solveGaussian(const Vector2D &A, const vector<double> &b) {
    int n = A.size(0);
    Vector2D augmented(n, n + 1);
    
    // Construct augmented matrix
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        augmented(i, j) = A(i, j);
      }
      augmented(i, n) = b[i];
    }
    
    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
      // Find pivot
      int pivot = i;
      for (int j = i + 1; j < n; ++j) {
        if (std::abs(augmented(j, i)) > std::abs(augmented(pivot, i))) {
          pivot = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        std::swap(augmented(i, j), augmented(pivot, j));
      }
      
      // Normalize pivot row
      double pivotValue = augmented(i, i);
      for (int j = i; j <= n; ++j) {
        augmented(i, j) /= pivotValue;
      }
      
      // Eliminate below
      for (int j = i + 1; j < n; ++j) {
        double factor = augmented(j, i);
        for (int k = i; k <= n; ++k) {
          augmented(j, k) -= factor * augmented(i, k);
        }
      }
    }
    
    // Back substitution
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
      x[i] = augmented(i, n);
      for (int j = i + 1; j < n; ++j) {
        x[i] -= augmented(i, j) * x[j];
      }
    }
    return x;
  }
} // namespace vecUtil
