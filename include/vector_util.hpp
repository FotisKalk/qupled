#ifndef VECTOR_UTIL_HPP
#define VECTOR_UTIL_HPP

#include <vector>
#include <vector2D.hpp>

// ------------------------------------------------------------------
// Utility functions to manipulate vectors from the standard library
// ------------------------------------------------------------------

namespace vecUtil {

  // Element-wise sum between two vectors
  std::vector<double> sum(const std::vector<double> &v1,
                          const std::vector<double> &v2);

  // Element-wise difference between two vectors
  std::vector<double> diff(const std::vector<double> &v1,
                           const std::vector<double> &v2);

  // Element-wise multiplication of two vectors
  std::vector<double> mult(const std::vector<double> &v1,
                           const std::vector<double> &v2);

  // Element-wise division of two vectors
  std::vector<double> div(const std::vector<double> &v1,
                          const std::vector<double> &v2);

  // Element-wise multiplication of a vector and a scalar
  std::vector<double> mult(const std::vector<double> &v, const double a);

  // Root mean square difference between two vectors
  double rms(const std::vector<double> &v1,
             const std::vector<double> &v2,
             const bool normalize);

  // Fill vector with constant values
  void fill(std::vector<double> &v, const double &num);

  Vector2D multiply(const Vector2D &A, const Vector2D &B);
  std::vector<double> multiply(const Vector2D &A, const std::vector<double> &x);
  Vector2D transpose(const Vector2D &A);
  std::vector<double> solveGaussian(const Vector2D &A, const std::vector<double> &b);

} // namespace vecUtil

#endif
