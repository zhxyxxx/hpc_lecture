#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>

#include "mkl_cblas.h"
#include "mkl_lapacke.h"

using namespace std;

int main() {
  int N = 16;
  vector<double> A00(N*N), A01(N*N), A10(N*N), A11(N*N);
  vector<double> x0(N), x1(N);
  vector<double> b0(N), b1(N);
  vector<int> ipiv(N);
  for (int i=0; i<N; i++) {
    x0[i] = drand48();
    x1[i] = drand48();
  }
  for (int i=0; i<N; i++) {
    b0[i] = b1[i] = 0;
    for (int j=0; j<N; j++) {
      A00[N*i+j] = drand48() + (i == j) * 10;
      A01[N*i+j] = drand48();
      A10[N*i+j] = drand48();
      A11[N*i+j] = drand48() + (i == j) * 10;
      b0[i] += A00[N*i+j] * x0[j] + A01[N*i+j] * x1[j];
      b1[i] += A10[N*i+j] * x0[j] + A11[N*i+j] * x1[j];
    }
  }
  // L00,U00 = lu(A00) (overwrite A00)
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A00.data(), N, ipiv.data());
  // U01 = L00^-1 * A01 (overwrite A01)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, N, 1.0, A00.data(), N, A01.data(), N);
  // L10 = A01 * U00^-1 (overwrite A10)
  cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A00.data(), N, A10.data(), N);
  // A11 -= L10 * U01 (overwrite A11)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, -1.0, A10.data(), N, A01.data(), N, 1.0, A11.data(),N);
  // L11,U11 = lu(A11) (overwrite A11)
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A11.data(), N, ipiv.data());
  // y0 = L00^-1 b0 (overwrite b0)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1.0, A00.data(), N, b0.data(), 1);
  // b1 -= L10 * y0 (overwrite b1)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, -1.0, A10.data(), N, b0.data(), 1, 1.0, b1.data(), 1);
  // y1 = L11^-1 * b1 (overwrite b1)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1.0, A11.data(), N, b1.data(), 1);
  // x1 = U11^-1 * y1 (overwrite b1)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1.0, A11.data(), N, b1.data(), 1);
  // y0 -= U01 * x1 (overwrite b0)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, -1.0, A01.data(), N, b1.data(), 1, 1.0, b0.data(),1);
  // x0 = U00^-1 * y0 (overwrite b1)
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1.0, A00.data(), N, b0.data(), 1);

  double diff = 0, norm = 0;
  for (int i=0; i<N; i++) {
    diff += (x0[i] - b0[i]) * (x0[i] - b0[i]);
    diff += (x1[i] - b1[i]) * (x1[i] - b1[i]);
    norm += x0[i] * x0[i];
    norm += x1[i] * x1[i];
  }
  printf("Error: %g\n",std::sqrt(diff/norm));
}
