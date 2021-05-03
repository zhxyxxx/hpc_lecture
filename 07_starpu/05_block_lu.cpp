#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>

#include "mkl_cblas.h"
#include "mkl_lapacke.h"

using namespace std;

int main() {
  int N = 256;
  vector<double> A(N*N);
  vector<double> x(N);
  vector<double> b(N);
  vector<int> ipiv(N);
  for (int i=0; i<N; i++)
    x[i] = drand48();
  for (int i=0; i<N; i++) {
    b[i] = 0;
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48() + (i == j) * 100;
      b[i] += A[N*i+j] * x[j];
    }
  }
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A.data(), N, ipiv.data());
  //LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', N, 1, A.data(), N, ipiv.data(), b.data(), 1);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1.0, A.data(), N, b.data(), 1);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1.0, A.data(), N, b.data(), 1);
  for (int i=0; i<N; i++)
    printf("%lf %lf %d\n",x[i],b[i],ipiv[i]);
}
