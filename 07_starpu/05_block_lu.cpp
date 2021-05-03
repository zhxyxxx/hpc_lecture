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
      A[N*i+j] = drand48();
      b[i] += A[N*i+j] * x[j];
    }
  }
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, &A[0], N, &ipiv[0]);
}
