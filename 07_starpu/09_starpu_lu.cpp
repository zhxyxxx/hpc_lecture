#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <starpu.h>

using namespace std;

int main() {
  int M = 3;
  int N = 16;
  vector<vector<double> > A(M*M);
  vector<vector<double> > x(M);
  vector<vector<double> > b(M);
  vector<int> ipiv(N);
  vector<starpu_data_handle_t> A_h(M*M);
  vector<starpu_data_handle_t> b_h(M);
  
  for (int m=0; m<M*M; m++) {
    A[m] = vector<double>(N*N);
    starpu_matrix_data_register(&A_h[m],0,(uintptr_t)A[m].data(),N,N,N,sizeof(double));
  }
  for (int m=0; m<M; m++) {
    x[m] = vector<double>(N);
    b[m] = vector<double>(N);
    starpu_vector_data_register(&b_h[m],0,(uintptr_t)b[m].data(),N,sizeof(double));
  }
  for (int m=0; m<M; m++) {
    for (int i=0; i<N; i++) {
      x[m][i] = drand48();
      b[m][i] = 0;
    }
  }
  for (int m=0; m<M; m++) {
    for (int n=0; n<M; n++) {
      for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
          A[M*m+n][N*i+j] = drand48() + (m == n) * (i == j) * 10;
          b[m][i] += A[M*m+n][N*i+j] * x[n][j];
	}
      }
    }
  }
  starpu_codelet cl;
  starpu_codelet_init(&cl);
  for (int l=0; l<M; l++) {
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A[M*l+l].data(), N, ipiv.data());
    for (int m=l+1; m<M; m++) {
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, N, 1.0, A[M*l+l].data(), N, A[M*l+m].data(), N);
      cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A[M*l+l].data(), N, A[M*m+l].data(), N);
    }
    for (int m=l+1; m<M; m++)
      for (int n=l+1; n<M; n++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, -1.0, A[M*m+l].data(), N, A[M*l+n].data(), N, 1.0, A[M*m+n].data(),N);
  }
  for (int m=0; m<M; m++) {
    for (int n=0; n<m; n++)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, -1.0, A[M*m+n].data(), N, b[n].data(), 1, 1.0, b[m].data(), 1);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, N, 1, 1.0, A[M*m+m].data(), N, b[m].data(), 1);
  }
  for (int m=M-1; m>=0; m--) {
    for (int n=M-1; n>m; n--)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, 1, N, -1.0, A[M*m+n].data(), N, b[n].data(), 1, 1.0, b[m].data(),1);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, 1, 1.0, A[M*m+m].data(), N, b[m].data(), 1);
  }
  starpu_task_wait_for_all();
  for (int m=0; m<M*M; m++)
    starpu_data_unregister(A_h[m]);
  for (int m=0; m<M; m++)
    starpu_data_unregister(b_h[m]);

  double diff = 0, norm = 0;
  for (int m=0; m<M; m++) {
    for (int i=0; i<N; i++) {
      diff += (x[m][i] - b[m][i]) * (x[m][i] - b[m][i]);
      norm += x[m][i] * x[m][i];
    }
  }
  printf("Error: %g\n",std::sqrt(diff/norm));
  //starpu_shutdown();
}
