#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <starpu.h>

using namespace std;

void getrf(void *buffers[], void *) {
  double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  int N = (int)STARPU_MATRIX_GET_NX(buffers[0]);
  vector<int> ipiv(N);
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N, N, A, N, ipiv.data());
}

struct starpu_codelet getrf_cl = {
  .cpu_funcs = {getrf},
  .nbuffers = 1,
};

void trsm(void *buffers[], void *cl_arg) {
  double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
  int NA = (int)STARPU_MATRIX_GET_NX(buffers[0]);
  int NB = (int)STARPU_MATRIX_GET_NY(buffers[1]);
  bool left, up;
  starpu_codelet_unpack_args(cl_arg, &left, &up);
  cblas_dtrsm(CblasRowMajor, left ? CblasLeft : CblasRight, up ? CblasUpper : CblasLower,
              CblasNoTrans, up ? CblasNonUnit : CblasUnit, NA, NB, 1.0, A, NA, B, NB);
}

struct starpu_codelet trsm_cl = {
  .cpu_funcs = {trsm},
  .nbuffers = 2,
};

void trsm(bool left, bool up, vector<double>& A, int NA, vector<double>& B, int NB) {
  cblas_dtrsm(CblasRowMajor, left ? CblasLeft : CblasRight, up ? CblasUpper : CblasLower,
              CblasNoTrans, up ? CblasNonUnit : CblasUnit, NA, NB, 1.0, A.data(), NA, B.data(), NB);
}

void gemm(void *buffers[], void *) {
  double *A = (double *)STARPU_MATRIX_GET_PTR(buffers[0]);
  double *B = (double *)STARPU_MATRIX_GET_PTR(buffers[1]);
  double *C = (double *)STARPU_MATRIX_GET_PTR(buffers[2]);
  int NA = (int)STARPU_MATRIX_GET_NX(buffers[0]);
  int NB = (int)STARPU_MATRIX_GET_NY(buffers[1]);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NA, NB, NA, -1.0,
              A, NA, B, NB, 1.0, C, NB);
}

struct starpu_codelet gemm_cl = {
  .cpu_funcs = {gemm},
  .nbuffers = 3,
};

void gemm(vector<double>& A, int NA, vector<double>& B, int NB, vector<double>& C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NA, NB, NA, -1.0,
              A.data(), NA, B.data(), NB, 1.0, C.data(), NB);
}

int main() {
  int M = 3;
  int N = 16;
  bool left = true, right = false, upper = true, lower = false;
  vector<vector<double> > A(M*M);
  vector<vector<double> > x(M);
  vector<vector<double> > b(M);
  vector<int> ipiv(N);
  vector<starpu_data_handle_t> A_h(M*M);
  vector<starpu_data_handle_t> b_h(M);

  int ret = starpu_init(NULL); 
  for (int m=0; m<M*M; m++) {
    A[m] = vector<double>(N*N);
    starpu_matrix_data_register(&A_h[m],0,(uintptr_t)A[m].data(),N,N,N,sizeof(double));
  }
  for (int m=0; m<M; m++) {
    x[m] = vector<double>(N);
    b[m] = vector<double>(N);
    starpu_matrix_data_register(&b_h[m],0,(uintptr_t)b[m].data(),N,1,1,sizeof(double));
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
  for (int l=0; l<M; l++) {
    starpu_task_insert(&getrf_cl, STARPU_RW, A_h[M*l+l], 0);
    for (int m=l+1; m<M; m++) {
      starpu_task_insert(&trsm_cl, STARPU_RW, A_h[M*l+l], STARPU_RW, A_h[M*l+m], 
			 STARPU_VALUE, &left, sizeof(bool),
			 STARPU_VALUE, &lower, sizeof(bool), 0);
      starpu_task_insert(&trsm_cl, STARPU_RW, A_h[M*l+l], STARPU_RW, A_h[M*m+l], 
			 STARPU_VALUE, &right, sizeof(bool),
			 STARPU_VALUE, &upper, sizeof(bool), 0);
    }
    for (int m=l+1; m<M; m++)
      for (int n=l+1; n<M; n++)
	starpu_task_insert(&gemm_cl, STARPU_RW, A_h[M*m+l], STARPU_RW, A_h[M*l+n], STARPU_RW, A_h[M*m+n], 0);
  }
  starpu_task_wait_for_all();
  for (int m=0; m<M; m++) {
    for (int n=0; n<m; n++)
      gemm(A[M*m+n], N, b[n], 1, b[m]);
    trsm(left, lower, A[M*m+m], N, b[m], 1);
  }
  for (int m=M-1; m>=0; m--) {
    for (int n=M-1; n>m; n--)
      gemm(A[M*m+n], N, b[n], 1, b[m]);
    trsm(left, upper, A[M*m+m], N, b[m], 1);
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
  starpu_shutdown();
}
