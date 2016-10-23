# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include <cuda.h>
#include <armadillo>
# include "cublas_v2.h"
# define IDX2C(i, j, ld) (((j) * (ld)) + (i))
# define m 6 // a - mxk matrix
# define n 4 // b - kxn matrix
# define k 5 // c - mxn matrix

using namespace arma;

__global__ void divide(double* a, double* b, double* c, long sizeN) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < sizeN) {
      c[index] = a[index] / b[index];
    }
}

int main() {
    cudaError_t cudaStat; // cudaMalloc status
    cublasStatus_t stat; // CUBLAS functions status
    cublasHandle_t handle; // CUBLAS context

    mat arma_a(m, k);
    mat arma_b(k, n);
    mat arma_c(m, n);

    int i,j;  // i-row index ,j- column index
    double* a = arma_a.memptr(); // mxk matrix a on the host
    double* b = arma_b.memptr(); // kxn matrix b on the host
    double* c = arma_c.memptr(); // mxn matrix c on the host

    // a = (double*) malloc(m*k* sizeof(double)); // host memory for a
    // b = (double*) malloc(k*n* sizeof(double)); // host memory for b
    // c = (double*) malloc(m*n* sizeof(double)); // host memory for c

    // define an mxk matrix a column by column
    // int ind = 11;                               // a:
    // for (j = 0; j < k; j++) {                   // 11 ,17 ,23 ,29 ,35
    //     for (i = 0; i < m; i++) {               // 12 ,18 ,24 ,30 ,36
    //         a[IDX2C(i, j, m)] = (double) ind++; // 13 ,19 ,25 ,31 ,37
    //     }                                       // 14 ,20 ,26 ,32 ,38
    // }                                           // 15 ,21 ,27 ,33 ,39
    //                                             // 16 ,22 ,28 ,34 ,40
    arma_a << 11 << 17 << 23 << 29 << 35 << endr
           << 12 << 18 << 24 << 30 << 36 << endr
           << 13 << 19 << 25 << 31 << 37 << endr
           << 14 << 20 << 26 << 32 << 38 << endr
           << 15 << 21 << 27 << 33 << 39 << endr
           << 16 << 22 << 28 << 34 << 40 << endr;


    // print a row by row
    // printf("a:\n");
    // for (i = 0; i < m; i++) {
    //     for (j = 0; j < k; j++) {
    //         printf("%5.0f", a[IDX2C(i, j, m)]);
    //     }
    //     printf("\n");
    // }

    arma_a.print("a: ");

    // define a kxn matrix b column by column
    // ind = 11;                                   // b:
    // for (j=0; j<n; j++) {                       // 11 ,16 ,21 ,26
    //     for (i=0; i<k; i++) {                   // 12 ,17 ,22 ,27
    //         b[IDX2C(i, j, k)] = (double) ind++; // 13 ,18 ,23 ,28
    //     }                                       // 14 ,19 ,24 ,29
    // }                                           // 15 ,20 ,25 ,30

    arma_b << 11 << 16 << 21 << 26 << endr
           << 12 << 17 << 22 << 27 << endr
           << 13 << 18 << 23 << 28 << endr
           << 14 << 19 << 24 << 29 << endr
           << 15 << 20 << 25 << 30 << endr;

    // print b row by row
    // printf ("b:\n");
    // for (i=0; i<k; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%5.0f", b[IDX2C(i, j, k)]);
    //     }
    //     printf("\n");
    // }

    arma_b.print("b: ");

    // // define an mxn matrix c column by column
    // ind =11;                                    // c:
    // for (j=0; j<n; j++) {                       // 11 ,17 ,23 ,29
    //     for (i=0; i<m; i++) {                   // 12 ,18 ,24 ,30
    //         c[IDX2C(i, j, m)] = (double) ind++; // 13 ,19 ,25 ,31
    //     }                                       // 14 ,20 ,26 ,32
    // }                                           // 15 ,21 ,27 ,33
    //                                             // 16 ,22 ,28 ,34

    arma_c << 11 << 17 << 23 << 29 << endr
           << 12 << 18 << 24 << 30 << endr
           << 13 << 19 << 25 << 31 << endr
           << 14 << 20 << 26 << 32 << endr
           << 15 << 21 << 27 << 33 << endr
           << 16 << 22 << 28 << 34 << endr;

    // print c row by row
    // printf("c:\n");
    // for (i=0; i<m; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%5.0f", c[IDX2C(i, j, m)]);
    //     }
    //     printf("\n");
    // }

    arma_c.print("c: ");

    // on the device
    double* d_a; // d_a - a on the device
    double* d_b; // d_b - b on the device
    double* d_c; // d_c - c on the device

    // memory alloc for a
    cudaStat = cudaMalloc((void**)& d_a, m*k* sizeof(*a)); // device

    // memory alloc for b
    cudaStat = cudaMalloc((void**)& d_b, k*n* sizeof(*b)); // device

    // memory alloc for c
    cudaStat = cudaMalloc((void**)& d_c, m*n* sizeof(*c)); // device

    stat = cublasCreate(& handle); // initialize CUBLAS context

    // copy matrices from the host to the device
    stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m); //a -> d_a
    stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k); //b -> d_b
    stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m); //c -> d_c

    double al = 1.0f;  // al =1
    double bet = 0.0f; // bet =1

    // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
    // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
    // al ,bet -scalars
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a, m, d_b, k, &bet, d_c, m);

    // long nm = n * m;
    // long num_blocks = (nm + 1024 - 1) / 1024;
    // divide<<<num_blocks, 1024>>>(d_a, d_a, d_c, nm);
    // stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &al, dX, n, dXtwo, n, &bet, dXtwo_new, m);
    stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m); // cp d_c - >c

    // printf("c after Sgemm :\n");
    // for (i=0; i<m; i++) {
    //     for (j=0; j<n; j++) {
    //         printf("%7.0f", c[IDX2C(i, j, m)]); // print c after Sgemm
    //     }
    //     printf ("\n");
    // }
    arma_c.print("c: ");

    cudaFree(d_a); // free device memory
    cudaFree(d_b); // free device memory
    cudaFree(d_c); // free device memory

    cublasDestroy(handle); // destroy CUBLAS context

    // free(a); // free host memory
    // free(b); // free host memory
    // free(c); // free host memory

    return EXIT_SUCCESS;
}
