#include <limits>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <armadillo>
#include <random>
#include <typeinfo>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;
using namespace arma;

int main() {
    // rowvec a = ones<rowvec>(4);
    int k = 2;
    int l = 3;

    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    cout << "Creating Cublas Handler...";
    stat = cublasCreate(&handle);
    cout << " created.\n";

    double al = 1.0;
    double bet = 0.0;


    // COMPUTE C = A*B -----
    double* C = (double*) malloc(2 * 3 * sizeof(double)); // matrix 2 x 3
    for (int i=0; i < 6; i++)
        C[i] = 0.0;

    cout << "C before: ";
    for (int i=0; i < 2*3; i++)
        cout << C[i] << ", ";
    cout << "\n";

    double* A = (double*) malloc(2 * 2 * 2 * sizeof(double)); // cube 2 x 2 x 2
    A[0] = 2.0;
    for (int i=1; i < 8; i++)
        A[i] = A[i-1] + 1.0;

    cout << "A before: ";
    for (int i=0; i < 8; i++)
        cout << A[i] << ", ";
    cout << "\n";

    double* dA;
    cudaStat = cudaMalloc((void**)& dA, 2 * 2 * 2 * sizeof(double));
    for (int i = 0; i < 2; i++) {
        int ind = (2 * 2) * i;
        double* dAi = &dA[ind];
        stat = cublasSetMatrix(2, 2, sizeof(double), A, 2, dAi, 2);
    }

    cout << "A after: ";
    for (int i=0; i < 8; i++)
        cout << A[i] << ", ";
    cout << "\n";

    double* B = (double*) malloc(2 * 3 * sizeof(double)); // matrix 2 x 3
    for (int i=0; i < 6; i++)
        B[i] = 3.0;

    cout << "B: ";
    for (int i=0; i < 6; i++)
        cout << B[i] << ", ";
    cout << "\n";

    double* dB;
    cudaStat = cudaMalloc((void**)& dB, 2 * 3 * sizeof(double));
    stat = cublasSetMatrix(2, 3, sizeof(double), B, 2, dB, 2);

    double* dC;
    cudaStat = cudaMalloc((void**)& dC, 2 * 3 * sizeof(double));

    int ind = (2 * 2) * 1;
    double* dAi = &dA[ind];
    double* Ai = &A[ind];
    cout << "Ai: ";
    for (int i=0; i < 4; i++)
        cout << Ai[i] << ", ";
    cout << "\n";

    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 3, 2, &al, dAi, 2, dB, 2, &bet, dC, 2);

    stat = cublasGetMatrix(2, 3, sizeof(double), dC, 2, C, 2);

    cout << "C after: ";
    for (int i=0; i < 2*3; i++)
        cout << C[i] << ", ";

    cout << "\n";
    // END ---------------

}
