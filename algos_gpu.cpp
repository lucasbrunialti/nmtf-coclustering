
#include <limits>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <armadillo>
#include <random>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace std;
using namespace arma;

#define DBL_MAX numeric_limits<double>::max()

__global__ void divideElemWise(double* a, double* b, double* c, long n) {
  long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] / b[index];
}

__global__ void multiplyElemWise(double* a, double* b, double* c, long n) {
  long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] * b[index];
}

__global__ void sumElemWise(double* a, double* b, double* c, long n) {
  long index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n)
    c[index] = a[index] + b[index];
}

__global__ void changeElem(double *arr, int idx, double val) {
  arr[idx] = val;
}

void ovnmtf(mat X, const long& k, const long& l, const long& num_iter) {
  long n = X.n_rows;
  long m = X.n_cols;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double al = 1.0;
  double bet = 0.0;
  const long THREADS_PER_BLOCK = 1024;

  double* X_raw = X.memptr();

  mat U(n, k);
  mat S(k, l);
  cube V(m, l, k);

  mat Ii;
  Ii.zeros(k, k);
  mat Ij;
  Ij.zeros(k, k);

  double* U_raw = U.memptr();
  double* S_raw = S.memptr();
  double* V_raw = V.memptr();
  for(int i=0; i < n*k; i++) U_raw[i] = unif(gen);
  for(int i=0; i < k*l; i++) S_raw[i] = unif(gen);
  for(int i=0; i < k*m*l; i++) V_raw[i] = unif(gen);

  // U.print("U before: ")
  // S.print("S before: ");
  // V.print("V before: ");

  mat U_best;
  U_best.zeros(n, k);
  mat S_best;
  S_best.zeros(k, l);
  cube V_best;
  V_best.zeros(m, l, k);

  mat V_tilde;
  V_tilde.zeros(k, m);
  mat V_tilde_best;
  V_tilde_best.zeros(k, m);

  double error_best = DBL_MAX;
  double error_ant = DBL_MAX;
  double error = DBL_MAX;

  // cout << "Creating Cublas Handler...";
  stat = cublasCreate(&handle);
  // cout << " created.\n";

  // cout << "Creating X, U, S, Vs into GPU...";
  double* dX;
  double* dU;
  double* dS;
  double* dV;

  cudaStat = cudaMalloc((void**)& dX, n * m * sizeof(double));
  cudaStat = cudaMalloc((void**)& dU, n * k * sizeof(double));
  cudaStat = cudaMalloc((void**)& dS, k * l * sizeof(double));
  cudaStat = cudaMalloc((void**)& dV, m * l * k * sizeof(double));
  // cout << " created." << endl;

  // cout << "Allocating X, U, S, Vs into GPU...";
  stat = cublasSetMatrix(n, m, sizeof(double), X_raw, n, dX, n);
  stat = cublasSetMatrix(n, k, sizeof(double), U_raw, n, dU, n);
  stat = cublasSetMatrix(k, l, sizeof(double), S_raw, k, dS, k);
  for (int i = 0; i < k; i++) {
    int ind = (m * l) * i;
    double* dVi = &dV[ind];
    stat = cublasSetMatrix(m, l, sizeof(double), V.slice(i).memptr(), m, dVi, m);
  }
  // cout << " allocated.\n";

  double* dIi;
  double* dIj;
  cudaStat = cudaMalloc((void**)& dIi, k * k * sizeof(double));
  cudaStat = cudaMalloc((void**)& dIj, k * k * sizeof(double));
  stat = cublasSetMatrix(k, k, sizeof(double), Ii.memptr(), k, dIi, k);
  stat = cublasSetMatrix(k, k, sizeof(double), Ij.memptr(), k, dIj, k);

  stat = cublasGetMatrix(n, k, sizeof(double), dU, n, U_raw, n);
  stat = cublasGetMatrix(k, l, sizeof(double), dS, k, S_raw, k);
  for (int i = 0; i < k; i++) {
    int indVi = (m * l) * i;
    double* dVi = &dV[indVi];
    stat = cublasGetMatrix(m, l, sizeof(double), dVi, m, V.slice(i).memptr(), m);
  }

  // U.print("U before: ");
  // S.print("S before: ");
  // V.print("V before: ");

  // mat gradUPos;
  // gradUPos.zeros(n, k);

  // mat gradUNeg;
  // gradUNeg.zeros(n, k);

  for (int iter_idx = 0; iter_idx < num_iter; iter_idx++) {

    double* dGradUPos;
    cudaStat = cudaMalloc((void**)& dGradUPos, n * k * sizeof(double));
    // stat = cublasSetMatrix(n, k, sizeof(double), gradUPos.memptr(), n, dGradUPos, n);
    cudaMemset(dGradUPos, 0, n * k * sizeof(double));

    // cublasGetMatrix(n, k, sizeof(double), dGradUPos, n, gradUPos.memptr(), n);

    double* dGradUNeg;
    cudaStat = cudaMalloc((void**)& dGradUNeg, n * k * sizeof(double));
    // stat = cublasSetMatrix(n, k, sizeof(double), gradUNeg.memptr(), n, dGradUNeg, n);
    cudaMemset(dGradUNeg, 0, n * k * sizeof(double));

    // cublasGetMatrix(n, k, sizeof(double), dGradUNeg, n, gradUNeg.memptr(), n);

    for (int i = 0; i < k; i++) {

      changeElem<<<1,1>>>(dIi, i*k + i, 1.0);

      for (int j = 0; j < k; j++) {

        changeElem<<<1,1>>>(dIj, j*k + j, 1.0);

        int indVi = (m * l) * i;
        double* dVi = &dV[indVi];
        int indVj = (m * l) * j;
        double* dVj = &dV[indVj];

        // cout << "Computing U...\n";

        // cout << "  Computing dVitVj...";
        // mat VitVj;
        // VitVj.zeros(l, l);
        double* dVitVj;
        cudaStat = cudaMalloc((void**)& dVitVj, l * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &al, dVi, m, dVj, m, &bet, dVitVj, l);

        // cublasGetMatrix(l, l, sizeof(double), dVitVj, l, VitVj.memptr(), l);
        // VitVj.print("VitVj: ");
        // cout << " Done.\n";

        // cout << "  Computing dStIj...";
        // mat StIj;
        // StIj.zeros(l, k);

        double* dStIj;
        cudaStat = cudaMalloc((void**)& dStIj, l * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, k, k, &al, dS, k, dIj, k, &bet, dStIj, l);

        // cublasGetMatrix(l, k, sizeof(double), dStIj, l, StIj.memptr(), l);
        // StIj.print("StIj: ");
        // cout << " Done.\n";

        // cout << "  Computing dVitVjStIj...";
        // mat VitVjStIj;
        // VitVjStIj.zeros(l, k);

        double* dVitVjStIj;
        cudaStat = cudaMalloc((void**)& dVitVjStIj, l * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, l, k, l, &al, dVitVj, l, dStIj, l, &bet, dVitVjStIj, l);

        // cublasGetMatrix(l, k, sizeof(double), dVitVjStIj, l, VitVjStIj.memptr(), l);
        // VitVjStIj.print("VitVjStIj: ");
        // cout << " Done.\n";

        cudaFree(dVitVj);
        cudaFree(dStIj);

        // cout << "  Computing dSVitVjStIj...";
        // mat SVitVjStIj;
        // SVitVjStIj.zeros(k, k);

        double* dSVitVjStIj;
        cudaStat = cudaMalloc((void**)& dSVitVjStIj, k * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, k, l, &al, dS, k, dVitVjStIj, l, &bet, dSVitVjStIj, k);

        // cublasGetMatrix(k, k, sizeof(double), dSVitVjStIj, k, SVitVjStIj.memptr(), k);
        // SVitVjStIj.print("SVitVjStIj: ");
        // cout << " Done.\n";

        cudaFree(dVitVjStIj);

        // cout << "  Computing dIiSVitVjStIj...";
        // mat SVitVjStIj;
        // SVitVjStIj.zeros(k, k);

        double* dIiSVitVjStIj;
        cudaStat = cudaMalloc((void**)& dIiSVitVjStIj, k * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, k, k, &al, dIi, k, dSVitVjStIj, k, &bet, dIiSVitVjStIj, k);

        // cublasGetMatrix(k, k, sizeof(double), dSVitVjStIj, k, SVitVjStIj.memptr(), k);
        // SVitVjStIj.print("SVitVjStIj: ");
        // cout << " Done.\n";

        cudaFree(dSVitVjStIj);

        // cout << "  Computing dUIiSVitVjStIj...";
        // mat UIiSVitVjStIj;
        // UIiSVitVjStIj.zeros(n, k);

        double* dUIiSVitVjStIj;
        cudaStat = cudaMalloc((void**)& dUIiSVitVjStIj, n * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k, &al, dU, n, dIiSVitVjStIj, k, &bet, dUIiSVitVjStIj, n);

        // cublasGetMatrix(n, k, sizeof(double), dUIiSVitVjStIj, n, UIiSVitVjStIj.memptr(), n);
        // UIiSVitVjStIj.print("UIiSVitVjStIj: ");
        // cout << " Done.\n";

        cudaFree(dIiSVitVjStIj);

        // cout << "  Computing dGradUPos...";
        long nk = n * k;
        long num_blocks = (nk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sumElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradUPos, dUIiSVitVjStIj, dGradUPos, nk);
        // cout << " Done.\n";

        cudaFree(dUIiSVitVjStIj);

        // cublasGetMatrix(n, k, sizeof(double), dGradUPos, n, gradUPos.memptr(), n);
        // gradUPos.print("gradUPos: ");

        changeElem<<<1,1>>>(dIj, j*k + j, 0.0);

      }

      // cout << "  Computing dStIi...";
      double* dStIi;
      cudaStat = cudaMalloc((void**)& dStIi, l * k * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, k, k, &al, dS, k, dIi, k, &bet, dStIi, l);
      // cout << " Done.\n";

      int indVi = (m * l) * i;
      double* dVi = &dV[indVi];

      // cout << "  Computing dViStIi...";
      double* dViStIi;
      cudaStat = cudaMalloc((void**)& dViStIi, m * k * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &al, dVi, m, dStIi, l, &bet, dViStIi, m);
      // cout << " Done.\n";

      cudaFree(dStIi);

      // cout << "  Computing dXViStIi...";
      double* dXViStIi;
      cudaStat = cudaMalloc((void**)& dXViStIi, n * k * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &al, dX, n, dViStIi, m, &bet, dXViStIi, n);
      // cout << " Done.\n";

      cudaFree(dViStIi);

      // cout << "  Computing dGradUNeg...";
      long nk = n * k;
      long num_blocks = (nk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      sumElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradUNeg, dXViStIi, dGradUNeg, nk);
      // cout << " Done.\n";

      // cublasGetMatrix(n, k, sizeof(double), dGradUNeg, n, gradUNeg.memptr(), n);
      // gradUNeg.print("gradUNeg: ");

      cudaFree(dXViStIi);

      changeElem<<<1,1>>>(dIi, i*k + i, 0.0);

    }

    // cout << "  Computing dGradU...";
    double* dGradU;
    cudaStat = cudaMalloc((void**)& dGradU, n * k * sizeof(double));
    long nk = n * k;
    long num_blocks = (nk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradUNeg, dGradUPos, dGradU, nk);
    // cout << " Done.\n";

    cudaFree(dGradUNeg);
    cudaFree(dGradUPos);

    // cout << "  Updating dU...";
    multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dU, dGradU, dU, nk);
    // cout << " Done.\n";

    cudaFree(dGradU);

    // cout << "Done!!!\n";

    // cout << "Computing Vs...\n";
    for (int i = 0; i < k; i++) {

      changeElem<<<1,1>>>(dIi, i*k + i, 1.0);

      // mat gradVPos;
      // gradVPos.zeros(m, l);

      double* dGradVPos;
      cudaStat = cudaMalloc((void**)& dGradVPos, m * l * sizeof(double));
      cudaMemset(dGradVPos, 0, m * l * sizeof(double));

      for (int j = 0; j < k; j++) {

        changeElem<<<1,1>>>(dIj, j*k + j, 1.0);

        // cout << "  Computing dIiS...";
        double* dIiS;
        cudaStat = cudaMalloc((void**)& dIiS, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIi, k, dS, k, &bet, dIiS, k);
        // cout << " Done.\n";

        // cout << "  Computing dUtU...";
        double* dUtU;
        cudaStat = cudaMalloc((void**)& dUtU, k * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &al, dU, n, dU, n, &bet, dUtU, k);
        // cout << " Done.\n";

        // cout << "  Computing dUtUIiS...";
        // As IiS is already computed
        double* dUtUIiS;
        cudaStat = cudaMalloc((void**)& dUtUIiS, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dUtU, k, dIiS, k, &bet, dUtUIiS, k);
        // cout << " Done.\n";

        cudaFree(dIiS);
        cudaFree(dUtU);

        // cout << "  Computing dIjUtUIiS...";
        double* dIjUtUIiS;
        cudaStat = cudaMalloc((void**)& dIjUtUIiS, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIj, k, dUtUIiS, k, &bet, dIjUtUIiS, k);
        // cout << " Done.\n";

        cudaFree(dUtUIiS);

        // cout << "  Computing dStIiUtUIiS...";
        double* dStIjUtUIiS;
        cudaStat = cudaMalloc((void**)& dStIjUtUIiS, l * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, l, k, &al, dS, k, dIjUtUIiS, k, &bet, dStIjUtUIiS, l);
        // cout << " Done.\n";

        cudaFree(dIjUtUIiS);

        int indVj = (m * l) * j;
        double* dVj = &dV[indVj];

        // cout << "  Computing dVjStIjUtUIiS...";
        double* dVjStIjUtUIiS;
        cudaStat = cudaMalloc((void**)& dVjStIjUtUIiS, m * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &al, dVj, m, dStIjUtUIiS, l, &bet, dVjStIjUtUIiS, m);
        // cout << " Done.\n";

        cudaFree(dStIjUtUIiS);

        // cout << "  Computing dGradVPos...";
        long ml = m * l;
        long num_blocks = (ml + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sumElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradVPos, dVjStIjUtUIiS, dGradVPos, ml);

        // cublasGetMatrix(m, l, sizeof(double), dGradVPos, m, gradVPos.memptr(), n);
        // gradVPos.print("dGradVPos: ");

        // cout << " Done.\n";

        cudaFree(dVjStIjUtUIiS);

        changeElem<<<1,1>>>(dIj, j*k + j, 0.0);

      }

      // cout << "  Computing dXtU...";
      double* dXtU;
      cudaStat = cudaMalloc((void**)& dXtU, m * k * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &al, dX, n, dU, n, &bet, dXtU, m);
      // cout << " Done.\n";

      // cout << "  Computing dIiS...";
      double* dIiS;
      cudaStat = cudaMalloc((void**)& dIiS, k * l * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIi, k, dS, k, &bet, dIiS, k);
      // cout << " Done.\n";

      // cout << "  Computing dXtUIiS...";
      double* dXtUIiS;
      cudaStat = cudaMalloc((void**)& dXtUIiS, m * l * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, k, &al, dXtU, m, dIiS, k, &bet, dXtUIiS, m);
      // cout << " Done.\n";

      cudaFree(dXtU);
      cudaFree(dIiS);

      int indVi = (m * l) * i;
      double* dVi = &dV[indVi];

      // cout << "  Computing dGradVi...";
      double* dGradVi;
      cudaStat = cudaMalloc((void**)& dGradVi, m * l * sizeof(double));
      long ml = m * l;
      long num_blocks = (ml + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dXtUIiS, dGradVPos, dGradVi, ml);
      // cout << " Done.\n";

      cudaFree(dXtUIiS);
      cudaFree(dGradVPos);

      // cout << "  Updating dVi...";
      num_blocks = (ml + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dVi, dGradVi, dVi, ml);
      // cout << " Done.\n";

      cudaFree(dGradVi);

      changeElem<<<1,1>>>(dIi, i*k + i, 0.0);

    }

    // cout << "Done!!!\n";

    // cout << "Computing S...\n";
    // mat gradSPos;
    // gradSPos.zeros(k, l);
    // mat gradSNeg;
    // gradSNeg.zeros(k, l);


    double* dGradSPos;
    cudaStat = cudaMalloc((void**)& dGradSPos, k * l * sizeof(double));
    cudaStat = cudaMemset(dGradSPos, 0, k * l * sizeof(double));

    double* dGradSNeg;
    cudaStat = cudaMalloc((void**)& dGradSNeg, k * l * sizeof(double));
    cudaStat = cudaMemset(dGradSNeg, 0, k * l * sizeof(double));

    for (int i = 0; i < k; i++) {

      changeElem<<<1,1>>>(dIi, i*k + i, 1.0);

      for (int j = 0; j < k; j++) {

        changeElem<<<1,1>>>(dIj, j*k + j, 1.0);

        int indVi = (m * l) * i;
        double* dVi = &dV[indVi];

        int indVj = (m * l) * j;
        double* dVj = &dV[indVj];

        // compute UtUSVtV
        // cout << "  Computing dVjtVi...";
        double* dVjtVi;
        cudaStat = cudaMalloc((void**)& dVjtVi, l * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &al, dVj, m, dVi, m, &bet, dVjtVi, l);
        // cout << " Done.\n";

        // cout << "  Computing dSVjtVi...";
        double* dSVjtVi;
        cudaStat = cudaMalloc((void**)& dSVjtVi, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, l, &al, dS, k, dVjtVi, l, &bet, dSVjtVi, k);
        // cout << " Done.\n";

        cudaFree(dVjtVi);

        // cout << "  Computing dIjSVjtVi...";
        double* dIjSVjtVi;
        cudaStat = cudaMalloc((void**)& dIjSVjtVi, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIj, k, dSVjtVi, k, &bet, dIjSVjtVi, k);
        // cout << " Done.\n";

        cudaFree(dSVjtVi);

        // cout << "  Computing dUtU...";
        double* dUtU;
        cudaStat = cudaMalloc((void**)& dUtU, k * k * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &al, dU, n, dU, n, &bet, dUtU, k);
        // cout << " Done.\n";

        // cout << "  Computing dUtUIjSVjtVi...";
        double* dUtUIjSVjtVi;
        cudaStat = cudaMalloc((void**)& dUtUIjSVjtVi, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dUtU, k, dIjSVjtVi, k, &bet, dUtUIjSVjtVi, k);
        // cout << " Done.\n";

        cudaFree(dIjSVjtVi);
        cudaFree(dUtU);

        // cout << "  Computing dIiUtUIjSVjtVi...";
        double* dIiUtUIjSVjtVi;
        cudaStat = cudaMalloc((void**)& dIiUtUIjSVjtVi, k * l * sizeof(double));
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIi, k, dUtUIjSVjtVi, k, &bet, dIiUtUIjSVjtVi, k);
        // cout << " Done.\n";

        cudaFree(dUtUIjSVjtVi);

        // cout << "  Computing dGradSPos...";
        long kl = k * l;
        long num_blocks = (kl + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        sumElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradSPos, dIiUtUIjSVjtVi, dGradSPos, kl);

        // cublasGetMatrix(k, l, sizeof(double), dGradSPos, k, gradSPos.memptr(), k);
        // gradSPos.print("dGradSPos: ");
        // cout << " Done.\n";

        cudaFree(dIiUtUIjSVjtVi);

        changeElem<<<1,1>>>(dIj, j*k + j, 0.0);

      }


      // cout << "  Computing dUtX...";
      double* dUtX;
      cudaStat = cudaMalloc((void**)& dUtX, k * m * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &al, dU, n, dX, n, &bet, dUtX, k);
      // cout << " Done.\n";

      int indVi = (m * l) * i;
      double* dVi = &dV[indVi];

      // cout << "  Computing dUtXVi...";
      double* dUtXVi;
      cudaStat = cudaMalloc((void**)& dUtXVi, k * l * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, m, &al, dUtX, k, dVi, m, &bet, dUtXVi, k);
      // cout << " Done.\n";

      cudaFree(dUtX);

      // cout << "  Computing dIiUtXVi...";
      double* dIiUtXVi;
      cudaStat = cudaMalloc((void**)& dIiUtXVi, k * l * sizeof(double));
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dIi, k, dUtXVi, k, &bet, dIiUtXVi, k);
      // cout << " Done.\n";

      cudaFree(dUtXVi);

      // cout << "  Computing dGradSNeg...";
      long kl = k * l;
      long num_blocks = (kl + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      sumElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradSNeg, dIiUtXVi, dGradSNeg, kl);

      // cublasGetMatrix(k, l, sizeof(double), dGradSNeg, k, gradSNeg.memptr(), k);
      // gradSNeg.print("dGradSNeg: ");
      // cout << " Done.\n";

      cudaFree(dIiUtXVi);

      changeElem<<<1,1>>>(dIi, i*k + i, 0.0);

    }

    // compute S
    // cout << "  Computing dGradS...";
    double* dGradS;
    cudaStat = cudaMalloc((void**)& dGradS, k * l * sizeof(double));
    long kl = k * l;
    num_blocks = (kl + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dGradSNeg, dGradSPos, dGradS, kl);
    // cout << " Done.\n";

    cudaFree(dGradSNeg);
    cudaFree(dGradSPos);

    // cout << "  Updating S...";
    multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dS, dGradS, dS, kl);
    // cout << " Done.\n";

    cudaFree(dGradS);

    // cout << "Done!!!\n";

    stat = cublasGetMatrix(n, k, sizeof(double), dU, n, U_raw, n);
    stat = cublasGetMatrix(k, l, sizeof(double), dS, k, S_raw, k);
    for (int i = 0; i < k; i++) {
      int indVi = (m * l) * i;
      double* dVi = &dV[indVi];
      stat = cublasGetMatrix(m, l, sizeof(double), dVi, m, V.slice(i).memptr(), m);
    }

    // Compute V_tilde
    V_tilde.zeros(k, m);
    for (int i=0; i < k; i++)
      V_tilde.row(i) = S.row(i) * V.slice(i).t();

    error_ant = error;
    error = accu(pow((X - U * V_tilde), 2));

    std::cout << "Error obtained: " << error << std::endl;

    if (error < error_best) {
      U_best = U;
      S_best = S;
      V_best = V;
      V_tilde_best = V_tilde;
      error_best = error;
    }

    double precision_err = 0.000001;
    if (abs(error - error_ant) <= precision_err) break;

  }

  cublasDestroy(handle);

  // mat Du = diagmat((ones<rowvec>(n) * U_best));
  // mat Dv = diagmat((ones<rowvec>(m) * V_best));

  // Du.print("Du: ");
  // Dv.print("Dv: ");

  // U_norm = U_best * (diagmat(S_best * Dv * ones<colvec>(l)));
  // V_norm = V_best * (diagmat(ones<rowvec>(k) * Du * S_best));

  mat Reconstruction = U_best * V_tilde_best;

  U_best.save("U.csv", csv_ascii);
  S_best.save("S.csv", csv_ascii);
  V_best.save("V.hdf5", hdf5_binary);

  Reconstruction.save("Reconstruction.csv", csv_ascii);

  ofstream errorfile("error.csv");
  errorfile << std::fixed << std::setprecision(8) << error_best;
  errorfile.close();

}

void onmtf(mat X, const long& k, const long& l, const long& num_iter) {
  long n = X.n_rows;
  long m = X.n_cols;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double al = 1.0;
  double bet = 0.0;
  const long THREADS_PER_BLOCK = 1024;

  double* X_raw = X.memptr();

  mat U(n, k);
  mat S(k, l);
  mat V(m, l);

  double* U_raw = U.memptr();
  double* S_raw = S.memptr();
  double* V_raw = V.memptr();
  for(int i=0; i < n*k; i++) U_raw[i] = unif(gen);
  for(int i=0; i < k*l; i++) S_raw[i] = unif(gen);
  for(int i=0; i < m*l; i++) V_raw[i] = unif(gen);

  // U.print("U before: ");
  // S.print("S before: ");
  // V.print("V before: ");

  mat U_best;
  U_best.zeros(n, k);
  mat S_best;
  S_best.zeros(k, l);
  mat V_best;
  V_best.zeros(m, l);

  mat U_norm;
  U_norm.zeros(n, k);
  mat V_norm;
  V_norm.zeros(m, l);

  double error_best = DBL_MAX;
  double error_ant = DBL_MAX;
  double error = DBL_MAX;

  // cout << "Creating Cublas Handler...";
  stat = cublasCreate(&handle);
  // cout << " created.\n";

  // cout << "Creating X, U, S, V into GPU...";
  double* dX;
  double* dU;
  double* dS;
  double* dV;
  cudaStat = cudaMalloc((void**)& dX, n * m * sizeof(double));
  cudaStat = cudaMalloc((void**)& dU, n * k * sizeof(double));
  cudaStat = cudaMalloc((void**)& dS, k * l * sizeof(double));
  cudaStat = cudaMalloc((void**)& dV, m * l * sizeof(double));
  // cout << " created.\n";

  // cout << "Allocating X, U, S, V into GPU...";
  stat = cublasSetMatrix(n, m, sizeof(double), X_raw, n, dX, n);
  stat = cublasSetMatrix(n, k, sizeof(double), U_raw, n, dU, n);
  stat = cublasSetMatrix(k, l, sizeof(double), S_raw, k, dS, k);
  stat = cublasSetMatrix(m, l, sizeof(double), V_raw, m, dV, m);
  // cout << " allocated.\n";

  for (int iter_idx = 0; iter_idx < num_iter; iter_idx++) {

    // compute XVSt
    // cout << "Computing VSt...";
    double* dVt;
    cudaStat = cudaMalloc((void**)& dVt, m * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, l, &al, dV, m, dS, k, &bet, dVt, m);
    // cout << " done.\n";

    // cout << "Computing XVSt...";
    double* dXVSt;
    cudaStat = cudaMalloc((void**)& dXVSt, n * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &al, dX, n, dVt, m, &bet, dXVSt, n);
    // cout << " done.\n";

    cudaFree(dVt);

    // compute USVtXtU
    // cout << "Computing dXtU...";
    double* dXtU;
    cudaStat = cudaMalloc((void**)& dXtU, m * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &al, dX, n, dU, n, &bet, dXtU, m);
    // cout << " done.\n";

    // cout << "Computing VtXtU...";
    double* dVtXtU;
    cudaStat = cudaMalloc((void**)& dVtXtU, l * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, k, m, &al, dV, m, dXtU, m, &bet, dVtXtU, l);
    // cout << " done.\n";

    cudaFree(dXtU);

    // cout << "Computing SVtXtU...";
    double* dSVtXtU;
    cudaStat = cudaMalloc((void**)& dSVtXtU, k * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, k, l, &al, dS, k, dVtXtU, l, &bet, dSVtXtU, k);
    // cout << " done.\n";

    cudaFree(dVtXtU);

    // cout << "Computing dUSVtXtU...";
    double* dUSVtXtU;
    cudaStat = cudaMalloc((void**)& dUSVtXtU, n * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k, &al, dU, n, dSVtXtU, k, &bet, dUSVtXtU, n);
    // cout << " done.\n";

    cudaFree(dSVtXtU);

    // compute U
    // cout << "Computing Grad U...";
    double* dGradU;
    cudaStat = cudaMalloc((void**)& dGradU, n * k * sizeof(double));
    long nk = n * k;
    long num_blocks = (nk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dXVSt, dUSVtXtU, dGradU, nk);
    // cout << " done.\n";

    cudaFree(dXVSt);
    cudaFree(dUSVtXtU);

    // cout << "Computing new U...";
    multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dU, dGradU, dU, nk);
    // cout << " done.\n";

    cudaFree(dGradU);

    // compute XtUS
    // cout << "Computing dXtU...";
    cudaStat = cudaMalloc((void**)& dXtU, m * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &al, dX, n, dU, n, &bet, dXtU, m);
    // cout << " done.\n";

    // cout << "Computing XtUS...";
    double* dXtUS;
    cudaStat = cudaMalloc((void**)& dXtUS, m * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, k, &al, dXtU, m, dS, k, &bet, dXtUS, m);
    // cout << " done.\n";

    cudaFree(dXtU);

    // compute VStUtXV
    // cout << "Computing XtU...";
    double* dUtX;
    cudaStat = cudaMalloc((void**)& dUtX, k * m * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &al, dU, n, dX, n, &bet, dUtX, k);
    // cout << " done.\n";

    // cout << "Computing dXtXV...";
    double* dUtXV;
    cudaStat = cudaMalloc((void**)& dUtXV, k * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, m, &al, dUtX, k, dV, m, &bet, dUtXV, k);
    // cout << " done.\n";

    cudaFree(dUtX);

    // cout << "Computing StUtXV...";
    double* dStUtXV;
    cudaStat = cudaMalloc((void**)& dStUtXV, l * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, l, k, &al, dS, k, dUtXV, k, &bet, dStUtXV, l);
    // cout << " done.\n";

    cudaFree(dUtXV);

    // cout << "Computing VStUtXV...";
    double* dVtUtXV;
    cudaStat = cudaMalloc((void**)& dVtUtXV, m * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &al, dV, m, dStUtXV, l, &bet, dVtUtXV, m);
    // cout << " done.\n";

    cudaFree(dStUtXV);

    // compute V
    // cout << "Computing Grad V...";
    double* dGradV;
    cudaStat = cudaMalloc((void**)& dGradV, m * l * sizeof(double));
    long ml = m * l;
    num_blocks = (ml + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dXtUS, dVtUtXV, dGradV, ml);
    // cout << " done.\n";

    cudaFree(dXtUS);
    cudaFree(dVtUtXV);

    // cout << "Computing new V...";
    multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dV, dGradV, dV, ml);
    // cout << " done.\n";

    cudaFree(dGradV);

    // compute UtX
    // double* dUtX;
    cudaStat = cudaMalloc((void**)& dUtX, k * m * sizeof(double));
    // cout << "Computing UtX...";
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &al, dU, n, dX, n, &bet, dUtX, k);
    // cout << " done.\n";

    // double* dUtXV;
    cudaStat = cudaMalloc((void**)& dUtXV, k * l * sizeof(double));
    // cout << "Computing XtXV...";
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, m, &al, dUtX, k, dV, m, &bet, dUtXV, k);
    // cout << " done.\n";

    cudaFree(dUtX);

    // compute UtUSVtV
    // cout << "Computing VtV...";
    double* dVtV;
    cudaStat = cudaMalloc((void**)& dVtV, l * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &al, dV, m, dV, m, &bet, dVtV, l);
    // cout << " done.\n";

    // cout << "Computing SVtV...";
    double* dSVtV;
    cudaStat = cudaMalloc((void**)& dSVtV, k * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, l, &al, dS, k, dVtV, l, &bet, dSVtV, k);
    // cout << " done.\n";

    cudaFree(dVtV);

    // cout << "Computing UtU...";
    double* dUtU;
    cudaStat = cudaMalloc((void**)& dUtU, k * k * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &al, dU, n, dU, n, &bet, dUtU, k);
    // cout << " done.\n";

    // cout << "Computing UtUSVtV...";
    double* dUtUSVtV;
    cudaStat = cudaMalloc((void**)& dUtUSVtV, k * l * sizeof(double));
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, l, k, &al, dUtU, k, dSVtV, k, &bet, dUtUSVtV, k);
    // cout << " done.\n";

    cudaFree(dSVtV);
    cudaFree(dUtU);

    // compute S
    // cout << "Computing Grad S...";
    double* dGradS;
    cudaStat = cudaMalloc((void**)& dGradS, k * l * sizeof(double));
    long kl = k * l;
    num_blocks = (kl + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    divideElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dUtXV, dUtUSVtV, dGradS, kl);
    // cout << " done.\n";

    cudaFree(dUtXV);
    cudaFree(dUtUSVtV);

    // cout << "Computing new S...";
    multiplyElemWise<<<num_blocks, THREADS_PER_BLOCK>>>(dS, dGradS, dS, kl);
    // cout << " done.\n";

    stat = cublasGetMatrix(n, k, sizeof(double), dU, n, U_raw, n);
    stat = cublasGetMatrix(k, l, sizeof(double), dS, k, S_raw, k);
    stat = cublasGetMatrix(m, l, sizeof(double), dV, m, V_raw, m);

    error_ant = error;
    error = accu(pow((X - U * S * V.t()), 2));

    std::cout << "Error obtained: " << error << std::endl;

    if (error < error_best) {
      U_best = U;
      S_best = S;
      V_best = V;
      error_best = error;
    }

    double precision_err = 0.000001;
    if (abs(error - error_ant) <= precision_err) break;

  }

  cublasDestroy(handle);

  mat Du = diagmat((ones<rowvec>(n) * U_best));
  mat Dv = diagmat((ones<rowvec>(m) * V_best));

  // Du.print("Du: ");
  // Dv.print("Dv: ");

  U_norm = U_best * (diagmat(S_best * Dv * ones<colvec>(l)));
  V_norm = V_best * (diagmat(ones<rowvec>(k) * Du * S_best));

  U_norm.save("U.csv", csv_ascii);
  S_best.save("S.csv", csv_ascii);
  V_norm.save("V.csv", csv_ascii);

  ofstream errorfile("error.csv");
  errorfile << std::fixed << std::setprecision(8) << error_best;
  errorfile.close();

}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    cout << "Missing arguments...\n";
    cout << argc;
    return 1;
  }
  cout.precision(11);
  cout.setf(ios::fixed);

  string algo_name = argv[1];
  const long k = atoi(argv[2]);
  const long l = atoi(argv[3]);
  const long num_iter = atoi(argv[4]);

  std::cout << "Reading X...";
  mat X;
  // X.load("data.csv");
  X.load("data.h5", hdf5_binary);
  std::cout << "  read " << X.n_rows << " rows and " << X.n_cols << " cols!\n";

  // X.print("X before: ");

  // X.raw_print(cout, "X: ");

  clock_t begin = clock();

  if (algo_name == "onmtf")
    onmtf(X, k, l, num_iter);
  else if (algo_name == "ovnmtf")
    ovnmtf(X, k, l, num_iter);
  else
    cout << "Wrong algo name...\n";

  clock_t end = clock();

  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  std::cout << "Time taken: " << elapsed_secs << std::endl;

  return EXIT_SUCCESS;
}
