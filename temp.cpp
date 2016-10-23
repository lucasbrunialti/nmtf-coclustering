#include <limits>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <armadillo>
#include <random>
#include <typeinfo>

using namespace std;
using namespace arma;

int main() {
  // rowvec a = ones<rowvec>(4);
  int k = 2;
  int l = 3;
  // int sl = 2;
  // cube B(k, l, sl);
  // double* B_raw = B.memptr();
  // for (int i = 0; i < k*l*sl; i++) {
  //   B_raw[i] = i;
  // }
  // B.print("B: ");


  // double** dVs = new double*[l];
  // cout << "dVs: " << dVs << ", type: " << typeid(dVs).name() << endl;
  // cout << "dVs[0]: " << dVs[0] << ", type: " << typeid(dVs[0]).name() << endl;
  // cout << "dVs[1]: " << dVs[1] << ", type: " << typeid(dVs[1]).name() << endl;
  // cout << "dVs[2]: " << dVs[2] << ", type: " << typeid(dVs[2]).name() << endl;
  // cout << "&dVs[0]: " << &dVs[0] << ", type: " << typeid(&dVs[0]).name() << endl;
  // cout << "&dVs[1]: " << &dVs[1] << ", type: " << typeid(&dVs[1]).name() << endl;
  // cout << "&dVs[2]: " << &dVs[2] << ", type: " << typeid(&dVs[2]).name() << endl;


  // double* x;
  // cout << "dVs: " << x[110] << ", type: " << typeid(x[110]).name() << endl;

  // mat C = { {1, 0, 0},
  //           {1, 0, 0},
  //           {0, 1, 0},
  //           {0, 1, 0},
  //           {0, 1, 1} };
  // C.t().print("C: ");

  // mat A = B.slice(0) * C.t();

  // A.print("A: ");

  // C.row(0) = C.row(4);
  // for (int i = 0; i < 3; i++)
  //   C.rows( find(C.col(i)  == 1.0) ).print("C i=" + to_string(i) + ": ");


  // cout << diagmat((a * B)) << endl;

//   int n = 10000;
//   int m = 10000;
//   int res = n*m;
//   cout << (res) << endl;
//     const int max_int = std::numeric_limits<int>::max();
// cout << (max_int) << endl;


  // B.print("B: ");
  // double* B_raw = B.memptr();

  // for (int i = 0; i < k*l*sl; i++) {
  //   cout << B_raw[i] << " ";
  // }


  // C = B;
  // B_raw[0] = 0.99999999;

  // B.print("B new: ");

  // C.print("C: ");
  // cout << B_raw[0] << endl;
  // cout << B_raw[1] << endl;
  // cout << B_raw[4] << endl;
  // cout << B_raw << endl;

  // B_raw[4] = 0.0;

  // B.print("B before: ");

  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<double> unif(0.0, 1.0);
  // for (int j=0; j < 1000; j++) {
  //   for (int i=0; i<k*l*sl; i++) B.memptr()[i] = unif(gen);
  //   B.print("B after: ");
  // }

  // for (int i= 0; i < 1000; i++)
  //   cout << unif(re) << endl;

  // double* A;
  // for (int i = 0; i < 2; i++) {
  //   A = (double*) malloc(5 * sizeof(double));
  //   for (int j = 0; j < 5; j++) A[j] = i;
  //   for (int j = 0; j < 5; j++) cout << A[j] << ", ";
  //   cout << endl;
  //   free(A);

  //   for (int j = 0; j < 5; j++) cout << A[j] << ", ";
  // }

  // int mil = mil;
  // cout << (1000/1024);

  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::uniform_real_distribution<double> unif(0.0, 1.0);
  // std::uniform_int_distribution<int> unifint(0, 1);
  // for (int i = 0; i< 10000; i++)
  //   cout << unifint(gen) << endl;

  // return 0;
}
