
#include <limits>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <armadillo>
#include <random>

using namespace std;
using namespace arma;

#define DBL_MAX numeric_limits<double>::max()

void binovnmtf(const mat& X, const int& k, const int& l, const int& num_iter) {
  int n = X.n_rows;
  int m = X.n_cols;

  mat U(n, k);
  mat S(k, l);
  cube V(m, l, k);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::uniform_int_distribution<int> unifint(0, 1);
  std::uniform_int_distribution<int> unifinttom(0, l-1);

  double* U_raw = U.memptr();
  double* S_raw = S.memptr();
  double* V_raw = V.memptr();
  for(int i=0; i < n*k; i++) U_raw[i] = unifint(gen);
  for(int i=0; i < k*l; i++) S_raw[i] = unif(gen);
  for(int i=0; i < m*l*k; i++) V_raw[i] = unifint(gen);

  mat U_best;
  U_best.zeros(n, k);
  mat S_best;
  S_best.zeros(k, l);
  cube V_best;
  V_best.zeros(m, l, k);

  std::cout << "U S V created" << std::endl;

  mat X_tilde;
  X_tilde.zeros(n, m);
  // mat U_tilde;
  // U_tilde.zeros(n, l);
  // mat V_new;
  // V_new.zeros(m, l);
  mat V_tilde;
  V_tilde.zeros(k, m);
  mat V_tilde_best;
  V_tilde_best.zeros(k, m);
  // mat U_new;
  // U_new.zeros(m, k);

  rowvec errors_v;
  errors_v.zeros(l);

  rowvec errors_u;
  zeros(k);

  double error_best = DBL_MAX;
  double error_ant = DBL_MAX;
  double error = DBL_MAX;

  for (int iter_idx = 0; iter_idx < num_iter; iter_idx++) {
    // Compute S
    std::cout << "Computing S...";
    for (int i=0; i < k; i++) {
      for (int j=0; j < l; j++) {
        uvec observations_row_cluster_i = find(U.col(i) == 1.0);
        uvec observations_col_cluster_j = find(V.slice(i).col(j) == 1.0);

        if (observations_row_cluster_i.is_empty() || observations_col_cluster_j.is_empty())
          S(i, j) = 0.0;
        else
          S(i, j) = mean(mean(X.submat( observations_row_cluster_i, observations_col_cluster_j )));
      }
    }
    std::cout << " done!" << std::endl;

    // Compute V
    std::cout << "Computing V..." << std::endl;
    for (int i=0; i < k; i++) {
      std::cout << "  - for cluster " << i << "...";
      // uvec observations_cluster_i = find(U.col(i) == 1.0);

      // if (observations_cluster_i.is_empty()) {
      //   cout << "observations_cluster_i empty!!!!" << endl;
      //   for (int j=0; j < m; j++)
      //     V.slice(i)(j, unifinttom(gen)) = 1;
      //   continue;
      // }

      mat U_tilde = U.col(i) * S.row(i);
      // mat X_tilde = X.rows( observations_cluster_i );
      mat X_tilde = X;

      V.slice(i).zeros(m, l);
      for (int j=0; j < m; j++) {
        errors_v.zeros(l);
        for (int col_clust_idx = 0; col_clust_idx < l; col_clust_idx++)
          errors_v(col_clust_idx) = sum(pow(X_tilde.col(j) - U_tilde.col(col_clust_idx), 2));

        uword ind;
        errors_v.min(ind);

        V.slice(i)(j, ind) = 1;
      }

      std::cout << " done!" << std::endl;
    }
    std::cout << "Done computed V!" << std::endl;

    std::cout << "Computing U...";
    // Compute V_tilde
    V_tilde.zeros(k, m);
    for (int i=0; i < k; i++)
      V_tilde.row(i) = S.row(i) * V.slice(i).t();

    U.zeros(n, k);
    for (int i = 0; i < n; i++) {
      errors_u.zeros(k);
      for (int row_clust_idx = 0; row_clust_idx < k; row_clust_idx++)
        errors_u(row_clust_idx) =
            sum(pow(X.row(i) - V_tilde.row(row_clust_idx), 2));

      uword ind;
      errors_u.min(ind);

      U(i, ind) = 1;
    }

    std::cout << " done!" << std::endl;

    error_ant = error;
    error = accu(pow(X - (U * V_tilde), 2));

    std::cout << "Error obtained: " << error << std::endl;

    if (error < error_best) {
      U_best = U;
      S_best = S;
      V_best = V;
      V_tilde_best = V_tilde;
      error_best = error;
    }

    double precision_err = 0.0001;
    if (abs(error - error_ant) <= precision_err) break;
  }

  mat Reconstruction = U_best * V_tilde_best;

  U_best.save("U.csv", csv_ascii);
  S_best.save("S.csv", csv_ascii);
  V_best.save("V.hdf5", hdf5_binary);
  Reconstruction.save("Reconstruction.csv", csv_ascii);

  ofstream errorfile("error.csv");
  errorfile << std::fixed << std::setprecision(8) << error_best;
  errorfile.close();

}

void fnmtf(const mat& X, const int& k, const int& l, const int& num_iter) {
  int m = X.n_rows;
  int n = X.n_cols;

  mat U(m, k);
  mat S(k, l);
  mat V(n, l);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  double* U_raw = U.memptr();
  double* S_raw = S.memptr();
  double* V_raw = V.memptr();
  for(int i=0; i < m*k; i++) U_raw[i] = unif(gen);
  for(int i=0; i < k*l; i++) S_raw[i] = unif(gen);
  for(int i=0; i < n*l; i++) V_raw[i] = unif(gen);

  // U.print("U before: ");
  // S.print("S before: ");
  // V.print("V before: ");

  mat U_best;
  U_best.zeros(m, k);
  mat S_best;
  S_best.zeros(k, l);
  mat V_best;
  V_best.zeros(n, l);

  std::cout << "U S V created" << std::endl;
  // U.print("U:");
  // S.print("S:");
  // V.print("V:");

  mat U_tilde;
  U_tilde.zeros(m, l);
  mat V_new;
  V_new.zeros(n, l);
  mat V_tilde;
  V_tilde.zeros(k, m);
  mat U_new;
  U_new.zeros(m, k);

  rowvec errors_v;
  errors_v.zeros(l);

  rowvec errors_u;

  double error_best = DBL_MAX;
  double error_ant = DBL_MAX;
  double error = DBL_MAX;

  for (int iter_idx = 0; iter_idx < num_iter; iter_idx++) {
    std::cout << "Computing S...";
    S = pinv(U.t() * U) * (U.t() * (X * V)) * pinv(V.t() * V);
    std::cout << " OK!\n";

    std::cout << "Computing V...";
    U_tilde = U * S;
    V_new.zeros(n, l);

    for (int j = 0; j < n; j++) {
      errors_v.zeros(l);
      for (int col_clust_idx = 0; col_clust_idx < l; col_clust_idx++)
        errors_v(col_clust_idx) =
            sum(pow(X.col(j) - U_tilde.col(col_clust_idx), 2));

      uword ind;
      errors_v.min(ind);

      V_new(j, ind) = 1;
    }

    V = V_new;
    std::cout << " OK!\n";

    std::cout << "Computing U...";
    V_tilde = S * V.t();
    U_new.zeros(m, k);

    for (int i = 0; i < m; i++) {
      errors_u.zeros(k);
      for (int row_clust_idx = 0; row_clust_idx < k; row_clust_idx++)
        errors_u(row_clust_idx) =
            sum(pow(X.row(i) - V_tilde.row(row_clust_idx), 2));

      uword ind;
      errors_u.min(ind);

      U_new(i, ind) = 1;
    }

    U = U_new;
    std::cout << " OK!\n";

    error_ant = error;
    error = accu(pow(X - (U * S * V.t()), 2));

    std::cout << "Error obtained: " << error << "\n";

    if (error < error_best) {
      U_best = U;
      S_best = S;
      V_best = V;
      error_best = error;
    }

    double precision_err = 0.000001;
    if (abs(error - error_ant) <= precision_err) break;
  }

  U_best.save("U.csv", csv_ascii);
  S_best.save("S.csv", csv_ascii);
  V_best.save("V.csv", csv_ascii);

  ofstream errorfile("error.csv");
  errorfile << std::fixed << std::setprecision(8) << error_best;
  errorfile.close();
}

void onmtf(const mat& X, const int& k, const int& l, const int& num_iter) {
  int m = X.n_rows;
  int n = X.n_cols;

  mat U(m, k);
  mat S(k, l);
  mat V(n, l);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  double* U_raw = U.memptr();
  double* S_raw = S.memptr();
  double* V_raw = V.memptr();
  for(int i=0; i < m*k; i++) U_raw[i] = unif(gen);
  for(int i=0; i < k*l; i++) S_raw[i] = unif(gen);
  for(int i=0; i < n*l; i++) V_raw[i] = unif(gen);

  // U.print("U before: ");
  // S.print("S before: ");
  // V.print("V before: ");

  mat U_best;
  U_best.zeros(m, k);
  mat S_best;
  S_best.zeros(k, l);
  mat V_best;
  V_best.zeros(n, l);

  mat U_norm;
  U_norm.zeros(m, k);
  mat V_norm;
  V_norm.zeros(n, l);

  std::cout << "U S V created" << std::endl;

  double error_best = DBL_MAX;
  double error_ant = DBL_MAX;
  double error = DBL_MAX;

  for (int iter_idx = 0; iter_idx < num_iter; iter_idx++) {
    U = U % ((X * V * S.t()) / (U * S * V.t() * X.t() * U));
    V = V % ((X.t() * U * S) / (V * S.t() * U.t() * X * V));
    S = S % ((U.t() * X * V) / (U.t() * U * S * V.t() * V));

    error_ant = error;
    error = accu(pow((X - U * S * V.t()), 2));

    std::cout << "Error obtained: " << error << "\n";

    if (error < error_best) {
      U_best = U;
      S_best = S;
      V_best = V;
      error_best = error;
    }

    double precision_err = 0.000001;
    if (abs(error - error_ant) <= precision_err) break;
  }

  mat Du = diagmat((ones<rowvec>(m) * U_best));
  mat Dv = diagmat((ones<rowvec>(n) * V_best));

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
  const int k = atoi(argv[2]);
  const int l = atoi(argv[3]);
  const int num_iter = atoi(argv[4]);

  std::cout << "Reading X...";
  mat X;
  // X.load("data.csv");
  X.load("data.h5", hdf5_binary);
  std::cout << "  read " << X.n_rows << " rows and " << X.n_cols << " cols!\n";

  // X.raw_print(cout, "X: ");

  clock_t begin = clock();

  if (algo_name == "fnmtf")
    fnmtf(X, k, l, num_iter);
  else if (algo_name == "onmtf")
    onmtf(X, k, l, num_iter);
  else if (algo_name == "bin_ovnmtf")
    binovnmtf(X, k, l, num_iter);
  else
    cout << "Wrong algo name...\n";

  clock_t end = clock();

  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  std::cout << "Time taken: " << elapsed_secs << std::endl;

  return 0;
}
