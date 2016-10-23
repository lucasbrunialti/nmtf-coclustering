
#include <limits>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#define DBL_MAX numeric_limits<double>::max()

int main(int argc, char* argv[]) {
    int k = atoi(argv[1]);
    int l = atoi(argv[2]);
    int num_iter = atoi(argv[3]);

    fstream fsdata;
    fsdata.open("data.txt", ios::in | ios::binary);

    if (!fsdata.is_open()) {
        std::cerr << "There was a problem opening the input file!\n";
        exit(1);
    }

    int m;
    fsdata >> m;
    fsdata.get();
    int n;
    fsdata >> n;
    fsdata.get();

    // std::cout << "m: " << m << ", n: " << n << std::endl;

    MatrixXd X(m, n);

    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            fsdata >> X(i, j);
            fsdata.get();
        }
    }

    fsdata.close();

    MatrixXd U(m, k);
    MatrixXd S(k, l);
    MatrixXd V(n, l);

    U = MatrixXd::Random(m, k);
    S = MatrixXd::Random(k, l);
    V = MatrixXd::Random(n, l);

    std::cout << "U S V created" << std::endl;

    MatrixXd U_tilde(m, l);
    U_tilde = MatrixXd::Zero(m, l);

    MatrixXd V_new(n, l);
    V_new = MatrixXd::Zero(n, l);

    MatrixXd V_tilde(k, m);
    V_tilde = MatrixXd::Zero(k, m);

    MatrixXd U_new(m, k);
    U_new = MatrixXd::Zero(m, k);

    VectorXd errors_v;
    errors_v = VectorXd::Zero(l);
    VectorXd errors_u;
    errors_u = VectorXd::Zero(k);

    double error_best = DBL_MAX;
    double error = DBL_MAX;

    clock_t begin = clock();

    std::cout << U << std::endl;
    std::cout << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;
    std::cout << V << std::endl;
    std::cout << std::endl;

    for (int iter_index = 0; i < num_iter; num_iter++) {

        S =

    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Time taken : " << elapsed_secs << std::endl;

    return 0;
}
