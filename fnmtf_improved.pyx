
import cython
cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def fnmtf_improved(double[:, ::1] X, int k, int l, int num_iter=100, int norm=0):
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]

    cdef unsigned int i = 0
    cdef unsigned int j = 0
    cdef unsigned int iter_index = 0
    cdef unsigned int row_clust_ind = 0
    cdef unsigned int col_clust_ind = 0
    cdef unsigned int ind = 0

    cdef double[:, ::1] U = np.random.rand(m, k).astype(np.float64)
    cdef double[:, ::1] U_best = np.random.rand(m, k).astype(np.float64)
    cdef double[:, ::1] S = np.random.rand(k, l).astype(np.float64)
    cdef double[:, ::1] S_best = np.random.rand(k, l).astype(np.float64)
    cdef double[:, ::1] V = np.random.rand(n, l).astype(np.float64)
    cdef double[:, ::1] V_best = np.random.rand(n, l).astype(np.float64)

    cdef double[:, ::1] U_tilde = np.empty((m, l), dtype=np.float64)
    cdef double[:, ::1] V_new = np.empty((n, l), dtype=np.float64)

    cdef double[:, ::1] V_tilde = np.empty((l, n), dtype=np.float64)
    cdef double[:, ::1] U_new = np.empty((m, k), dtype=np.float64)

    cdef double error_best = 10e9999
    cdef double error = 10e9999
    cdef double[:] errors_v = np.zeros(l, dtype=np.float64)
    cdef double[:] errors_u = np.zeros(k, dtype=np.float64)

    for iter_index in range(num_iter):
        S[:, :] = np.dot( np.dot(np.linalg.pinv(np.dot(U.T, U)), np.dot(np.dot(U.T, X), V)), np.linalg.pinv(np.dot(V.T, V)) )

        # solve subproblem to update V
        U_tilde[:, :] = np.dot(U, S)
        V_new[:, :] = np.empty((n, l), dtype=np.int)
        for j in range(n):
            errors_v = np.zeros(l, dtype=np.float64)
            for col_clust_ind in range(l):
                errors_v[col_clust_ind] = np.sum(np.square(np.subtract(X[:, j], U_tilde[:, col_clust_ind])))
            ind = np.argmin(errors_v)
            V_new[j, ind] = 1.0
        V[:, :] = V_new

        # solve subproblem to update U
        V_tilde[:, :] = np.dot(S, V.T)
        U_new[:, :] = np.empty((m, k), dtype=np.int)
        for i in range(m):
            errors_u = np.zeros(k, dtype=np.float64)
            for row_clust_ind in range(k):
                errors_u[row_clust_ind] = np.sum(np.square(np.subtract(X[i, :], V_tilde[row_clust_ind, :])))
            ind = np.argmin(errors_u)
            U_new[i, ind] = 1.0
        U[:, :] = U_new

        error_ant = error
        error = np.sum(np.square(np.subtract(X, np.dot(np.dot(U, S), V.T))))

        if error < error_best:
            U_best[:, :] = U
            S_best[:, :] = S
            V_best[:, :] = V
            error_best = error
