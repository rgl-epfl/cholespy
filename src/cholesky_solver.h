#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"

enum MatrixType {
    CSC = 0,
    CSR,
    COO
};

/**
 * Cholesky Solver Class
 *
 * Takes as in put an arbitrary COO, CSC or CSR matrix, and factorizes it using
 * CHOLMOD.
 */
template<typename Float>
class CholeskySolver {
public:
    /**
     * Build the solver
     *
     * @param n_rows The number of rows in the matrix
     * @param nnz The number of nonzero entries
     * @param ii Array of row indices if type==COO, column (resp. row) pointer array if type==CSC (resp. CSR)
     * @param ii Array of row indices if type==COO or CSC, column indices if type==CSR
     * @param x Array of nonzero entries
     * @param type The type of the matrix representation. Can be COO, CSC or CSR
     */
    CholeskySolver(int n_rows, int nnz, int *ii, int *jj, double *x, MatrixType type);

    ~CholeskySolver();

    // Solve the whole system using the Cholesky factorization on the CPU
    void solve_cpu(int n_rhs, Float *b, Float *x, int mode);


private:

    // Factorize the CSC matrix using CHOLMOD
    void factorize(int *col_ptr, int *rows, double *data);

    int m_nrhs = 0;
    int m_n;
    int m_nnz;

    // Pointers used for the analysis, freed if solving on the GPU, kept if solving on the CPU
    cholmod_factor *m_factor;
    cholmod_dense  *m_tmp_chol = nullptr;
    cholmod_common  m_common;
};
