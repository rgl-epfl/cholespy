#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"
#include "cuda_driver.h"

enum MatrixType {
    CSC = 0,
    CSR,
    COO
};

/**
 * Cholesky Solver Class
 *
 * Takes as in put an arbitrary COO, CSC or CSR matrix, and factorizes it using
 * CHOLMOD. If it receives CUDA arrays as input, it runs an analysis of the
 * factor on the GPU for faster, parallel solving of the triangular system in
 * the solving phase.
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
     * @param cpu Whether or not to run the CPU version of the solver.
     */
    CholeskySolver(int n_rows, int nnz, int *ii, int *jj, double *x, MatrixType type, bool cpu);

    ~CholeskySolver();

    // Solve the whole system using the Cholesky factorization on the GPU
    void solve_cuda(int n_rhs, CUdeviceptr b, CUdeviceptr x);

    // Solve the whole system using the Cholesky factorization on the CPU
    void solve_cpu(int n_rhs, Float *b, Float *x);

    // Return whether the solver solves on the CPU or on the GPU
    bool is_cpu() { return m_cpu; };

private:

    // Factorize the CSC matrix using CHOLMOD
    void factorize(int *col_ptr, int *rows, double *data);

    // Run the analysis of a triangular matrix obtained through Cholesky
    void analyze_cuda(int n_rows, int n_entries, void *csr_rows, void *csr_cols, Float *csr_data, bool lower);

	// Solve one triangular system
    void launch_kernel(bool lower, CUdeviceptr x);

    int m_nrhs = 0;
    int m_n;
    int m_nnz;

    // CPU or GPU solver?
    bool m_cpu;

    // Pointers used for the analysis, freed if solving on the GPU, kept if solving on the CPU
    cholmod_factor *m_factor;
    cholmod_dense  *m_tmp_chol = nullptr;
    cholmod_common  m_common;

    // Pointers used for the GPU variant

    // Permutation
    CUdeviceptr m_perm_d;

    // CSR Lower triangular
    CUdeviceptr m_lower_rows_d;
    CUdeviceptr m_lower_cols_d;
    CUdeviceptr m_lower_data_d;

    // CSR Upper triangular
    CUdeviceptr m_upper_rows_d;
    CUdeviceptr m_upper_cols_d;
    CUdeviceptr m_upper_data_d;

    // Mask of already processed rows, used for both analysis and solve
    CUdeviceptr m_processed_rows_d;

    // ID of current row being processed by a given block
    CUdeviceptr m_stack_id_d;

    // Sorted indices of rows in each level
    CUdeviceptr m_lower_levels_d;
    CUdeviceptr m_upper_levels_d;

    // Temporary array used for solving the triangular systems in place
    CUdeviceptr m_tmp_d = 0;
};
