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

template<typename Float>
class CholeskySolver {
public:
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
    cholmod_common m_common;

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
