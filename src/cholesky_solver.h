#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"
#include <cuda.h>
template<typename Float>
class CholeskySolver {
public:
    CholeskySolver(uint n_verts, uint n_faces, uint* faces, double lambda);

    ~CholeskySolver();

    // Solve the whole system using the Cholesky factorization
    std::vector<Float> solve(Float *b);

private:

    // Build the (I+λL) matrix in CSC representation
    void build_matrix(uint n_verts, uint n_faces, uint *faces, double lambda, std::vector<int> &col_ptr, std::vector<int> &rows, std::vector<double> &data);

    // Factorize the CSC matrix using CHOLMOD
    void factorize(std::vector<int> &col_ptr, std::vector<int> &rows, std::vector<double> &data);

    // Run the analysis of a triangular matrix obtained through Cholesky
    void analyze(uint n_rows, uint n_entries, void *csr_rows, void *csr_cols, Float* csr_data, bool lower);

	// Solve one triangular system
    void solve(bool lower);

    uint m_n;
    uint *m_perm;

    // CSR Lower triangular
    CUdeviceptr m_lower_rows_d;
    CUdeviceptr m_lower_cols_d;
    CUdeviceptr m_lower_data_d;

    // CSR Upper triangular
    CUdeviceptr m_upper_rows_d;
    CUdeviceptr m_upper_cols_d;
    CUdeviceptr m_upper_data_d;

    // Mask of already processed rows, usesd for both analysis and solve
    CUdeviceptr m_processed_rows_d;

    // Sorted indices of rows in each level
    CUdeviceptr m_lower_levels_d;
    CUdeviceptr m_upper_levels_d;

    // Solution GPU address
    CUdeviceptr m_x_d;
};
