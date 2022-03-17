#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "cuda_helpers.h"

#define BLOCK_SIZE 1024

template<typename Float> class SparseTriangularSolver {
public:
    SparseTriangularSolver(uint n_rows, uint n_entries, CUdeviceptr csc_cols_d, CUdeviceptr csc_rows_d, CUdeviceptr csr_rows_d, CUdeviceptr csr_cols_d, CUdeviceptr csr_data_d, bool lower);
    ~SparseTriangularSolver();

    std::vector<Float> solve(Float *b);

private:
    // Lower or upper triangular matrix
    bool m_lower;
    // Starting index of each "chain" in m_level_ptr_d (see Naumov 2011 for the definition of a chain)
    std::vector<uint> m_chain_ptr;
    // CSR format arrays
    CUdeviceptr m_rows_d;
    CUdeviceptr m_cols_d;
    CUdeviceptr m_data_d;
    // Row count
    uint m_n_rows;
    // Starting index of each level in m_levels
    CUdeviceptr m_level_ptr_d;
    // Same array but on the CPU
    std::vector<uint> m_level_ptr_h;//TODO: is this really necessary?
    // Sorted indices of rows in each level
    CUdeviceptr m_levels_d;
    // RHS GPU address
    CUdeviceptr m_b_d;
    // Solution GPU address
    CUdeviceptr m_x_d;
};

using SparseTriangularSolverF = SparseTriangularSolver<float>;
using SparseTriangularSolverD = SparseTriangularSolver<double>;
