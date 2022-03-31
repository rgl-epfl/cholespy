#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include "cuda_helpers.h"

#define BLOCK_SIZE 1024

template<typename Float> class SparseTriangularSolver {
public:
    SparseTriangularSolver(uint n_rows, uint n_entries, uint *rows, uint *cols, Float *data, bool lower);
    ~SparseTriangularSolver();

    std::vector<Float> solve(Float *b);

private:
    // Lower or upper triangular matrix
    bool m_lower;
    // CSR format arrays
    CUdeviceptr m_rows_d;
    CUdeviceptr m_cols_d;
    CUdeviceptr m_data_d;
    // Mask of already processed rows, usesd for both analysis and solve
    CUdeviceptr m_processed_rows;
    // Row count
    uint m_n_rows;
    // Sorted indices of rows in each level
    CUdeviceptr m_levels_d;
    // Solution GPU address
    CUdeviceptr m_x_d;
};

using SparseTriangularSolverF = SparseTriangularSolver<float>;
using SparseTriangularSolverD = SparseTriangularSolver<double>;
