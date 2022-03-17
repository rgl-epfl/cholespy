#include <stdio.h>

// Analysis kernels

// Find rows that belong to the first level, i.e. rows with only the diagonal element.
extern "C" __global__ void find_roots(uint n_rows, uint *level_count, uint *level_ind, uint* csr_rows, uint* csr_columns, bool *r_root) {
    uint row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx >= n_rows)
        return;
    // First level is rows with only one element on the diagonal
    if (csr_rows[row_idx+1] - csr_rows[row_idx] == 1) {
        r_root[row_idx] = true;
        level_ind[row_idx] = 0;
        atomicAdd(level_count, 1);
    } else {
        // Initialize buffers to their default value.
        r_root[row_idx] = false;
        level_ind[row_idx] = n_rows;
    }

    // TODO: check that the diag. entry is not 0 and that the nonzero entry is indeed the diag one
    // TODO: maybe don't assume there are no explicit zeros in the matrix?
}

extern "C" __global__ void analyze(uint n_rows, uint level, bool *r_root, bool *c_root, uint *level_ind, uint *csc_cols, uint* csc_rows) {
    uint row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx >= n_rows || level_ind[row_idx] != level-1)
        return;

    // The candidates for the next level are the rows j such that M[i,j] is not
    // zero for each i in the previous level, i.e. the row indices of the nnz
    // entries of column i.
    uint col_start = csc_cols[row_idx], col_end = csc_cols[row_idx+1];
    for (uint i=col_start; i<col_end; i++) {
        if (csc_rows[i] != row_idx) {
            c_root[csc_rows[i]] = true;
        }
    }
}

extern "C" __global__ void find_roots_in_candidates(uint n_rows, uint level, uint *level_count, bool *w_root, bool *c_root, uint *level_ind, uint* csr_rows, uint* csr_cols) {
    uint row_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (row_idx >= n_rows)
        return;

    if (!c_root[row_idx]) {
        w_root[row_idx] = false;
        return;
    }

    uint row_start = csr_rows[row_idx], row_end = csr_rows[row_idx+1];

    w_root[row_idx] = true;
    for (uint i=row_start; i<row_end; i++) {
        // level_ind is initialized to n_rows
        if (csr_cols[i] != row_idx && level_ind[csr_cols[i]] >= level) {
            // Not all dependencies have been resolved
            w_root[row_idx] = false;
            break;
        }
    }

    if (w_root[row_idx]) {
        level_ind[row_idx] = level;
        atomicAdd(level_count, 1);
    }

    // Reset the candidate entries for next level
    c_root[row_idx] = false;
}

// Solve kernels

template<typename Float>
__device__ void solve_row_multiblock(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {
    uint row_idx = level_ptr[level] + blockDim.x * blockIdx.x + threadIdx.x;
    if (row_idx >= level_ptr[level+1])
        return;
    uint row = levels[row_idx];
    uint row_start = rows[row];
    uint row_end = rows[row + 1];
    uint diag_ptr;
    if (lower) {
        diag_ptr = row_end - 1;
        row_end--;
    } else {
        diag_ptr = row_start;
        row_start++;
    }

    Float r = 0.0f;
    for (uint i=row_start; i<row_end; i++) {
        r += values[i]*x[columns[i]];
    }

    x[row] -= r;
    x[row] /= values[diag_ptr];
}

template<typename Float>
__device__ void solve_chain(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {
    for (uint level=chain_start; level<chain_end; level++) {
        uint level_start = level_ptr[level];
        uint level_end = level_ptr[level+1];
        uint row_idx = level_start + threadIdx.x;
        if (row_idx < level_end) {
            uint row = levels[row_idx];
            uint row_start = rows[row];
            uint row_end = rows[row + 1];
            uint diag_ptr;
            if (lower) {
                diag_ptr = row_end - 1;
                row_end--;
            } else {
                diag_ptr = row_start;
                row_start++;
            }

            Float r = 0.0f;
            for (uint i=row_start; i<row_end; i++) {
                r += values[i]*x[columns[i]];
            }

            x[row] -= r;
            x[row] /= values[diag_ptr];
        }
        __syncthreads(); // Synchronize before moving to next level
    }
}


extern "C" __global__ void solve_row_multiblock_float(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, float* values, float* b, float* x, bool lower) {
    solve_row_multiblock<float>(level, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_row_multiblock_double(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, double* values, double* b, double* x, bool lower) {
    solve_row_multiblock<double>(level, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_chain_float(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, float* values, float* b, float* x, bool lower) {
    solve_chain<float>(chain_start, chain_end, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_chain_double(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, double* values, double* b, double* x, bool lower) {
    solve_chain<double>(chain_start, chain_end, level_ptr, levels, rows, columns, values, b, x, lower);
}
