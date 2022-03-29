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

__device__ int stack_id = 0;

template<typename Float>
__device__ void solve_lower(uint nrows, uint *levels, volatile bool *solved_rows, uint* rows, uint* columns, Float* values, volatile Float* x) {
    int lvl_idx = atomicAdd(&stack_id, 1);
    if (lvl_idx >= nrows)
        return;

    uint row = levels[lvl_idx];
    uint row_start = rows[row];
    uint row_end = rows[row + 1] - 1;
    Float diag_entry = values[row_end];
    Float r = x[row];
    uint col;
    for (uint i=row_start; i<row_end; i++) {
        col = columns[i];
        // Busy wait for the corresponding entry in x to be solved
        while (!solved_rows[col])
            continue;
        r -= values[i] * x[col];
    }
    x[row] = r / diag_entry;
    // Signal other threads that this entry is available
    solved_rows[row] = true;
    if (lvl_idx == nrows-1)
        stack_id = 0;
}

template<typename Float>
__device__ void solve_upper(uint nrows, uint *levels, volatile bool *solved_rows, uint* rows, uint* columns, Float* values, volatile Float* x) {
    int lvl_idx = atomicAdd(&stack_id, 1);
    if (lvl_idx >= nrows)
        return;

    uint row = levels[lvl_idx];
    uint row_start = rows[row];
    uint row_end = rows[row + 1];
    Float diag_entry = values[row_start];
    Float r = x[row];
    uint col;
    for (uint i=row_end-1; i>row_start; i--) {
        col = columns[i];
        // Busy wait for the corresponding entry in x to be solved
        while (!solved_rows[col])
            continue;
        r -= values[i] * x[col];
    }
    x[row] = r / diag_entry;
    // Signal other threads that this entry is available
    solved_rows[row] = true;
    if (lvl_idx == nrows-1)
        stack_id = 0;
}

extern "C" __global__ void solve_lower_float(uint nrows, uint*levels, bool *solved_rows, uint *rows, uint *columns, float *values, float*x) {
    solve_lower<float>(nrows, levels, solved_rows, rows, columns, values, x);
}

extern "C" __global__ void solve_lower_double(uint nrows, uint*levels, bool *solved_rows, uint *rows, uint *columns, double *values, double*x) {
    solve_lower<double>(nrows, levels, solved_rows, rows, columns, values, x);
}

extern "C" __global__ void solve_upper_float(uint nrows, uint*levels, bool *solved_rows, uint *rows, uint *columns, float *values, float*x) {
    solve_upper<float>(nrows, levels, solved_rows, rows, columns, values, x);
}

extern "C" __global__ void solve_upper_double(uint nrows, uint*levels, bool *solved_rows, uint *rows, uint *columns, double *values, double*x) {
    solve_upper<double>(nrows, levels, solved_rows, rows, columns, values, x);
}
