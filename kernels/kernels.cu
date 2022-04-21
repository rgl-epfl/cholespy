#include <stdio.h>

#define CACHE_SIZE 128
// Analysis kernels

__device__ int row_idx = 0;

extern "C" __global__ void analysis_lower(int n_rows, int *max_lvl, volatile bool *analyzed_rows, volatile int *row_levels, int *rows, int *cols) {
    int row = atomicAdd(&row_idx, 1);
    if (row >= n_rows)
        return;

    int row_start = rows[row];
    int row_end = rows[row + 1] - 1;

    int col;
    int row_lvl = 0; // We determine to which level this row is going to be added
    for (int i=row_start; i<row_end; i++) {
        col = cols[i];
        while (!analyzed_rows[col])
            continue;
        int col_lvl = row_levels[col];
        if (row_lvl <= col_lvl)
            row_lvl = col_lvl + 1;
    }

    atomicMax(max_lvl, row_lvl);
    row_levels[row] = row_lvl;
    analyzed_rows[row] = true;
    // Wrap up
    if (row == n_rows - 1)
        row_idx = 0;
}

extern "C" __global__ void analysis_upper(int n_rows, int *max_lvl, volatile bool *analyzed_rows, volatile int *row_levels, int *rows, int *cols) {
    int row = n_rows - 1 - atomicAdd(&row_idx, 1);
    if (row < 0)
        return;

    int row_start = rows[row];
    int row_end = rows[row + 1] - 1;

    int col;
    int row_lvl = 0;
    for (int i=row_end; i>row_start; i--) {
        col = cols[i];
        while (!analyzed_rows[col])
            continue;
        int col_lvl = row_levels[col];
        if (row_lvl <= col_lvl)
            row_lvl = col_lvl + 1;
    }

    atomicMax(max_lvl, row_lvl);
    row_levels[row] = row_lvl;
    analyzed_rows[row] = true;
    // Wrap up
    if (row == 0)
        row_idx = 0;
}

// Solve kernels

template<typename Float>
__device__ void solve_lower(int nrhs, int nrows, int *stack_id, int *levels, volatile bool *solved_rows, int *rows, int *columns, Float *values, volatile Float *x, Float *b, int *perm) {

    __shared__ int lvl_idx;
    __shared__ int cols_cache[CACHE_SIZE];
    __shared__ Float vals_cache[CACHE_SIZE];

    int thread_idx = threadIdx.x;
    // The current block solves the row at index *stack_id in levels
    if (thread_idx == 0) {
        lvl_idx = atomicAdd(stack_id, 1);
    }
    __syncthreads();

    if (lvl_idx >= nrows)
        return;

    int row = levels[lvl_idx];
    int row_start = rows[row];
    int row_end = rows[row + 1] - 1;
    Float diag_entry = values[row_end];
    Float r;
    if (thread_idx < nrhs)
        r = b[perm[row] * nrhs + thread_idx];
    int col;
    Float val;
    for (int i=row_start; i<row_end; ++i) {
        int cache_idx = (i-row_start) % CACHE_SIZE;
        if (cache_idx == 0) {
            // Update the cache
            if (i + thread_idx < row_end) {
                cols_cache[thread_idx] = columns[i + thread_idx];
                vals_cache[thread_idx] = values[i + thread_idx];
            }
            __syncthreads();
        }

        if (thread_idx < nrhs) {
            // Read current column and corresponding entry in the cache
            col = cols_cache[cache_idx];
            val = vals_cache[cache_idx];
        }
        // Busy wait for the corresponding entry in x to be solved
        if (thread_idx == 0) {
            while (!solved_rows[col])
                continue;
        }
        __syncthreads();

        if (thread_idx < nrhs)
            r -= val * x[col * nrhs + thread_idx];

    }

    // Write the final value
    if (thread_idx < nrhs)
        x[row * nrhs + thread_idx] = r / diag_entry;

    // Make sure we write all entries before signaling other blocks
    __threadfence();
    __syncthreads();

    if (thread_idx != 0)
        return;

    // Signal other blocks that this entry is available
    solved_rows[row] = true;
}

template<typename Float>
__device__ void solve_upper(int nrhs, int nrows, int *stack_id, int *levels, volatile bool *solved_rows, int *rows, int *columns, Float *values, volatile Float *x, Float *b, int *perm) {

    __shared__ int lvl_idx;
    __shared__ int cols_cache[CACHE_SIZE];
    __shared__ Float vals_cache[CACHE_SIZE];

    int thread_idx = threadIdx.x;
    // The current block solves the row at index *stack_id in levels
    if (thread_idx == 0)
        lvl_idx = atomicAdd(stack_id, 1);
    __syncthreads();

    if (lvl_idx >= nrows)
        return;

    int row = levels[lvl_idx];
    int row_start = rows[row];
    int row_end = rows[row + 1] - 1;
    Float diag_entry = values[row_start];
    Float r;
    if (thread_idx < nrhs)
        r = x[row * nrhs + thread_idx];
    int col;
    Float val;
    for (int i=row_end; i>row_start; --i) {
        int cache_idx = (row_end - i) % CACHE_SIZE;
        if (cache_idx == 0) {
            // Update the cache
            if (i - thread_idx > row_start) {
                vals_cache[thread_idx] = values[i - thread_idx];
                cols_cache[thread_idx] = columns[i - thread_idx];
            }
            __syncthreads();
        }

        if (thread_idx < nrhs) {
            // Read current column and corresponding entry in the cache
            col = cols_cache[cache_idx];
            val = vals_cache[cache_idx];
        }
        // Busy wait for the corresponding entry in x to be solved
        if (thread_idx == 0) {
            while (!solved_rows[col])
                continue;
        }
        __syncthreads();

        if (thread_idx < nrhs)
            r -= val * x[col * nrhs + thread_idx];

    }

    // Write the final value
    if (thread_idx < nrhs) {
        x[row * nrhs + thread_idx] = r / diag_entry;
        b[perm[row] * nrhs + thread_idx] = r / diag_entry;
    }


    // Make sure we write all entries before signaling other blocks
    __threadfence();
    __syncthreads();

    if (thread_idx != 0)
        return;

    // Signal other blocks that this entry is available
    solved_rows[row] = true;
}

extern "C" __global__ void solve_lower_float(int nrhs, int nrows, int *stack_id,  int *levels, bool *solved_rows, int *rows, int *columns, float *values, float *x, float *b, int *perm) {
    solve_lower<float>(nrhs, nrows, stack_id, levels, solved_rows, rows, columns, values, x, b, perm);
}

extern "C" __global__ void solve_lower_double(int nrhs, int nrows, int *stack_id, int *levels, bool *solved_rows, int *rows, int *columns, double *values, double *x, double *b, int *perm) {
    solve_lower<double>(nrhs, nrows, stack_id, levels, solved_rows, rows, columns, values, x, b, perm);
}

extern "C" __global__ void solve_upper_float(int nrhs, int nrows, int *stack_id, int *levels, bool *solved_rows, int *rows, int *columns, float *values, float *x, float *b, int *perm) {
    solve_upper<float>(nrhs, nrows, stack_id, levels, solved_rows, rows, columns, values, x, b, perm);
}

extern "C" __global__ void solve_upper_double(int nrhs, int nrows, int *stack_id, int *levels, bool *solved_rows, int *rows, int *columns, double *values, double *x, double *b, int *perm) {
    solve_upper<double>(nrhs, nrows, stack_id, levels, solved_rows, rows, columns, values, x, b, perm);
}
