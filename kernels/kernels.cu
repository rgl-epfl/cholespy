#include <stdio.h>

// Analysis kernels

__device__ int row_idx = 0;

extern "C" __global__ void analysis_lower(uint n_rows, uint *max_lvl, volatile bool *analyzed_rows, volatile uint *row_levels, uint *rows, uint *cols) {
    int row = atomicAdd(&row_idx, 1);
    if (row >= n_rows)
        return;

    uint row_start = rows[row];
    uint row_end = rows[row + 1] - 1;

    uint col;
    uint row_lvl = 0; // We determine to which level this row is going to be added
    for (uint i=row_start; i<row_end; i++) {
        col = cols[i];
        while (!analyzed_rows[col])
            continue;
        uint col_lvl = row_levels[col];
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

extern "C" __global__ void analysis_upper(uint n_rows, uint *max_lvl, volatile bool *analyzed_rows, volatile uint *row_levels, uint *rows, uint *cols) {
    int row = n_rows - 1 - atomicAdd(&row_idx, 1);
    if (row < 0)
        return;

    uint row_start = rows[row];
    uint row_end = rows[row + 1];

    uint col;
    uint row_lvl = 0;
    for (uint i=row_end-1; i>row_start; i--) {
        col = cols[i];
        while (!analyzed_rows[col])
            continue;
        uint col_lvl = row_levels[col];
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
