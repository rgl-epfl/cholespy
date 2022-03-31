#include "triangle_solve.h"
#include "cuda_setup.h"

// Switch 2 pointers
void swap_pointers(void *a, void *b) {
    void *tmp = a;
    a = b;
    b = tmp;
}

template<typename Float>
SparseTriangularSolver<Float>::SparseTriangularSolver(uint n_rows, uint n_entries, uint *rows, uint *cols, Float *data, bool lower)
 : m_lower(lower), m_n_rows(n_rows){

    // Initialize CUDA and load the kernels if not already done
    if (!init)
        initCuda<Float>();
    // CSR Matrix arrays
    cuda_check(cuMemAlloc(&m_rows_d, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(m_rows_d, rows, (1+n_rows)*sizeof(uint), 0));
    cuda_check(cuMemAlloc(&m_cols_d, n_entries*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(m_cols_d, cols, n_entries*sizeof(uint), 0));
    cuda_check(cuMemAlloc(&m_data_d, n_entries*sizeof(Float)));
    cuda_check(cuMemcpyHtoDAsync(m_data_d, data, n_entries*sizeof(Float), 0));

    // Row i belongs in level level_ind[i]
    CUdeviceptr level_ind_d;
    cuda_check(cuMemAlloc(&level_ind_d, n_rows*sizeof(uint)));
    cuda_check(cuMemsetD32(level_ind_d, 0, n_rows));

    cuda_check(cuMemAlloc(&m_processed_rows, n_rows*sizeof(bool)));
    cuda_check(cuMemsetD8(m_processed_rows, 0, n_rows)); // Initialize to all false

    CUdeviceptr max_lvl_d;
    cuda_check(cuMemAlloc(&max_lvl_d, sizeof(uint)));
    cuda_check(cuMemsetD32(max_lvl_d, 0, 1));

    void *args[6] = {
        &n_rows,
        &max_lvl_d,
        &m_processed_rows,
        &level_ind_d,
        &m_rows_d,
        &m_cols_d
    };

    CUfunction analysis_kernel = (lower ? analysis_lower : analysis_upper);
    cuda_check(cuLaunchKernel(analysis_kernel,
                            n_rows, 1, 1,
                            1, 1, 1,
                            0, 0, args, 0));

    uint *level_ind_h = (uint *)malloc(n_rows*sizeof(uint));
    cuda_check(cuMemcpyDtoHAsync(level_ind_h, level_ind_d, n_rows*sizeof(uint), 0));

    uint max_lvl_h = 0;
    cuda_check(cuMemcpyDtoHAsync(&max_lvl_h, max_lvl_d, sizeof(uint), 0));

    // Construct the (sorted) level array

    uint *levels_h = (uint*)malloc(n_rows*sizeof(uint));

    std::vector<uint> level_ptr;
    level_ptr.resize(max_lvl_h + 1, 0);

    // TODO: Try to do some of this on the GPU, or maybe with drjit types?
    // Count the number of rows per level
    for (uint i=0; i<n_rows; i++) {
        level_ptr[1+level_ind_h[i]]++;
    }

    // Convert into the list of pointers to the start of each level
    for (uint i=0, S=0; i<max_lvl_h; i++){
        S += level_ptr[i+1];
        level_ptr[i+1] = S;
    }

    // Move all rows to their place in the level array
    for (uint i=0; i<n_rows; i++) {
        uint row_level = level_ind_h[i]; // Row i belongs to level row_level
        levels_h[level_ptr[row_level]] = i;
        level_ptr[row_level]++;
    }

    cuda_check(cuMemAlloc(&m_levels_d, n_rows*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(m_levels_d, levels_h, n_rows*sizeof(uint), 0));

    // Allocate space for the solution
    cuda_check(cuMemAlloc(&m_x_d, n_rows*sizeof(Float)));

    // Free useless stuff
    free(levels_h);
    free(level_ind_h);
    cuda_check(cuMemFree(level_ind_d));
}

template<typename Float>
SparseTriangularSolver<Float>::~SparseTriangularSolver() {
    cuda_check(cuMemFree(m_levels_d));
    cuda_check(cuMemFree(m_processed_rows));
    cuda_check(cuMemFree(m_x_d));
    cuda_check(cuMemFree(m_rows_d));
    cuda_check(cuMemFree(m_cols_d));
    cuda_check(cuMemFree(m_data_d));
}

template<typename Float>
std::vector<Float> SparseTriangularSolver<Float>::solve(Float *b) {

    cuda_check(cuMemcpyHtoDAsync(m_x_d, b, m_n_rows*sizeof(Float), 0));
    cuda_check(cuMemsetD8(m_processed_rows, 0, m_n_rows)); // Initialize to all false

    void *args[7] = {
        &m_n_rows,
        &m_levels_d,
        &m_processed_rows,
        &m_rows_d,
        &m_cols_d,
        &m_data_d,
        &m_x_d,
    };
    CUfunction solve_kernel = (m_lower ? solve_lower : solve_upper);
    cuda_check(cuLaunchKernel(solve_kernel,
                            m_n_rows, 1, 1,
                            1, 1, 1,
                            0, 0, args, 0));

    Float *x_h = (Float*) malloc(m_n_rows*sizeof(Float));
    cuda_check(cuMemcpyDtoHAsync(x_h, m_x_d, m_n_rows*sizeof(Float), 0));
    return std::vector<Float>(x_h, x_h+m_n_rows);
}

template class SparseTriangularSolver<float>;
template class SparseTriangularSolver<double>;
