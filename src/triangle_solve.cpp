#include "triangle_solve.h"

// TODO: This is a bit dirty, there's probably a better way
extern CUfunction solve_chain;
extern CUfunction solve_row_multiblock;
extern CUfunction solve_lower;
extern CUfunction solve_upper;
extern CUfunction find_roots;
extern CUfunction analyze;
extern CUfunction find_roots_in_candidates;

// Switch 2 pointers
void swap_pointers(void *a, void *b) {
    void *tmp = a;
    a = b;
    b = tmp;
}

template<typename Float>
SparseTriangularSolver<Float>::SparseTriangularSolver(uint n_rows, uint n_entries, CUdeviceptr csc_cols_d, CUdeviceptr csc_rows_d, CUdeviceptr csr_rows_d, CUdeviceptr csr_cols_d, CUdeviceptr csr_data_d, bool lower)
 : m_lower(lower), m_n_rows(n_rows), m_rows_d(csr_rows_d), m_cols_d(csr_cols_d), m_data_d(csr_data_d) {

    // Candidate array
    CUdeviceptr c_root;
    cuda_check(cuMemAlloc(&c_root, n_rows*sizeof(bool)));
    cuda_check(cuMemsetD8(c_root, 0, n_rows)); // Initialize to all false
    // level write array (the current level)
    CUdeviceptr w_root;
    cuda_check(cuMemAlloc(&w_root, n_rows*sizeof(bool)));
    cuda_check(cuMemsetD8(w_root, 0, n_rows)); // Initialize to all false
    // level read array (the previous level)
    CUdeviceptr r_root;
    cuda_check(cuMemAlloc(&r_root, n_rows*sizeof(bool)));
    cuda_check(cuMemsetD8(r_root, 0, n_rows)); // Initialize to all false

    // Counter of rows already in levels
    CUdeviceptr level_count_d;
    uint level_count_h = 0;
    cuda_check(cuMemAlloc(&level_count_d, sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(level_count_d, &level_count_h, sizeof(uint), 0));

    // Row i belongs in level level_ind[i]
    uint *level_ind_h = (uint *)malloc(n_rows*sizeof(uint));
    CUdeviceptr level_ind_d;
    cuda_check(cuMemAlloc(&level_ind_d, n_rows*sizeof(uint)));

    m_level_ptr_h.reserve(n_rows);
    m_level_ptr_h.push_back(0);
    m_chain_ptr.reserve(n_rows + 1); // TODO: could use a more educated guess depending on block size and n_rows
    m_chain_ptr.push_back(0);
    uint level_size = 0;
    uint level_count_prev = 0;

    //find_roots
    uint num_blocks = (n_rows + BLOCK_SIZE -1) / BLOCK_SIZE;
    void *fr_args[6] = {&n_rows, &level_count_d, &level_ind_d, &m_rows_d, &m_cols_d, &r_root};
    cuda_check(cuLaunchKernel(find_roots,
                            num_blocks, 1, 1,
                            BLOCK_SIZE, 1, 1,
                            0, 0, fr_args, 0));

    cuda_check(cuMemcpyDtoHAsync(&level_count_h, level_count_d, sizeof(uint), 0));
    m_level_ptr_h.push_back(level_count_h);

    if (level_count_h > BLOCK_SIZE) {
        // First level is too big to be part of a chain
        m_chain_ptr.push_back(1);
    }

    uint level = 1;

    // Arguments for the kernels
    void *analyze_args[7] = {&n_rows, &level, &r_root, &c_root, &level_ind_d, &csc_cols_d, &csc_rows_d};
    void *frc_args[8] = {&n_rows, &level, &level_count_d, &w_root, &c_root, &level_ind_d, &m_rows_d, &m_cols_d};

    while (level_count_h < n_rows) {
        // analyze
        cuda_check(cuLaunchKernel(analyze,
                                num_blocks, 1, 1,
                                BLOCK_SIZE, 1, 1,
                                0, 0, analyze_args, 0));
        // find_roots_in_candidates
        cuda_check(cuLaunchKernel(find_roots_in_candidates,
                                num_blocks, 1, 1,
                                BLOCK_SIZE, 1, 1,
                                0, 0, frc_args, 0));

        //update level count
        level_count_prev = level_count_h;
        cuda_check(cuMemcpyDtoHAsync(&level_count_h, level_count_d, sizeof(uint), 0));
        level_size = level_count_h - level_count_prev;
        m_level_ptr_h.push_back(level_count_h);

        // Chains
        if (level_size > BLOCK_SIZE) {
            if (m_chain_ptr.back() == level)
                m_chain_ptr.push_back(level+1); // previous level was also too large
            else {
                m_chain_ptr.push_back(level);
                m_chain_ptr.push_back(level+1);
            }
        }
        level++;
        // Swap read and write buffers
        swap_pointers((void *)r_root, (void *)w_root);
    }
    // Add final element if not already there (i.e. if the last level is part of
    // a chain)
    if (m_chain_ptr.back() != level)
        m_chain_ptr.push_back(level);

    m_chain_ptr.shrink_to_fit();
    m_level_ptr_h.shrink_to_fit();

    // Build the actual solve data structure
    cuda_check(cuMemcpyDtoHAsync(level_ind_h, level_ind_d, n_rows*sizeof(uint), 0));
    // Construct the (sorted) level array
    uint *levels_h = (uint*)malloc(n_rows*sizeof(uint));
    for (uint i=0; i<n_rows; i++) {
        uint row_level = level_ind_h[i]; // Row i belongs to level row_level
        levels_h[m_level_ptr_h[row_level]] = i;
        m_level_ptr_h[row_level]++;
    }

    // Undo the modifications to m_level_ptr from the previous step
    for(uint i = 0, last = 0; i <= level; i++){
        uint temp = m_level_ptr_h[i];
        m_level_ptr_h[i] = last;
        last = temp;
    }

    cuda_check(cuMemAlloc(&m_level_ptr_d, m_level_ptr_h.size()*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(m_level_ptr_d, &m_level_ptr_h[0], m_level_ptr_h.size()*sizeof(uint), 0));
    cuda_check(cuMemAlloc(&m_levels_d, n_rows*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(m_levels_d, levels_h, n_rows*sizeof(uint), 0));
    // solution
    cuda_check(cuMemAlloc(&m_x_d, n_rows*sizeof(Float)));

    free(levels_h);
    free(level_ind_h);
    cuda_check(cuMemFree(level_ind_d));
    cuda_check(cuMemFree(level_count_d));
}

template<typename Float>
SparseTriangularSolver<Float>::~SparseTriangularSolver() {
    cuda_check(cuMemFree(m_level_ptr_d));
    cuda_check(cuMemFree(m_levels_d));
    cuda_check(cuMemFree(m_x_d));
}

template<typename Float>
std::vector<Float> SparseTriangularSolver<Float>::solve(Float *b) {

    cuda_check(cuMemcpyHtoDAsync(m_x_d, b, m_n_rows*sizeof(Float), 0));

    CUdeviceptr solved_rows;
    cuda_check(cuMemAlloc(&solved_rows, m_n_rows*sizeof(bool)));
    cuda_check(cuMemsetD8(solved_rows, 0, m_n_rows)); // Initialize to all false

    void *args[7] = {
        &m_n_rows,
        &m_levels_d,
        &solved_rows,
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
    cuda_check(cuMemFree(solved_rows));
    return std::vector<Float>(x_h, x_h+m_n_rows);
}

template class SparseTriangularSolver<float>;
template class SparseTriangularSolver<double>;
