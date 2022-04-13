#include "cholesky_solver.h"
#include "cuda_setup.h"
#include <algorithm>

template <typename Float>
CholeskySolver<Float>::CholeskySolver(uint nrhs, uint n_verts, uint n_faces, uint *faces, double lambda) : m_n(n_verts), m_nrhs(nrhs) {

    // Initialize CUDA and load the kernels if not already done
    initCuda<Float>();

    // Placeholders for the CSC matrix data
    std::vector<int> col_ptr, rows;
    std::vector<double> data;

    // Build the actual matrix
    build_matrix(n_verts, n_faces, faces, lambda, col_ptr, rows, data);

    // Mask of rows already processed
    cuda_check(cuMemAlloc(&m_processed_rows_d, m_n*sizeof(bool)));
    cuda_check(cuMemsetD8(m_processed_rows_d, 0, m_n)); // Initialize to all false

    // Row id
    cuda_check(cuMemAlloc(&m_stack_id_d, sizeof(uint)));
    cuda_check(cuMemsetD32(m_stack_id_d, 0, 1));

    // Run the Cholesky factorization through CHOLMOD and run the analysis
    factorize(col_ptr, rows, data);

    // Allocate space for the solution
    cuda_check(cuMemAlloc(&m_x_d, m_n*m_nrhs*sizeof(Float)));
}

template <typename Float>
void CholeskySolver<Float>::build_matrix(uint n_verts, uint n_faces, uint *faces, double lambda, std::vector<int> &col_ptr, std::vector<int> &rows, std::vector<double> &data) {
    // We start by building the (lower half of the) (I + λ L) matrix in the COO format.

    // indices of nonzero entries
    std::vector<uint> ii;
    std::vector<uint> jj;

    // Heuristic based on average connectivity on a triangle mesh.
    ii.reserve(7 * n_verts);
    jj.reserve(7 * n_verts);
    std::vector<uint> col_entries;
    col_entries.resize(n_verts, 0);

    // Add one entry per edge
    for (uint i=0; i<n_faces; i++) {
        for (uint j=0; j<3; j++) {
            for (uint k=j+1; k<3; k++) {
                uint s = faces[3*i + j];
                uint d = faces[3*i + k];
                if (s > d) {
                    // L[s,d]
                    ii.push_back(s);
                    jj.push_back(d);
                    col_entries[d]++;
                } else {
                    // L[d,s]
                    ii.push_back(d);
                    jj.push_back(s);
                    col_entries[s]++;
                }
            }
        }
    }

    // Add diagonal indices
    for(uint i=0; i<n_verts; i++) {
        ii.push_back(i);
        jj.push_back(i);
        col_entries[i]++;
    }

    ii.shrink_to_fit();
    jj.shrink_to_fit();

    uint nnz = ii.size();

    // Then we convert the COO representation to CSC

    std::vector<uint> tmp_col_ptr;
    tmp_col_ptr.resize(n_faces+1, 0);
    std::vector<uint> tmp_rows;
    tmp_rows.resize(nnz, 0);

    /*
    Build the column pointer array, where tmp_col_ptr[i] is the start of the
    i-th column in the other arrays
    */
    uint cumsum=0;
    for (uint i=0; i<n_verts; i++) {
        tmp_col_ptr[i] = cumsum;
        cumsum += col_entries[i];
    }
    tmp_col_ptr[n_verts] = cumsum;

    /*
    Now move the row indices of each entry so that entries in column j are in
    positions tmp_col_ptr[j] to tmp_col_ptr[j+1]-1
    */
    for (uint i=0; i<nnz; i++) {
        uint col = jj[i];
        uint dst = tmp_col_ptr[col];

        tmp_rows[dst] = ii[i];
        tmp_col_ptr[col]++;
    }

    // Undo the modifications to tmp_col_ptr from the previous step
    for(uint i = 0, last = 0; i <= n_verts; i++){
        uint temp = tmp_col_ptr[i];
        tmp_col_ptr[i] = last;
        last = temp;
    }

    // Sort indices in each column to ease the removal of duplicates
    for (uint i=0; i<n_verts; i++) {
        std::sort(tmp_rows.begin() + tmp_col_ptr[i], tmp_rows.begin() + tmp_col_ptr[i+1]);
    }

    rows.reserve(nnz);
    data.reserve(nnz);
    col_ptr.resize(n_verts+1, 0);
    cumsum = 0;

    std::vector<uint> adjacency;
    adjacency.resize(n_verts, 0);
    // Remove duplicates
    for (uint col=0; col<n_verts; col++) {
        uint i = tmp_col_ptr[col];
        uint n_elements = 0;
        uint row = tmp_rows[i];
        while (i<tmp_col_ptr[col+1]) {
            if (row != col) {
                // Count unique off diag entries per row
                // We increment both indices because we only store half of the entries
                adjacency[row]++;
                adjacency[col]++;
            }
            rows.push_back(row);
            data.push_back(-lambda);
            n_elements++;
            uint previous_row = row;
            while (row == previous_row && i<tmp_col_ptr[col+1]) {
                // Ignore duplicate entries (all off-diagonal entries of the laplacian are ones)
                i++;
                row = tmp_rows[i];
            }
        }
        // Correct element count
        col_entries[col] = n_elements;
        // Correct column start pointer
        col_ptr[col] = cumsum;
        cumsum += n_elements;
    }
    col_ptr[n_verts] = cumsum;
    data.shrink_to_fit();
    rows.shrink_to_fit();

    // Set diagonal indices proper values
    for (uint j=0; j<n_verts; j++) {
        for (uint i=col_ptr[j]; i<col_ptr[j+1]; i++) {
            if (j == rows[i]) // diagonal element
                data[i] = adjacency[j] * lambda + 1.0;
        }
    }

}

template <typename Float>
void CholeskySolver<Float>::factorize(std::vector<int> &col_ptr, std::vector<int> &rows, std::vector<double> &data) {
    cholmod_sparse *A;
    cholmod_factor *F;
    cholmod_common c;

    cholmod_start(&c);

    c.supernodal = CHOLMOD_SIMPLICIAL; // TODO: if using Cholmod to solve, try with supernodal here
    c.final_ll = 1; // compute LL' factorization instead of LDL´ (default for simplicial)
    c.print = 5; // log level TODO: Remove
    //TODO: Not sure this is necessary, need to benchmark
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_NESDIS;

    A = cholmod_allocate_sparse(
        m_n,
        m_n,
        data.size(),
        1,
        1,
        -1,
        CHOLMOD_REAL,
        &c
    );

    int *A_colptr = (int*) A->p;
    int *A_rows = (int*) A->i;
    // CHOLMOD currently only supports the double precision version of the decomposition
    double *A_data = (double*) A->x;
    for (uint j=0; j<m_n; j++) {
        A_colptr[j] = (int) col_ptr[j];
        for (uint i=col_ptr[j]; i<col_ptr[j+1]; i++) {
            A_rows[i] = (int) rows[i];
            A_data[i] = data[i];
        }
    }
    A_colptr[m_n] = rows.size();

    F = cholmod_analyze(A, &c);
    cholmod_factorize(A, F, &c);

    // Copy permutation
    m_perm = (uint*) malloc(m_n*sizeof(uint));
    int *perm = (int*) F->Perm;
    for (uint i=0; i<m_n; i++) {
        m_perm[i] = (uint) perm[i];
    }

    cholmod_sparse *lower_csc = cholmod_factor_to_sparse(F, &c);
    // The transpose of a CSC (resp. CSR) matrix is its CSR (resp. CSC) representation
    cholmod_sparse *lower_csr = cholmod_transpose(lower_csc, 1, &c);

    // Since we can only factorize in double precision mode, we have to recast the data array to Float
    Float *csc_data;
    Float *csr_data;
    if (std::is_same_v<Float, double>) {
        csc_data = (Float *) lower_csc->x;
        csr_data = (Float *) lower_csr->x;
    } else {
        csc_data = (Float *)malloc(lower_csc->nzmax * sizeof(Float));
        csr_data = (Float *)malloc(lower_csr->nzmax * sizeof(Float));

        double *csc_data_ptr = (double *) lower_csc->x;
        double *csr_data_ptr = (double *) lower_csr->x;

        for (uint32_t i=0; i < lower_csc->nzmax; i++) {
            csc_data[i] = (Float) csc_data_ptr[i];
            csr_data[i] = (Float) csr_data_ptr[i];
        }
    }


    uint n_rows = lower_csc->nrow;
    uint n_entries = lower_csc->nzmax;

    // The CSC representation of a matrix is the same as the CSR of its transpose
    analyze(n_rows, n_entries, lower_csr->p, lower_csr->i, csr_data, true);

    // To prepare the transpose we merely need to swap the roles of the CSR and CSC representations (CSC rows -> CSR cols, CSC cols -> CSR rows)
    analyze(n_rows, n_entries, lower_csc->p, lower_csc->i, csc_data, false);

    // Free CHOLMOD stuff
    cholmod_free_sparse(&A, &c);
    cholmod_free_sparse(&lower_csc, &c);
    cholmod_free_sparse(&lower_csr, &c);
    cholmod_free_factor(&F, &c);
    cholmod_finish(&c);
    if (!std::is_same_v<Float, double>) {
        free(csc_data);
        free(csr_data);
    }
}

template <typename Float>
void CholeskySolver<Float>::analyze(uint n_rows, uint n_entries, void *csr_rows, void *csr_cols, Float* csr_data, bool lower) {

    CUdeviceptr *rows_d = (lower ? &m_lower_rows_d : &m_upper_rows_d);
    CUdeviceptr *cols_d = (lower ? &m_lower_cols_d : &m_upper_cols_d);
    CUdeviceptr *data_d = (lower ? &m_lower_data_d : &m_upper_data_d);
    CUdeviceptr *levels_d = (lower ? &m_lower_levels_d : &m_upper_levels_d);

    // CSR Matrix arrays
    cuda_check(cuMemAlloc(rows_d, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(*rows_d, csr_rows, (1+n_rows)*sizeof(uint), 0));
    cuda_check(cuMemAlloc(cols_d, n_entries*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(*cols_d, csr_cols, n_entries*sizeof(uint), 0));
    cuda_check(cuMemAlloc(data_d, n_entries*sizeof(Float)));
    cuda_check(cuMemcpyHtoDAsync(*data_d, csr_data, n_entries*sizeof(Float), 0));

    // Row i belongs in level level_ind[i]
    CUdeviceptr level_ind_d;
    cuda_check(cuMemAlloc(&level_ind_d, n_rows*sizeof(uint)));
    cuda_check(cuMemsetD32(level_ind_d, 0, n_rows));

    cuda_check(cuMemsetD8(m_processed_rows_d, 0, n_rows)); // Initialize to all false

    CUdeviceptr max_lvl_d;
    cuda_check(cuMemAlloc(&max_lvl_d, sizeof(uint)));
    cuda_check(cuMemsetD32(max_lvl_d, 0, 1));

    void *args[6] = {
        &n_rows,
        &max_lvl_d,
        &m_processed_rows_d,
        &level_ind_d,
        rows_d,
        cols_d
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

    cuda_check(cuMemAlloc(levels_d, n_rows*sizeof(uint)));
    cuda_check(cuMemcpyHtoDAsync(*levels_d, levels_h, n_rows*sizeof(uint), 0));

    // Free useless stuff
    free(levels_h);
    free(level_ind_h);
    cuda_check(cuMemFree(level_ind_d));
}

template<typename Float>
void CholeskySolver<Float>::solve(bool lower) {
    // Initialize buffers
    cuda_check(cuMemsetD8(m_processed_rows_d, 0, m_n)); // Initialize to all false
    cuda_check(cuMemsetD32(m_stack_id_d, 0, 1));

    CUdeviceptr rows_d = (lower ? m_lower_rows_d : m_upper_rows_d);
    CUdeviceptr cols_d = (lower ? m_lower_cols_d : m_upper_cols_d);
    CUdeviceptr data_d = (lower ? m_lower_data_d : m_upper_data_d);
    CUdeviceptr levels_d = (lower ? m_lower_levels_d : m_upper_levels_d);

    void *args[9] = {
        &m_nrhs,
        &m_n,
        &m_stack_id_d,
        &levels_d,
        &m_processed_rows_d,
        &rows_d,
        &cols_d,
        &data_d,
        &m_x_d,
    };
    CUfunction solve_kernel = (lower ? solve_lower : solve_upper);
    cuda_check(cuLaunchKernel(solve_kernel,
                            m_n, 1, 1,
                            128, 1, 1,
                            0, 0, args, 0));
}

template <typename Float>
std::vector<Float> CholeskySolver<Float>::solve(Float *b) {
    // TODO fallback to cholmod in the CPU array case
    // TODO: Do this on the GPU?
    Float *tmp = (Float *)malloc(m_n * m_nrhs * sizeof(Float));
    for (uint i=0; i<m_n; ++i)
        for (uint j=0; j<m_nrhs; ++j) {
            tmp[m_n * j + i] = b[m_n * j + m_perm[i]];
        }

    cuda_check(cuMemcpyHtoDAsync(m_x_d, tmp, m_n*m_nrhs*sizeof(Float), 0));

    solve(true);
    solve(false);

    cuda_check(cuMemcpyDtoHAsync(tmp, m_x_d, m_n*m_nrhs*sizeof(Float), 0));

	std::vector<Float> sol(m_n*m_nrhs);
    // Invert permutation
    for (uint i=0; i<m_n; ++i)
        for (uint j=0; j<m_nrhs; ++j)
            sol[m_n * j + m_perm[i]] = tmp[m_n * j + i];

    free(tmp);
    return sol;
}

template <typename Float>
CholeskySolver<Float>::~CholeskySolver() {
    free(m_perm);

    cuda_check(cuMemFree(m_processed_rows_d));

    cuda_check(cuMemFree(m_stack_id_d));

    cuda_check(cuMemFree(m_x_d));

    cuda_check(cuMemFree(m_lower_rows_d));
    cuda_check(cuMemFree(m_lower_cols_d));
    cuda_check(cuMemFree(m_lower_data_d));

    cuda_check(cuMemFree(m_upper_rows_d));
    cuda_check(cuMemFree(m_upper_cols_d));
    cuda_check(cuMemFree(m_upper_data_d));

    cuda_check(cuMemFree(m_lower_levels_d));
    cuda_check(cuMemFree(m_upper_levels_d));
}

template class CholeskySolver<float>;
template class CholeskySolver<double>;
