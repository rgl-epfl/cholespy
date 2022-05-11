#include "cholesky_solver.h"
#include "cuda_driver.h"
#include <algorithm>
#include <exception>

// Sparse matrix utility functions, inspired from Scipy https://github.com/scipy/scipy

void coo_to_csc(int n_rows, int nnz, const int *coo_i, const int *coo_j, const double *coo_x, int *csc_col_ptr, int *csc_rows, double *csc_data) {

    // Count non zero entries per column
    for (int i=0; i<nnz; ++i) {
        csc_col_ptr[coo_j[i]]++;
    }

    /*
    Build the column pointer array, where tmp_col_ptr[i] is the start of the
    i-th column in the other arrays
    */
    for (int i=0, S=0; i<n_rows; i++) {
        int tmp = csc_col_ptr[i];
        csc_col_ptr[i] = S;
        S += tmp;
    }
    csc_col_ptr[n_rows] = nnz;

    /*
    Now move the row indices of each entry so that entries in column j are in
    positions tmp_col_ptr[j] to tmp_col_ptr[j+1]-1
    */
    for (int i=0; i<nnz; i++) {
        int col = coo_j[i];
        int dst = csc_col_ptr[col];

        csc_rows[dst] = coo_i[i];
        csc_data[dst] = coo_x[i];

        csc_col_ptr[col]++;
    }

    // Undo the modifications to tmp_col_ptr from the previous step
    for(int i = 0, last = 0; i <= n_rows; i++){
        int temp = csc_col_ptr[i];
        csc_col_ptr[i] = last;
        last = temp;
    }

    // We now have a CSC representation of our matrix, potentially with duplicates.
}

void csr_to_csc(int n_rows, int nnz, const int *csr_row_ptr, const int *csr_cols, const double *csr_data, int *csc_col_ptr, int *csc_rows, double *csc_data) {

    // Count non zero entries per column
    for (int i=0; i<nnz; ++i) {
        csc_col_ptr[csr_cols[i]]++;
    }

    /*
    Build the column pointer array, where tmp_col_ptr[i] is the start of the
    i-th column in the other arrays
    */
    for (int i=0, S=0; i<n_rows; ++i) {
        int tmp = csc_col_ptr[i];
        csc_col_ptr[i] = S;
        S += tmp;
    }
    csc_col_ptr[n_rows] = nnz;

    /*
    Now move the row indices of each entry so that entries in column j are in
    positions tmp_col_ptr[j] to tmp_col_ptr[j+1]-1
    */
    for (int i=0; i<n_rows; ++i) {
        for (int j=csr_row_ptr[i]; j<csr_row_ptr[i+1]; ++j){
            int col = csr_cols[j];
            int dst = csc_col_ptr[col];

            csc_rows[dst] = i;
            csc_data[dst] = csr_data[j];

            csc_col_ptr[col]++;
        }
    }

    // Undo the modifications to tmp_col_ptr from the previous step
    for(int i = 0, last = 0; i <= n_rows; ++i){
        int temp = csc_col_ptr[i];
        csc_col_ptr[i] = last;
        last = temp;
    }

    // We now have a CSC representation of our matrix, potentially with duplicates.
}

// Re-order (in place) the data of a CSC matrix so that row indices are sorted
void csc_sort_indices(int n_rows, int nnz, int *col_ptr, int *rows, double *data) {
    std::vector<std::pair<int, double>> tmp;

    for (int i=0; i<n_rows; ++i) {
        int col_start = col_ptr[i], col_end = col_ptr[i+1];
        tmp.resize(col_end - col_start);
        // Fill the temporary pair array with the entriex from the current column
        for (int j=col_start; j<col_end; ++j) {
            tmp[j - col_start].first = rows[j];
            tmp[j - col_start].second = data[j];
        }

        // Sort the row
        std::sort(tmp.begin(), tmp.end(), [](std::pair<int, double> &a, std::pair<int, double> &b){return a.first < b.first;});

        // Re order
        for (int j=col_start; j<col_end; ++j) {
            rows[j] = tmp[j - col_start].first;
            data[j] = tmp[j - col_start].second;
        }
    }
}

// Sum all duplicate entries in a CSC matrix (with sorted indices) and modify it in place
void csc_sum_duplicates(int n_rows, int &m_nnz, int **col_ptr, int **rows, double **data) {
    int nnz = 0;
    int col_end = 0;

    for (int i=0; i<n_rows; ++i) {
        int j = col_end;
        col_end = (*col_ptr)[i+1];
        while (j < col_end) {
            int row = (*rows)[j];
            double x = (*data)[j];
            j++;
            // Sum consecutive entries with same row index
            while (j < col_end && (*rows)[j] == row)
                x += (*data)[j++];
            (*rows)[nnz] = row;
            (*data)[nnz] = x;
            nnz++;
        }
        (*col_ptr)[i+1] = nnz;
    }

    // Update nnz
    m_nnz = nnz;

    // Remove the last (unused) elements
    *rows = (int *) realloc(*rows, nnz*sizeof(int));
    *data = (double *) realloc(*data, nnz*sizeof(double));

    if (*rows == nullptr || *data == nullptr)
        throw std::runtime_error("Failed to resize matrix arrays!");
}

template <typename Float>
CholeskySolver<Float>::CholeskySolver(int n_rows, int nnz, int *ii, int *jj, double *x, MatrixType type, bool cpu) : m_n(n_rows), m_nnz(nnz), m_cpu(cpu) {


    // Placeholders for the CSC matrix data
    int *col_ptr, *rows;
    double *data;

    // Allocate data for the CSC matrix if conversion is needed
    if (type != MatrixType::CSC) {
        col_ptr = (int *) calloc(n_rows + 1, sizeof(int));
        rows = (int *) malloc(nnz * sizeof(int));
        data = (double *) malloc(nnz * sizeof(double));
    }

    // Check the matrix and convert it to CSC if necessary
    if (type == MatrixType::COO)
        coo_to_csc(n_rows, nnz, ii, jj, x, col_ptr, rows, data);
    else if (type == MatrixType::CSR)
        csr_to_csc(n_rows, nnz, ii, jj, x, col_ptr, rows, data);
    else {
        col_ptr = ii;
        rows = jj;
        data = x;
    }

    // CHOLMOD expects a CSC matrix without duplicate entries, so we sum them:
    csc_sort_indices(n_rows, nnz, col_ptr, rows, data);
    csc_sum_duplicates(n_rows, m_nnz, &col_ptr, &rows, &data);
    if (!m_cpu) {
        // Mask of rows already processed
        cuda_check(cuMemAlloc(&m_processed_rows_d, m_n*sizeof(bool)));
        cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false

        // Row id
        cuda_check(cuMemAlloc(&m_stack_id_d, sizeof(int)));
        cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));
    }

    // Run the Cholesky factorization through CHOLMOD and run the analysis
    factorize(col_ptr, rows, data);

    if (type != MatrixType::CSC) {
        free(col_ptr);
        free(rows);
        free(data);
    }
}

template <typename Float>
void CholeskySolver<Float>::factorize(int *col_ptr, int *rows, double *data) {
    cholmod_sparse *A;

    cholmod_start(&m_common);

    m_common.supernodal = CHOLMOD_SIMPLICIAL;
    m_common.final_ll = 1; // compute LL' factorization instead of LDL´ (default for simplicial)
    m_common.nmethods = 1;
    m_common.method[0].ordering = CHOLMOD_NESDIS;

    A = cholmod_allocate_sparse(
        m_n,
        m_n,
        m_nnz,
        1,
        1,
        -1,
        CHOLMOD_REAL,
        &m_common
    );

    // Copy the matrix contents in the CHOLMOD matrix
    int *A_colptr = (int *) A->p;
    int *A_rows = (int *) A->i;
    // CHOLMOD currently only supports the double precision version of the decomposition
    double *A_data = (double*) A->x;
    for (int j=0; j<m_n; j++) {
        A_colptr[j] = col_ptr[j];
        for (int i=col_ptr[j]; i<col_ptr[j+1]; i++) {
            A_rows[i] = rows[i];
            A_data[i] = data[i];
        }
    }
    A_colptr[m_n] = m_nnz;

    // Compute the Cholesky factorization
    m_factor = cholmod_analyze(A, &m_common);
    cholmod_factorize(A, m_factor, &m_common);

    // The previous call sets this flag if it failed
    if (m_common.status == CHOLMOD_NOT_POSDEF)
        throw std::invalid_argument("Matrix is not positive definite!");

    // Setup GPU solving analysis phase
    if (!m_cpu) {
        // Copy permutation
        cuda_check(cuMemAlloc(&m_perm_d, m_n*sizeof(int)));
        cuda_check(cuMemcpyAsync(m_perm_d, m_factor->Perm, m_n*sizeof(int), 0));

        cholmod_sparse *lower_csc = cholmod_factor_to_sparse(m_factor, &m_common);
        // The transpose of a CSC (resp. CSR) matrix is its CSR (resp. CSC) representation
        cholmod_sparse *lower_csr = cholmod_transpose(lower_csc, 1, &m_common);

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

            for (int32_t i=0; i < lower_csc->nzmax; i++) {
                csc_data[i] = (Float) csc_data_ptr[i];
                csr_data[i] = (Float) csr_data_ptr[i];
            }
        }

        int n_rows = lower_csc->nrow;
        int n_entries = lower_csc->nzmax;

        // The CSC representation of a matrix is the same as the CSR of its transpose
        analyze_cuda(n_rows, n_entries, lower_csr->p, lower_csr->i, csr_data, true);

        // To prepare the transpose we merely need to swap the roles of the CSR and CSC representations (CSC rows -> CSR cols, CSC cols -> CSR rows)
        analyze_cuda(n_rows, n_entries, lower_csc->p, lower_csc->i, csc_data, false);

        if (!std::is_same_v<Float, double>) {
            free(csc_data);
            free(csr_data);
        }
        cholmod_free_sparse(&lower_csc, &m_common);
        cholmod_free_sparse(&lower_csr, &m_common);
    }

    cholmod_free_sparse(&A, &m_common);

    // The context and factor will be needed for solving on the CPU, so only free them if we solve on the GPU
    if (!m_cpu) {
        cholmod_free_factor(&m_factor, &m_common);
        cholmod_finish(&m_common);
    }
}

template <typename Float>
void CholeskySolver<Float>::analyze_cuda(int n_rows, int n_entries, void *csr_rows, void *csr_cols, Float *csr_data, bool lower) {

    CUdeviceptr *rows_d = (lower ? &m_lower_rows_d : &m_upper_rows_d);
    CUdeviceptr *cols_d = (lower ? &m_lower_cols_d : &m_upper_cols_d);
    CUdeviceptr *data_d = (lower ? &m_lower_data_d : &m_upper_data_d);
    CUdeviceptr *levels_d = (lower ? &m_lower_levels_d : &m_upper_levels_d);

    // CSR Matrix arrays
    cuda_check(cuMemAlloc(rows_d, (1+n_rows)*sizeof(int)));
    cuda_check(cuMemcpyAsync(*rows_d, csr_rows, (1+n_rows)*sizeof(int), 0));
    cuda_check(cuMemAlloc(cols_d, n_entries*sizeof(int)));
    cuda_check(cuMemcpyAsync(*cols_d, csr_cols, n_entries*sizeof(int), 0));
    cuda_check(cuMemAlloc(data_d, n_entries*sizeof(Float)));
    cuda_check(cuMemcpyAsync(*data_d, csr_data, n_entries*sizeof(Float), 0));

    // Row i belongs in level level_ind[i]
    CUdeviceptr level_ind_d;
    cuda_check(cuMemAlloc(&level_ind_d, n_rows*sizeof(int)));
    cuda_check(cuMemsetD32Async(level_ind_d, 0, n_rows, 0));

    cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, n_rows, 0)); // Initialize to all false

    CUdeviceptr max_lvl_d;
    cuda_check(cuMemAlloc(&max_lvl_d, sizeof(int)));
    cuda_check(cuMemsetD32Async(max_lvl_d, 0, 1, 0));

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

    int *level_ind_h = (int *) malloc(n_rows*sizeof(int));
    cuda_check(cuMemcpyAsync((CUdeviceptr) level_ind_h, level_ind_d, n_rows*sizeof(int), 0));

    int max_lvl_h = 0;
    cuda_check(cuMemcpyAsync((CUdeviceptr) &max_lvl_h, max_lvl_d, sizeof(int), 0));
    int n_levels = max_lvl_h + 1;

    // Construct the (sorted) level array
    int *levels_h = (int *) malloc(n_rows*sizeof(int));
    std::vector<int> level_ptr(n_levels + 1, 0);
    // Count the number of rows per level
    for (int i=0; i<n_rows; i++) {
        level_ptr[1+level_ind_h[i]]++;
    }

    // Convert into the list of pointers to the start of each level
    for (int i=0, S=0; i<n_levels; i++){
        S += level_ptr[i+1];
        level_ptr[i+1] = S;
    }

    // Move all rows to their place in the level array
    for (int i=0; i<n_rows; i++) {
        int row_level = level_ind_h[i]; // Row i belongs to level row_level
        levels_h[level_ptr[row_level]] = i;
        level_ptr[row_level]++;
    }

    cuda_check(cuMemAlloc(levels_d, n_rows*sizeof(int)));
    cuda_check(cuMemcpyAsync(*levels_d, levels_h, n_rows*sizeof(int), 0));

    // Free useless stuff
    free(levels_h);
    free(level_ind_h);
    cuda_check(cuMemFree(level_ind_d));
}

template<typename Float>
void CholeskySolver<Float>::launch_kernel(bool lower, CUdeviceptr x) {
    // Initialize buffers
    cuda_check(cuMemsetD8Async(m_processed_rows_d, 0, m_n, 0)); // Initialize to all false
    cuda_check(cuMemsetD32Async(m_stack_id_d, 0, 1, 0));

    CUdeviceptr rows_d = (lower ? m_lower_rows_d : m_upper_rows_d);
    CUdeviceptr cols_d = (lower ? m_lower_cols_d : m_upper_cols_d);
    CUdeviceptr data_d = (lower ? m_lower_data_d : m_upper_data_d);
    CUdeviceptr levels_d = (lower ? m_lower_levels_d : m_upper_levels_d);

    void *args[11] = {
        &m_nrhs,
        &m_n,
        &m_stack_id_d,
        &levels_d,
        &m_processed_rows_d,
        &rows_d,
        &cols_d,
        &data_d,
        &m_tmp_d,
        &x, // This is the array we read from (i.e. b) for lower, and where we write to (i.e. x) for upper
        &m_perm_d
    };

    CUfunction solve_kernel;
    if(std::is_same_v<Float, float>)
        solve_kernel = (lower ? solve_lower_float : solve_upper_float);
    else
        solve_kernel = (lower ? solve_lower_double : solve_upper_double);

    cuda_check(cuLaunchKernel(solve_kernel,
                            m_n, 1, 1,
                            128, 1, 1,
                            0, 0, args, 0));
}

template <typename Float>
void CholeskySolver<Float>::solve_cuda(int n_rhs, CUdeviceptr b, CUdeviceptr x) {

    if (n_rhs != m_nrhs) {
        if (n_rhs > 128)
            throw std::invalid_argument("The number of RHS should be less than 128.");
        // We need to modify the allocated memory for the solution
        if (m_tmp_d)
            cuda_check(cuMemFree(m_tmp_d));
        cuda_check(cuMemAlloc(&m_tmp_d, n_rhs * m_n * sizeof(Float)));
        m_nrhs = n_rhs;
    }

    // Solve lower
    launch_kernel(true, b);
    // Solve upper
    launch_kernel(false, x);
}

template<typename Float>
void CholeskySolver<Float>::solve_cpu(int n_rhs, Float *b, Float *x) {

    if (n_rhs != m_nrhs) {
        // We need to modify the allocated memory for the solution
        if (m_tmp_chol)
            cholmod_free_dense(&m_tmp_chol, &m_common);
        m_tmp_chol = cholmod_allocate_dense(m_n,
                                            n_rhs,
                                            m_n,
                                            CHOLMOD_REAL,
                                            &m_common
                                            );
        m_nrhs = n_rhs;
    }
    // Set cholmod object fields, converting from C style ordering to F style
    double *tmp = (double *)m_tmp_chol->x;
    for (int i=0; i<m_n; ++i)
        for (int j=0; j<n_rhs; ++j)
            tmp[i + j*m_n] = (double) b[i*n_rhs + j];

    cholmod_dense *cholmod_x = cholmod_solve(CHOLMOD_A, m_factor, m_tmp_chol, &m_common);

    double *sol = (double *) cholmod_x->x;
    for (int i=0; i<m_n; ++i)
        for (int j=0; j<n_rhs; ++j)
            x[i*n_rhs + j] = (Float) sol[i + j*m_n];

    cholmod_free_dense(&cholmod_x, &m_common);
}

template <typename Float>
CholeskySolver<Float>::~CholeskySolver() {
    if (m_cpu){
        cholmod_free_factor(&m_factor, &m_common);
        cholmod_finish(&m_common);
    } else {
        scoped_set_context guard(cu_context);

        cuda_check(cuMemFree(m_processed_rows_d));
        cuda_check(cuMemFree(m_stack_id_d));
        cuda_check(cuMemFree(m_perm_d));
        cuda_check(cuMemFree(m_tmp_d));
        cuda_check(cuMemFree(m_lower_rows_d));
        cuda_check(cuMemFree(m_lower_cols_d));
        cuda_check(cuMemFree(m_lower_data_d));
        cuda_check(cuMemFree(m_upper_rows_d));
        cuda_check(cuMemFree(m_upper_cols_d));
        cuda_check(cuMemFree(m_upper_data_d));
        cuda_check(cuMemFree(m_lower_levels_d));
        cuda_check(cuMemFree(m_upper_levels_d));
    }
}

template class CholeskySolver<float>;
template class CholeskySolver<double>;
