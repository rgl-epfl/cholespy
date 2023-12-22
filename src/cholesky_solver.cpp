#include "cholesky_solver.h"
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
CholeskySolver<Float>::CholeskySolver(int n_rows, int nnz, int *ii, int *jj, double *x, MatrixType type) : m_n(n_rows), m_nnz(nnz) {


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

    cholmod_free_sparse(&A, &m_common);
}

template<typename Float>
void CholeskySolver<Float>::solve_cpu(int n_rhs, Float *b, Float *x, int mode) {

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

    cholmod_dense *cholmod_x = cholmod_solve(mode, m_factor, m_tmp_chol, &m_common);

    double *sol = (double *) cholmod_x->x;
    for (int i=0; i<m_n; ++i)
        for (int j=0; j<n_rhs; ++j)
            x[i*n_rhs + j] = (Float) sol[i + j*m_n];

    cholmod_free_dense(&cholmod_x, &m_common);
}

template <typename Float>
CholeskySolver<Float>::~CholeskySolver() {
    cholmod_free_factor(&m_factor, &m_common);
    cholmod_finish(&m_common);
}

template class CholeskySolver<float>;
template class CholeskySolver<double>;
