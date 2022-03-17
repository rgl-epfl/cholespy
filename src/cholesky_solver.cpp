#include "cholesky_solver.h"
#include "cuda_setup.h"

template <typename Float>
CholeskySolver<Float>::CholeskySolver(uint n_verts, uint n_faces, uint *faces, Float lambda) {

    // Initialize CUDA and load the kernels if not already done
    if (!init)
        initCuda<Float>();

    Laplacian<Float> L(n_verts, n_faces, faces, lambda);

    cholmod_sparse *A;
    cholmod_factor *F;
    cholmod_common c;

    cholmod_start(&c);

    c.supernodal = CHOLMOD_SIMPLICIAL;
    c.final_ll = 1; // compute LL' factorization instead of LDL´ (default for simplicial)
    c.print = 5; // log level
    //TODO: Not sure this is necessary, need to benchmark
    c.nmethods = 1;
    c.method[0].ordering = CHOLMOD_NESDIS;

    std::vector<uint> col_ptr = L.col_ptr();
    std::vector<uint> rows = L.rows();
    std::vector<Float> data = L.data();

    m_n = col_ptr.size() - 1;

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
            A_data[i] = (double) data[i];
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

    /*
        The constructor of the SparseTriangularSolver class takes as argument
        the CSC and CSR representations of the matrix. This is convenient since
        the CSR representation of the lower triangular matrix is the CSC of the
        upper one (and vice versa). Therefore we upload to the GPU the CSC
        representation of the lower and of its transpose, and then swap the
        pointers in the constructors.
    */

    uint n_rows = lower_csc->nrow;
    uint n_entries = lower_csc->nzmax;

    // CSC Matrix arrays
    cuda_check(cuMemAlloc(&m_csc_cols_d, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemcpyHtoD(m_csc_cols_d, lower_csc->p, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemAlloc(&m_csc_rows_d, n_entries*sizeof(uint)));
    cuda_check(cuMemcpyHtoD(m_csc_rows_d, lower_csc->i, n_entries*sizeof(uint)));
    cuda_check(cuMemAlloc(&m_csc_data_d, n_entries*sizeof(Float)));
    cuda_check(cuMemcpyHtoD(m_csc_data_d, csc_data, n_entries*sizeof(Float)));

    // CSR Matrix arrays
    cuda_check(cuMemAlloc(&m_csr_rows_d, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemcpyHtoD(m_csr_rows_d, lower_csr->p, (1+n_rows)*sizeof(uint)));
    cuda_check(cuMemAlloc(&m_csr_cols_d, n_entries*sizeof(uint)));
    cuda_check(cuMemcpyHtoD(m_csr_cols_d, lower_csr->i, n_entries*sizeof(uint)));
    cuda_check(cuMemAlloc(&m_csr_data_d, n_entries*sizeof(Float)));
    cuda_check(cuMemcpyHtoD(m_csr_data_d, csr_data, n_entries*sizeof(Float)));


    m_solver_l = new SparseTriangularSolver<Float>(n_rows, n_entries, m_csc_cols_d, m_csc_rows_d, m_csr_rows_d, m_csr_cols_d, m_csr_data_d, true);
    // To prepare the transpose we merely need to swap the roles of the CSR and CSC representations (CSC rows -> CSR cols, CSC cols -> CSR rows)
    m_solver_u = new SparseTriangularSolver<Float>(n_rows, n_entries, m_csr_rows_d, m_csr_cols_d, m_csc_cols_d, m_csc_rows_d, m_csc_data_d, false);

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
std::vector <Float> CholeskySolver<Float>::solve(Float *b) {
    std::vector<Float> tmp, sol;
    sol.resize(m_n, 0.);
    tmp.resize(m_n, 0.);

    for (uint i=0; i<m_n; i++)
        tmp[i] = b[m_perm[i]];

    tmp = m_solver_l->solve(&tmp[0]);
    tmp = m_solver_u->solve(&tmp[0]);

    for (uint i=0; i<m_n; i++)
        sol[m_perm[i]] = tmp[i];

    return sol;
}

template <typename Float>
CholeskySolver<Float>::~CholeskySolver() {
    cuda_check(cuMemFree(m_csr_rows_d));
    cuda_check(cuMemFree(m_csr_cols_d));
    cuda_check(cuMemFree(m_csr_data_d));
    cuda_check(cuMemFree(m_csc_rows_d));
    cuda_check(cuMemFree(m_csc_cols_d));
    cuda_check(cuMemFree(m_csc_data_d));
    free(m_perm);
}

template class CholeskySolver<float>;
template class CholeskySolver<double>;
