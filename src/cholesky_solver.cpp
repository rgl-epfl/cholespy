#include "cholesky_solver.h"

template <typename Float>
CholeskySolver<Float>::CholeskySolver(uint n_verts, uint n_faces, uint *faces, Float lambda) {

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

    cholmod_sparse *lower_tri_csc = cholmod_factor_to_sparse(F, &c);
    // Transpose to get the other solver
    cholmod_sparse *upper_tri_csc = cholmod_transpose(lower_tri_csc, 1, &c);

    // Since we can only factorize in double precision mode, we have to recast to Float
    Float *lower_data;
    Float *upper_data;
    if (std::is_same_v<Float, double>) {
        lower_data = (Float *) lower_tri_csc->x;
        upper_data = (Float *) upper_tri_csc->x;
    } else {
        lower_data = (Float *)malloc(lower_tri_csc->nzmax * sizeof(Float));
        upper_data = (Float *)malloc(upper_tri_csc->nzmax * sizeof(Float));

        double *lower_data_ptr = (double *) lower_tri_csc->x;
        double *upper_data_ptr = (double *) upper_tri_csc->x;

        for (uint32_t i=0; i < lower_tri_csc->nzmax; i++) {
            lower_data[i] = (Float) lower_data_ptr[i];
            upper_data[i] = (Float) upper_data_ptr[i];
        }
    }

    /*
        SparseTriangularSolver expects a CSR matrix, but we have a CSC, which is
        the CSR of the transpose. Therefore by giving the CSC representation of
        the lower triangular factor, we actually process its transpose and thus
        generate the upper triangular solver.
    */
    m_solver_u = new SparseTriangularSolver<Float>(lower_tri_csc->nrow, lower_tri_csc->nzmax, (uint*)lower_tri_csc->p, (uint*)lower_tri_csc->i, lower_data, false);
    m_solver_l = new SparseTriangularSolver<Float>(upper_tri_csc->nrow, upper_tri_csc->nzmax, (uint*)upper_tri_csc->p, (uint*)upper_tri_csc->i, upper_data, true);

    // Free CHOLMOD stuff
    cholmod_free_sparse(&A, &c);
    cholmod_free_sparse(&lower_tri_csc, &c);
    cholmod_free_sparse(&upper_tri_csc, &c);
    cholmod_free_factor(&F, &c);
    cholmod_finish(&c);
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
    free(m_perm);
}

template class CholeskySolver<float>;
template class CholeskySolver<double>;
