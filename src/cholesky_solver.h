#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"
#include "triangle_solve.h"
#include "laplacian.h"
#include "cuda_helpers.h"

template<typename Float>
class CholeskySolver {
public:
    CholeskySolver(uint n_verts, uint n_faces, uint* faces, Float lambda);

    ~CholeskySolver();

    std::vector<Float> solve(Float *b);

private:
    uint m_n;
    uint *m_perm;

	// CSC of the lower factor / CSR of the upper
    CUdeviceptr m_csc_cols_d;
    CUdeviceptr m_csc_rows_d;
    CUdeviceptr m_csc_data_d;
	// CSR of the lower factor / CSC of the upper
    CUdeviceptr m_csr_cols_d;
    CUdeviceptr m_csr_rows_d;
    CUdeviceptr m_csr_data_d;

    SparseTriangularSolver<Float> *m_solver_l;
    SparseTriangularSolver<Float> *m_solver_u;

};
