#pragma once

#include <iostream>
#include <vector>
#include "cholmod.h"
#include "triangle_solve.h"
#include "laplacian.h"

template<typename Float>
class CholeskySolver {
public:
    CholeskySolver(uint n_verts, uint n_faces, uint* faces, Float lambda);

    ~CholeskySolver();

    std::vector<Float> solve(Float *b);

private:
    uint m_n;
    uint *m_perm;

    SparseTriangularSolver<Float> *m_solver_l;
    SparseTriangularSolver<Float> *m_solver_u;

};
