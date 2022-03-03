#pragma once

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <math.h>

/**
 * Class Laplacian
 *
 * This class implements a CSC representation of the matrix I + lambda*L, where
 * L is the combinatorial Laplacian (for now, only a triangle mesh is
 * supported.) TODO: find a better suited name.
 *
 * The purpose of this class is to construct the data required by CHOLMOD to run
 * the Cholesky factorization of the matrix. As a consequence, we only build the
 * lower half of the matrix, since it is symmetric and CHOLMOD effectively only
 * uses this part.
 */
template <typename Float>
class Laplacian {
public:
    Laplacian(uint n_verts, uint n_faces, uint* faces, Float lambda);

    std::vector<uint> col_ptr() {return m_col_ptr;}
    std::vector<uint> rows() {return m_rows;}
    std::vector<Float> data() {return m_data;}

private:
    std::vector<uint>  m_col_ptr;
    std::vector<uint>  m_rows;
    std::vector<Float> m_data;
};
