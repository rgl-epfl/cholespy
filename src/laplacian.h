#include <stdio.h>
#include <vector>
#include <algorithm>
#include <math.h>

// Uniform laplacian in the CSC format
template <typename Float>
class Laplacian {
public:
	Laplacian(uint n_verts, uint n_faces, uint* faces, Float lambda);

	std::vector<uint> col_ptr() {return m_col_ptr;}
	std::vector<uint> rows() {return m_rows;}
	std::vector<Float> data() {return m_data;}

private:
	Float 	           m_lambda;
	std::vector<uint>  m_col_ptr;
	std::vector<uint>  m_rows;
	std::vector<Float> m_data;
};
