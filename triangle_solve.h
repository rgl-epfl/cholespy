#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>

#define BLOCK_SIZE 2

template<typename Float> class SparseTriangularSolver {
public:
	SparseTriangularSolver(uint n_rows, uint n_elements, uint *rows, uint *cols, Float *data, bool lower);
	~SparseTriangularSolver();

	Float* solve(Float *b);

private:
	// Lower or upper triangular matrix
	bool m_lower;
	// Starting index of each "chain" in m_level_ptr_d (see Naumov 2011 for the definition of a chain)
	std::vector<uint> m_chain_ptr;
	// CSR format arrays
	uint *m_rows_d;
	uint *m_cols_d;
	Float *m_data_d;
	// Row count
	uint m_n_rows;
	// Starting index of each level in m_levels
	uint *m_level_ptr_d;
	// Same array but on the CPU
	std::vector<uint> m_level_ptr_h;//TODO: is this really necessary?
	// Sorted indices of rows in each level
	uint *m_levels_d;
	// RHS GPU address
	Float *m_b_d;
	// Solution GPU address
	Float *m_x_d;
};

using SparseTriangularSolverF = SparseTriangularSolver<float>;
using SparseTriangularSolverD = SparseTriangularSolver<double>;
