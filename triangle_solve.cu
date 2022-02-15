#include "triangle_solve.h"

template<typename Float>
SparseTriangularSolver<Float>::SparseTriangularSolver(uint n_rows, uint n_entries, uint* rows, uint* cols, Float* data, bool lower) : m_lower(lower), m_n_rows(n_rows) {
	std::vector<uint> levels;//TODO: this could be an array, as it always has as many elements as rows
	levels.reserve(n_rows);

	m_level_ptr_h.reserve(n_rows + 1); // Worst-case size (100% dense triangular matrix)
	m_level_ptr_h.push_back(0);

	m_chain_ptr.reserve(n_rows + 1); // TODO: could use a more educated guess depending on block size and n_rows
	m_chain_ptr.push_back(0);

	Float* dag_values = (Float*) malloc(n_entries*sizeof(Float));
	std::copy(data, data+n_entries, dag_values);

	std::vector<uint> candidates;
	for(uint i = 0; i < n_rows; i++)
		candidates.push_back(i);

	uint level = 0;
	uint level_idx = 0;
	while(levels.size() < n_rows) {
		uint level_size = 0;
		for(uint candidate : candidates) {
			bool independent = true;
			for(uint j=rows[candidate]; j<rows[candidate + 1]; j++) {
				if (cols[j] != candidate && dag_values[j] != 0.0f) {
					independent = false;
					break;
				}
			}

			if (independent) {
				levels.push_back(candidate);
				level_size++;
			}
		}

		// Sort indices in the current level
		std::sort(levels.end() - level_size, levels.end());

		candidates.clear();
		for (int i=level; i<level+level_size; i++) {
			int row = levels[i];
			for (int j=0; j<n_rows; j++) {
				for (int k=rows[j]; k<rows[j+1]; k++) {
					if (cols[k] == row && cols[k] != j) {
						dag_values[k] = 0.0f;

						bool candidate = true;
						// TODO: replace this with a set
						for (uint c : candidates) {
							if (c == j) {
								candidate = false;
								break;
							}
						}
						if (candidate)
							candidates.push_back(j);
					}
				}
			}
		}

		level += level_size;
		level_idx++;
		if (level_size > BLOCK_SIZE) {
			if (m_chain_ptr.back() == level_idx-1)
				m_chain_ptr.push_back(level_idx);
			else {
				m_chain_ptr.push_back(level_idx-1);
				m_chain_ptr.push_back(level_idx);
			}
		}
		m_level_ptr_h.push_back(level);
	}
	if (m_chain_ptr.back() != level_idx)
		m_chain_ptr.push_back(level_idx);

	m_chain_ptr.shrink_to_fit();
	m_level_ptr_h.shrink_to_fit();

	free(dag_values);

	// Copy stuff to the GPU
	// CSR Matrix arrays
    cudaMalloc((void **)&m_rows_d, (1+n_rows)*sizeof(uint));
    cudaMemcpy(m_rows_d, rows, (1+n_rows)*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_cols_d, n_entries*sizeof(uint));
    cudaMemcpy(m_cols_d, cols, n_entries*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data_d, n_entries*sizeof(Float));
    cudaMemcpy(m_data_d, data, n_entries*sizeof(Float), cudaMemcpyHostToDevice);
	// Solve data structure
    cudaMalloc((void **)&m_level_ptr_d, m_level_ptr_h.size()*sizeof(uint));
    cudaMemcpy(m_level_ptr_d, &m_level_ptr_h[0], m_level_ptr_h.size()*sizeof(uint), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_levels_d, n_rows*sizeof(uint));
    cudaMemcpy(m_levels_d, &levels[0], n_rows*sizeof(uint), cudaMemcpyHostToDevice);
	// RHS and solution
	cudaMalloc((void**)&m_b_d, n_rows*sizeof(Float));
	cudaMalloc((void**)&m_x_d, n_rows*sizeof(Float));
}

template<typename Float>
SparseTriangularSolver<Float>::~SparseTriangularSolver() {
	cudaFree(m_rows_d);
	cudaFree(m_cols_d);
	cudaFree(m_data_d);
	cudaFree(m_level_ptr_d);
	cudaFree(m_levels_d);
	cudaFree(m_b_d);
	cudaFree(m_x_d);
}

template<typename Float>
Float* SparseTriangularSolver<Float>::solve(Float *b) {

	cudaMemcpy(m_b_d, b, m_n_rows*sizeof(Float), cudaMemcpyHostToDevice);
	cudaMemcpy(m_x_d, b, m_n_rows*sizeof(Float), cudaMemcpyHostToDevice);//TODO: device to device copy instead?

	for (int i=0; i<m_chain_ptr.size()-1; i++) {
        if (m_chain_ptr[i]+1 == m_chain_ptr[i+1]){
			// Multi block kernel
			//TODO: this requires storing level_ptr on the CPU, is this really necessary?
			int num_blocks = (m_level_ptr_h[m_chain_ptr[i+1]] - m_level_ptr_h[m_chain_ptr[i]] + BLOCK_SIZE - 1) / BLOCK_SIZE;
			solve_row_multiblock<<<num_blocks, BLOCK_SIZE>>>(m_chain_ptr[i], m_level_ptr_d, m_levels_d, m_rows_d, m_cols_d, m_data_d, m_b_d, m_x_d, m_lower);
		} else {
			// Chain fits in one block
			solve_chain<<<1, BLOCK_SIZE>>>(m_chain_ptr[i], m_chain_ptr[i+1], m_level_ptr_d, m_levels_d, m_rows_d, m_cols_d, m_data_d, m_b_d, m_x_d, m_lower);
		}
        cudaDeviceSynchronize();
	}

	Float *x_h = (Float*) malloc(m_n_rows*sizeof(Float));
	cudaMemcpy(x_h, m_x_d, m_n_rows*sizeof(Float), cudaMemcpyDeviceToHost);
	return x_h;
}

template<typename Float>
__global__ void solve_row_multiblock(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {
    uint row_idx = level_ptr[level] + blockDim.x * blockIdx.x + threadIdx.x;
	if (row_idx >= level_ptr[level+1])
		return;
	uint row = levels[row_idx];
	uint row_start = rows[row];
	uint row_end = rows[row + 1];
	uint diag_ptr;
	if (lower) {
		diag_ptr = row_end - 1;
		row_end--;
	} else {
		diag_ptr = row_start;
		row_start++;
	}

	Float r = 0.0f;
	for (uint i=row_start; i<row_end; i++) {
		r += values[i]*x[columns[i]];
	}

	x[row] -= r;
	x[row] /= values[diag_ptr];
}

template<typename Float>
__global__ void solve_chain(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {

	for (uint level=chain_start; level<chain_end; level++) {
		uint level_start = level_ptr[level];
		uint level_end = level_ptr[level+1];
		uint row_idx = level_start + threadIdx.x;
		if (row_idx >= level_end)
			continue;
		uint row = levels[row_idx];
		uint row_start = rows[row];
		uint row_end = rows[row + 1];
		uint diag_ptr;
		if (lower) {
			diag_ptr = row_end - 1;
			row_end--;
		} else {
			diag_ptr = row_start;
			row_start++;
		}

		Float r = 0.0f;
		for (uint i=row_start; i<row_end; i++) {
			r += values[i]*x[columns[i]];
		}

		x[row] -= r;
		x[row] /= values[diag_ptr];
		__syncthreads(); // Synchronize before moving to next level
	}
}

int main(void) {

	typedef float Float;

    uint n_rows = 9;
    uint n_entries = 18;

	uint rows_h[] = {0, 1, 2, 3, 5, 7, 9, 11, 14, 18};
	uint columns_h[] = {0, 1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 4, 7, 2, 3, 4, 8};
	Float values_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};


	SparseTriangularSolver<Float> solver(n_rows, n_entries, rows_h, columns_h, values_h, true);

	Float b_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

	Float *x_h = solver.solve(b_h);

	for (int i=0; i<9; i++)
		std::cout << x_h[i] << " ";
	std::cout << std::endl;

	return 0;
}

