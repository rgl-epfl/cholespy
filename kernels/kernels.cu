#include <stdio.h>


template<typename Float>
__device__ void solve_row_multiblock(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {
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
__device__ void solve_chain(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, Float* values, Float* b, Float* x, bool lower) {
	for (uint level=chain_start; level<chain_end; level++) {
		uint level_start = level_ptr[level];
		uint level_end = level_ptr[level+1];
		uint row_idx = level_start + threadIdx.x;
		if (row_idx < level_end) {
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
		__syncthreads(); // Synchronize before moving to next level
	}
}


extern "C" __global__ void solve_row_multiblock_float(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, float* values, float* b, float* x, bool lower) {
	solve_row_multiblock<float>(level, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_row_multiblock_double(uint level, uint* level_ptr, uint *levels, uint* rows, uint* columns, double* values, double* b, double* x, bool lower) {
	solve_row_multiblock<double>(level, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_chain_float(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, float* values, float* b, float* x, bool lower) {
	solve_chain<float>(chain_start, chain_end, level_ptr, levels, rows, columns, values, b, x, lower);
}

extern "C" __global__ void solve_chain_double(uint chain_start, uint chain_end, uint *level_ptr, uint *levels, uint* rows, uint* columns, double* values, double* b, double* x, bool lower) {
	solve_chain<double>(chain_start, chain_end, level_ptr, levels, rows, columns, values, b, x, lower);
}
