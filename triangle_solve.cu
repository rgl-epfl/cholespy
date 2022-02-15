#include <iostream>
#include <math.h>
#include <vector>

#define BLOCK_SIZE 2

void analyse(int n_rows, int n_elements, int* rows, int* columns, float* values, std::vector<size_t> &level_ptr, std::vector<size_t> &levels, std::vector<size_t> &chain_ptr) {
	level_ptr.push_back(0);
	chain_ptr.push_back(0);

	float* dag_values = (float*) malloc(n_elements*sizeof(float));
	std::copy(values, values+n_elements, dag_values);

	std::vector<size_t> candidates;
	for(size_t i = 0; i < n_rows; i++)
		candidates.push_back(i);

	size_t level = 0;
	size_t level_idx = 0;
	while(levels.size() < n_rows) {
		int level_size = 0;
		for(size_t candidate : candidates) {
			bool independent = true;
			for(size_t j=rows[candidate]; j<rows[candidate + 1]; j++) {
				if (columns[j] != candidate && dag_values[j] != 0.0f) {
					independent = false;
					break;
				}
			}

			if (independent) {
				levels.push_back(candidate);
				level_size++;
			}
		}

		candidates.clear();
		for (int i=level; i<level+level_size; i++) {
			int row = levels[i];
			for (int j=0; j<n_rows; j++) {
				for (int k=rows[j]; k<rows[j+1]; k++) {
					if (columns[k] == row && columns[k] != j) {
						dag_values[k] = 0.0f;

						bool candidate = true;
						for (size_t c : candidates) {
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
			if (chain_ptr.back() == level_idx-1)
				chain_ptr.push_back(level_idx);
			else {
				chain_ptr.push_back(level_idx-1);
				chain_ptr.push_back(level_idx);
			}
		}
		level_ptr.push_back(level);
	}
	if (chain_ptr.back() != level_idx)
		chain_ptr.push_back(level_idx);
}

__global__
void solve_row_multiblock(size_t level, size_t* level_ptr, size_t *levels, int* rows, int* columns, float* values, float* b, float* x, bool lower=true) {
    size_t row_idx = level_ptr[level] + blockDim.x * blockIdx.x + threadIdx.x;
	if (row_idx >= level_ptr[level+1])
		return;
	size_t row = levels[row_idx];
	size_t row_start = rows[row];
	size_t row_end = rows[row + 1];
	size_t diag_ptr;
	if (lower) {
		diag_ptr = row_end - 1;
		row_end--;
	} else {
		diag_ptr = row_start;
		row_start++;
	}

	float r = 0.0f;
	for (size_t i=row_start; i<row_end; i++) {
		r += values[i]*x[columns[i]];
	}

	x[row] -= r;
	x[row] /= values[diag_ptr];
}

__global__
void solve_chain(size_t chain_start, size_t chain_end, size_t *level_ptr, size_t *levels, int* rows, int* columns, float* values, float* b, float* x, bool lower=true) {

	for (int level=chain_start; level<chain_end; level++) {
		size_t level_start = level_ptr[level];
		size_t level_end = level_ptr[level+1];
		size_t row_idx = level_start + threadIdx.x;
		if (row_idx >= level_end)
			continue;
		size_t row = levels[row_idx];
		size_t row_start = rows[row];
		size_t row_end = rows[row + 1];
		size_t diag_ptr;
		if (lower) {
			diag_ptr = row_end - 1;
			row_end--;
		} else {
			diag_ptr = row_start;
			row_start++;
		}

		float r = 0.0f;
		for (size_t i=row_start; i<row_end; i++) {
			r += values[i]*x[columns[i]];
		}

		x[row] -= r;
		x[row] /= values[diag_ptr];
		__syncthreads(); // Synchronize before moving to next level
	}
}

void solve_full(std::vector<size_t> chain_ptr, std::vector<size_t> level_ptr_h, size_t* levels, size_t* level_ptr_d, int n_rows, int n_elements, int* rows, int *columns, float* values, float* b, float* x, bool lower=true){

	for (int i=0; i<chain_ptr.size()-1; i++) {
        if (chain_ptr[i]+1 == chain_ptr[i+1]){
			// Multi block kernel
			int num_blocks = (level_ptr_h[chain_ptr[i+1]] - level_ptr_h[chain_ptr[i]] + BLOCK_SIZE - 1) / BLOCK_SIZE;
			solve_row_multiblock<<<num_blocks, BLOCK_SIZE>>>(chain_ptr[i], level_ptr_d, levels, rows, columns, values, b, x, lower);
		} else {
			// Chain fits in one block
			solve_chain<<<1, BLOCK_SIZE>>>(chain_ptr[i], chain_ptr[i+1], level_ptr_d, levels, rows, columns, values, b, x, lower);
		}
        cudaDeviceSynchronize();
	}
}

int main(void) {

    int n_rows = 9;
    int n_entries = 18;

	int rows_h[] = {0, 1, 2, 3, 5, 7, 9, 11, 14, 18};
	int columns_h[] = {0, 1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 4, 7, 2, 3, 4, 8};
	float values_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};

	std::vector<size_t> level_ptr_h;
	std::vector<size_t> levels_h;
	std::vector<size_t> chain_ptr_h;

	analyse(n_rows, n_entries, rows_h, columns_h, values_h, level_ptr_h, levels_h, chain_ptr_h);
	for (int i=0; i<chain_ptr_h.size(); i++)
		std::cout << chain_ptr_h[i] << " ";
	std::cout << std::endl;
	for (int i=0; i<level_ptr_h.size(); i++)
		std::cout << level_ptr_h[i] << " ";
	std::cout << std::endl;

	float b_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
	float* x_h = new float[9];
	std::copy(b_h, b_h+n_rows, x_h);

    int* rows_d;
    int* columns_d;
    size_t* levels_d;
    size_t* level_ptr_d;
    float* x_d;
    float* b_d;
    float* values_d;
    cudaMalloc((void **)&rows_d, (1+n_rows)*sizeof(int));
    cudaMemcpy(rows_d, rows_h, (1+n_rows)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&columns_d, n_entries*sizeof(int));
    cudaMemcpy(columns_d, columns_h, n_entries*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&values_d, n_entries*sizeof(float));
    cudaMemcpy(values_d, values_h, n_entries*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&levels_d, n_rows*sizeof(size_t));
    cudaMemcpy(levels_d, &levels_h[0], n_rows*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&level_ptr_d, level_ptr_h.size()*sizeof(size_t));
    cudaMemcpy(level_ptr_d, &level_ptr_h[0], level_ptr_h.size()*sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&x_d, n_entries*sizeof(float));
    cudaMemcpy(x_d, x_h, n_entries*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&b_d, n_entries*sizeof(float));
    cudaMemcpy(b_d, b_h, n_entries*sizeof(float), cudaMemcpyHostToDevice);

	solve_full(chain_ptr_h, level_ptr_h, levels_d, level_ptr_d, n_rows, n_entries, rows_d, columns_d, values_d, b_d, x_d, true);

    cudaMemcpy(x_h, x_d, n_entries*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i=0; i<9; i++)
		std::cout << x_h[i] << " ";
	std::cout << std::endl;

    cudaFree(rows_d);
    cudaFree(columns_d);
    cudaFree(values_d);
    cudaFree(x_d);
    cudaFree(b_d);
    cudaFree(levels_d);
	return 0;
}

