#include <cuda.h>
#include <stdio.h>
#include <type_traits>
#include "../kernels/kernels.h"

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (%s): \"%s\" in "
                 "%s:%i.\n", (int) errval, name, msg, file, line);
    }
}

CUdevice cu_device;
CUcontext cu_context;
CUmodule cu_module;
CUfunction solve_chain;
CUfunction solve_row_multiblock;
bool init = false;

// TODO: I'm not sure it's a great idea to template this like this
template <typename Float>
void initCuda() {
	cuda_check(cuInit(0));
	cuda_check(cuDeviceGet(&cu_device, 0));
	cuda_check(cuCtxCreate(&cu_context, 0, cu_device));
	cuda_check(cuModuleLoadData(&cu_module, (void *) imageBytes));
	if (std::is_same_v<Float, float>) {
		cuda_check(cuModuleGetFunction(&solve_chain, cu_module, (char *)"solve_chain_float"));
		cuda_check(cuModuleGetFunction(&solve_row_multiblock, cu_module, (char *)"solve_row_multiblock_float"));
	} else {
		cuda_check(cuModuleGetFunction(&solve_chain, cu_module, (char *)"solve_chain_double"));
		cuda_check(cuModuleGetFunction(&solve_row_multiblock, cu_module, (char *)"solve_row_multiblock_double"));
	}
}
