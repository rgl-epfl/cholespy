#pragma once
#include <cuda.h>
#include <stdio.h>
#include <type_traits>
#include "../kernels/kernels.h"
#include "cuda_helpers.h"

CUdevice cu_device;
CUcontext cu_context;
CUmodule cu_module;
CUfunction solve_chain;
CUfunction solve_row_multiblock;
CUfunction solve_upper;
CUfunction solve_lower;
CUfunction find_roots;
CUfunction analyze;
CUfunction find_roots_in_candidates;
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
        cuda_check(cuModuleGetFunction(&solve_lower, cu_module, (char *)"solve_lower_float"));
        cuda_check(cuModuleGetFunction(&solve_upper, cu_module, (char *)"solve_upper_float"));
    } else {
        cuda_check(cuModuleGetFunction(&solve_chain, cu_module, (char *)"solve_chain_double"));
        cuda_check(cuModuleGetFunction(&solve_row_multiblock, cu_module, (char *)"solve_row_multiblock_double"));
        cuda_check(cuModuleGetFunction(&solve_lower, cu_module, (char *)"solve_lower_double"));
        cuda_check(cuModuleGetFunction(&solve_upper, cu_module, (char *)"solve_upper_double"));
    }
    cuda_check(cuModuleGetFunction(&find_roots, cu_module, (char *)"find_roots"));
    cuda_check(cuModuleGetFunction(&analyze, cu_module, (char *)"analyze"));
    cuda_check(cuModuleGetFunction(&find_roots_in_candidates, cu_module, (char *)"find_roots_in_candidates"));
}
