#pragma once
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
CUfunction solve_upper_float;
CUfunction solve_upper_double;
CUfunction solve_lower_float;
CUfunction solve_lower_double;
CUfunction analysis_lower;
CUfunction analysis_upper;
bool init = false;

void initCuda() {
    if (init)
        return;

    cuda_check(cuInit(0));
    cuda_check(cuDeviceGet(&cu_device, 0));
    // TODO: flags?
    cuda_check(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuda_check(cuCtxSetCurrent(cu_context));
    cuda_check(cuModuleLoadData(&cu_module, (void *) imageBytes));
    cuda_check(cuModuleGetFunction(&solve_lower_float, cu_module, (char *)"solve_lower_float"));
    cuda_check(cuModuleGetFunction(&solve_lower_double, cu_module, (char *)"solve_lower_double"));
    cuda_check(cuModuleGetFunction(&solve_upper_float, cu_module, (char *)"solve_upper_float"));
    cuda_check(cuModuleGetFunction(&solve_upper_double, cu_module, (char *)"solve_upper_double"));
    cuda_check(cuModuleGetFunction(&analysis_lower, cu_module, (char *)"analysis_lower"));
    cuda_check(cuModuleGetFunction(&analysis_upper, cu_module, (char *)"analysis_upper"));
    // Do all this only once
    init = true;

    // TODO: release context ?
}
