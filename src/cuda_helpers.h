#pragma once
#include <cuda.h>

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line);