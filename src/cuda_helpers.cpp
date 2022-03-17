#include "cuda_helpers.h"
#include <stdio.h>

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (%s): \"%s\" in "
                 "%s:%i.\n", (int) errval, name, msg, file, line);
    }
}
