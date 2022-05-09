#pragma once
#include <cstdio>
using size_t = std::size_t;

#define CUDA_ERROR_DEINITIALIZED 4
#define CUDA_SUCCESS 0

using CUcontext    = struct CUctx_st *;
using CUmodule     = struct CUmod_st *;
using CUstream     = struct CUstream_st *;
using CUfunction   = struct CUfunc_st *;
using CUresult     = int;
using CUdevice     = int;
using CUdeviceptr  = void *;

extern CUresult (*cuDeviceGet)(CUdevice *, int);
extern CUresult (*cuDevicePrimaryCtxRelease)(CUdevice);
extern CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
extern CUresult (*cuGetErrorName)(CUresult, const char **);
extern CUresult (*cuGetErrorString)(CUresult, const char **);
extern CUresult (*cuInit)(unsigned int);
extern CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                                  unsigned int, unsigned int, unsigned int,
                                  unsigned int, unsigned int, CUstream, void **,
                                  void **);
extern CUresult (*cuMemAlloc)(void **, size_t);
extern CUresult (*cuMemFree)(void *);
extern CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream);
extern CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream);
extern CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream);
extern CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
extern CUresult (*cuModuleLoadData)(CUmodule *, const void *);
extern CUresult (*cuModuleUnload)(CUmodule);
extern CUresult (*cuCtxPushCurrent)(CUcontext);
extern CUresult (*cuCtxPopCurrent)(CUcontext*);

/// Assert that a CUDA operation is correctly issue
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

extern CUdevice cu_device;
extern CUcontext cu_context;
extern CUmodule cu_module;
extern CUfunction solve_upper_float;
extern CUfunction solve_upper_double;
extern CUfunction solve_lower_float;
extern CUfunction solve_lower_double;
extern CUfunction analysis_lower;
extern CUfunction analysis_upper;

extern bool init_cuda();
extern void shutdown_cuda();
extern void cuda_check_impl(CUresult errval, const char *file, const int line);

struct scoped_set_context {
    scoped_set_context(CUcontext ctx) {
        cuda_check(cuCtxPushCurrent(ctx));
    }
    ~scoped_set_context() {
        cuda_check(cuCtxPopCurrent(nullptr));
    }
};
