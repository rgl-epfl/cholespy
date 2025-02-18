#pragma once
#include <cstdio>
#include <cstdint>
#include <unordered_map>
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
extern CUresult (*cuCtxGetDevice)(CUdevice*);

/// Assert that a CUDA operation is correctly issue
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

extern std::unordered_map<uint32_t, CUcontext> device_contexts;
extern std::unordered_map<uint32_t, CUdevice> device_devices;
extern std::unordered_map<uint32_t, CUmodule> device_modules;
extern std::unordered_map<uint32_t, CUfunction> device_solve_upper_float;
extern std::unordered_map<uint32_t, CUfunction> device_solve_upper_double;
extern std::unordered_map<uint32_t, CUfunction> device_solve_lower_float;
extern std::unordered_map<uint32_t, CUfunction> device_solve_lower_double;
extern std::unordered_map<uint32_t, CUfunction> device_analysis_lower;
extern std::unordered_map<uint32_t, CUfunction> device_analysis_upper;

extern bool init_cuda(uint32_t deviceID);
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
