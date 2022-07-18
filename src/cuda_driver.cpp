#include "cuda_driver.h"
#include "../kernels/kernels.h"
#include <string.h>
#include <stdio.h>
#include <iostream>

#if defined(_WIN32)
#  include <windows.h>
#  define dlsym(ptr, name) GetProcAddress((HMODULE) ptr, name)
#else
#  include <dlfcn.h>
#endif


static void *handle = nullptr;

CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
CUresult (*cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice) = nullptr;
CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;
CUresult (*cuInit)(unsigned int) = nullptr;
CUresult (*cuLaunchKernel)(CUfunction f, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, CUstream, void **,
                           void **) = nullptr;
CUresult (*cuMemAlloc)(void **, size_t) = nullptr;
CUresult (*cuMemFree)(void *) = nullptr;
CUresult (*cuMemcpyAsync)(void *, const void *, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD32Async)(void *, unsigned int, size_t, CUstream) = nullptr;
CUresult (*cuMemsetD8Async)(void *, unsigned char, size_t, CUstream) = nullptr;
CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
CUresult (*cuModuleUnload)(CUmodule) = nullptr;
CUresult (*cuCtxPushCurrent)(CUcontext) = nullptr;
CUresult (*cuCtxPopCurrent)(CUcontext*) = nullptr;

CUdevice cu_device;
CUcontext cu_context;
CUmodule cu_module;
CUfunction solve_upper_float;
CUfunction solve_upper_double;
CUfunction solve_lower_float;
CUfunction solve_lower_double;
CUfunction analysis_lower;
CUfunction analysis_upper;

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (%s): \"%s\" in "
                 "%s:%i.\n", (int) errval, name, msg, file, line);
    }
}

bool init_cuda() {

    if (handle)
        return true;

#if defined(_WIN32)
    handle = (void *) LoadLibraryA("nvcuda.dll");
#elif defined(__APPLE__)
    handle = nullptr;
#else
    handle = dlopen("libcuda.so", RTLD_LAZY);
#endif

    if (!handle)
        return false;

    const char *symbol = nullptr;

    #define LOAD(name, ...)                                      \
        symbol = strlen(__VA_ARGS__ "") > 0                      \
            ? (#name "_" __VA_ARGS__) : #name;                   \
        name = decltype(name)(dlsym(handle, symbol));  \
        if (!name)                                               \
            break;                                               \
        symbol = nullptr

    do {
        LOAD(cuDevicePrimaryCtxRelease, "v2");
        LOAD(cuDevicePrimaryCtxRetain);
        LOAD(cuDeviceGet);
        LOAD(cuCtxPushCurrent, "v2");
        LOAD(cuCtxPopCurrent, "v2");
        LOAD(cuGetErrorName);
        LOAD(cuGetErrorString);
        LOAD(cuInit);
        LOAD(cuMemAlloc, "v2");
        LOAD(cuMemFree, "v2");

        /* By default, cholespy dispatches to the legacy CUDA stream. That
           makes it easier to reliably exchange information with packages that
           enqueue work on other CUDA streams */
#if defined(CHOLESPY_USE_PER_THREAD_DEFAULT_STREAM)
        LOAD(cuLaunchKernel, "ptsz");
        LOAD(cuMemcpyAsync, "ptsz");
        LOAD(cuMemsetD8Async, "ptsz");
        LOAD(cuMemsetD32Async, "ptsz");
#else
        LOAD(cuLaunchKernel);
        LOAD(cuMemcpyAsync);
        LOAD(cuMemsetD8Async);
        LOAD(cuMemsetD32Async);
#endif

        LOAD(cuModuleGetFunction);
        LOAD(cuModuleLoadData);
        LOAD(cuModuleUnload);
    } while (false);

    if (symbol) {
        fprintf(stderr,
                "cuda_init(): could not find symbol \"%s\" -- disabling "
                "CUDA backend!", symbol);
        return false;
    }

    cuda_check(cuInit(0));
    cuda_check(cuDeviceGet(&cu_device, 0));
    cuda_check(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuda_check(cuCtxPushCurrent(cu_context));
    cuda_check(cuModuleLoadData(&cu_module, (void *) imageBytes));
    cuda_check(cuModuleGetFunction(&solve_lower_float, cu_module, (char *)"solve_lower_float"));
    cuda_check(cuModuleGetFunction(&solve_lower_double, cu_module, (char *)"solve_lower_double"));
    cuda_check(cuModuleGetFunction(&solve_upper_float, cu_module, (char *)"solve_upper_float"));
    cuda_check(cuModuleGetFunction(&solve_upper_double, cu_module, (char *)"solve_upper_double"));
    cuda_check(cuModuleGetFunction(&analysis_lower, cu_module, (char *)"analysis_lower"));
    cuda_check(cuModuleGetFunction(&analysis_upper, cu_module, (char *)"analysis_upper"));

    return true;
}

void shutdown_cuda() {
    if (!handle)
        return;

    cuda_check(cuDevicePrimaryCtxRelease(cu_device));

#if defined(_WIN32)
    FreeLibrary((HMODULE) handle);
#elif !defined(__APPLE__)
    dlclose(handle);
#endif
}
