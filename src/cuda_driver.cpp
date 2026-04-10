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

void* handle = nullptr;

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
CUresult (*cuCtxGetDevice)(CUdevice*) = nullptr;

std::unordered_map<uint32_t, CUcontext> device_contexts;
std::unordered_map<uint32_t, CUdevice> device_devices;
std::unordered_map<uint32_t, CUmodule> device_modules;
std::unordered_map<uint32_t, CUfunction> device_solve_upper_float;
std::unordered_map<uint32_t, CUfunction> device_solve_upper_double;
std::unordered_map<uint32_t, CUfunction> device_solve_lower_float;
std::unordered_map<uint32_t, CUfunction> device_solve_lower_double;
std::unordered_map<uint32_t, CUfunction> device_analysis_lower;
std::unordered_map<uint32_t, CUfunction> device_analysis_upper;

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *name = nullptr, *msg = nullptr;
        cuGetErrorName(errval, &name);
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (%s): \"%s\" in "
                 "%s:%i.\n", (int) errval, name, msg, file, line);
    }
}

bool init_cuda(uint32_t deviceID) {

    if (device_contexts.find(deviceID) != device_contexts.end()) {
        return true;
    }

    if (handle == nullptr){

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
            LOAD(cuCtxGetDevice);

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

    }


    CUdevice cu_device;
    CUcontext cu_context;
    CUmodule cu_module;
    CUfunction solve_lower_double;
    CUfunction solve_upper_double;
    CUfunction solve_lower_float;
    CUfunction solve_upper_float;
    CUfunction analysis_lower;
    CUfunction analysis_upper;

    cuda_check(cuInit(0));
    cuda_check(cuDeviceGet(&cu_device, deviceID));
    cuda_check(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuda_check(cuCtxPushCurrent(cu_context));
    cuda_check(cuModuleLoadData(&cu_module, (void *) imageBytes));
    cuda_check(cuModuleGetFunction(&solve_lower_float, cu_module, (char *)"solve_lower_float"));
    cuda_check(cuModuleGetFunction(&solve_lower_double, cu_module, (char *)"solve_lower_double"));
    cuda_check(cuModuleGetFunction(&solve_upper_float, cu_module, (char *)"solve_upper_float"));
    cuda_check(cuModuleGetFunction(&solve_upper_double, cu_module, (char *)"solve_upper_double"));
    cuda_check(cuModuleGetFunction(&analysis_lower, cu_module, (char *)"analysis_lower"));
    cuda_check(cuModuleGetFunction(&analysis_upper, cu_module, (char *)"analysis_upper"));

    // record context infos 
    device_contexts[deviceID] = cu_context;
    device_devices[deviceID] = cu_device;
    device_modules[deviceID] = cu_module;
    device_solve_lower_double[deviceID] = solve_lower_double;
    device_solve_upper_double[deviceID] = solve_upper_double;
    device_solve_lower_float[deviceID] = solve_lower_float;
    device_solve_upper_float[deviceID] = solve_upper_float;
    device_analysis_lower[deviceID] = analysis_lower;
    device_analysis_upper[deviceID] = analysis_upper;

    return true;
}

void shutdown_cuda() {
    // release all gpu resources when unloading python module
    for (auto iter = device_devices.begin(); iter != device_devices.end(); iter++){
        CUdevice cu_device = iter->second;
        cuda_check(cuDevicePrimaryCtxRelease(cu_device));
    }

#if defined(_WIN32)
    FreeLibrary((HMODULE) handle);
#elif !defined(__APPLE__)
    dlclose(handle);
#endif
}
