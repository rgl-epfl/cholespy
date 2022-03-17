cmake_minimum_required(VERSION 3.17...3.19)

project(cholesky LANGUAGES CXX C VERSION "0.1.0")

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)
# Find CUDA to link to the CUDA Driver API
find_package(CUDAToolkit REQUIRED)
# CHOLMOD package
find_package(SuiteSparse CONFIG REQUIRED)

add_library(
  cholesky_cuda STATIC

  src/cuda_setup.h
  src/cuda_helpers.h           src/cuda_helpers.cpp
  src/cholesky_solver.h      src/cholesky_solver.cpp
  src/triangle_solve.h       src/triangle_solve.cpp
  src/laplacian.h            src/laplacian.cpp
  kernels/kernels.h
)

set_property(TARGET cholesky_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
# set_property(TARGET cholesky_cuda PROPERTY CXX_STANDARD "17")
target_compile_features(cholesky_cuda PRIVATE cxx_std_17)

target_link_libraries(cholesky_cuda
  CUDA::cuda_driver
  SuiteSparse::cholmod
)


pybind11_add_module(_cholesky_core MODULE src/main.cpp)
set_property(TARGET _cholesky_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(_cholesky_core PRIVATE cxx_std_17)
target_link_libraries(_cholesky_core PUBLIC cholesky_cuda)

#target_include_directories(_cholesky_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
target_compile_definitions(_cholesky_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _cholesky_core DESTINATION .)
