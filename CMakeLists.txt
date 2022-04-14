cmake_minimum_required(VERSION 3.17...3.22)

project(cholesky LANGUAGES CXX C VERSION "0.1.0")

# Nanobind setup from https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt
if (NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

if (SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
elseif (MSVC)
  # MSVC needs a little extra help finding the Python library
  find_package(PythonInterp)
  find_package(Python)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)
# Find CUDA to link to the CUDA Driver API
find_package(CUDAToolkit REQUIRED)
# CHOLMOD package
find_package(SuiteSparse CONFIG REQUIRED)

add_library(
  cholesky_cuda STATIC

  kernels/kernels.h
  src/cuda_setup.h
  src/cholesky_solver.h      src/cholesky_solver.cpp
)

set_property(TARGET cholesky_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
# set_property(TARGET cholesky_cuda PROPERTY CXX_STANDARD "17")
target_compile_features(cholesky_cuda PRIVATE cxx_std_17)

target_link_libraries(cholesky_cuda
  CUDA::cuda_driver
  SuiteSparse::cholmod
)


nanobind_add_module(_cholesky_core NB_STATIC src/main.cpp)
set_property(TARGET _cholesky_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(_cholesky_core PRIVATE cxx_std_17)
target_link_libraries(_cholesky_core PUBLIC cholesky_cuda)

#target_include_directories(_cholesky_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
target_compile_definitions(_cholesky_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _cholesky_core LIBRARY DESTINATION .)
