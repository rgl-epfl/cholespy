cmake_minimum_required(VERSION 3.18...3.22)

project(cholespy LANGUAGES CXX C VERSION "0.1.5")

# Nanobind setup from https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt
if (NOT SKBUILD)
  message(WARNING "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

#  Check if submodules have been checked out, or fail early
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/suitesparse-metis-for-windows/cmake")
  message(FATAL_ERROR "The dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
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

if (UNIX AND NOT APPLE)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
endif()

# Run `nanobind.cmake_dir()` from Python to detect install location
execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)
# CHOLMOD package
add_subdirectory(ext/suitesparse-metis-for-windows)
include_directories("${suitesparseconfig_SOURCE_DIR}")
include_directories("${CHOLMOD_SOURCE_DIR}/Include")

nanobind_add_module(
  _cholespy_core NB_STATIC

  kernels/kernels.h
  src/cuda_driver.h          src/cuda_driver.cpp
  src/cholesky_solver.h      src/cholesky_solver.cpp
  src/docstr.h               src/main.cpp)

set_property(TARGET _cholespy_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET metis PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(_cholespy_core PRIVATE cxx_std_17)
target_link_libraries(_cholespy_core PUBLIC cholmod)

target_compile_definitions(_cholespy_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _cholespy_core LIBRARY DESTINATION .)

if (WIN32)
  file(GLOB SUITESPARSE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ext/suitesparse-metis-for-windows/lapack_windows/x64/*.dll)
  message(STATUS ${SUITESPARSE_FILES})
  install(FILES ${SUITESPARSE_FILES} DESTINATION .)
endif()
