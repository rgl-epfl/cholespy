cmake_minimum_required(VERSION 3.18...3.22)

project(cholespy LANGUAGES CXX C)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

#  Check if submodules have been checked out, or fail early
if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/suitesparse-metis-for-windows/cmake")
  message(FATAL_ERROR "The dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)
# CHOLMOD package
add_subdirectory(ext/suitesparse-metis-for-windows)
include_directories("${suitesparseconfig_SOURCE_DIR}")
include_directories("${CHOLMOD_SOURCE_DIR}/Include")

nanobind_add_module(
  _cholespy_core NB_STATIC

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  src/cholesky_solver.h
  src/cholesky_solver.cpp
  src/docstr.h
  src/cholespy_core.cpp)

set_property(TARGET _cholespy_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET metis PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(_cholespy_core PRIVATE cxx_std_17)
target_link_libraries(_cholespy_core PUBLIC cholmod)

target_compile_definitions(_cholespy_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _cholespy_core LIBRARY DESTINATION sksparse_minimal)

if (WIN32)
  file(GLOB SUITESPARSE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ext/suitesparse-metis-for-windows/lapack_windows/x64/*.dll)
  message(STATUS ${SUITESPARSE_FILES})
  install(FILES ${SUITESPARSE_FILES} DESTINATION sksparse_minimal)
endif()
