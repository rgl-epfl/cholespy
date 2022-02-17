cmake_minimum_required(VERSION 3.15...3.19)

project(cholesky LANGUAGES CXX CUDA VERSION "0.1.0")

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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)
set(CMAKE_CUDA_FLAGS "")
add_library(cholesky_cuda STATIC src/triangle_solve.h src/triangle_solve.cu)

set_target_properties(cholesky_cuda PROPERTIES COMPILE_OPTIONS "")
set_property(TARGET cholesky_cuda PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET cholesky_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(cholesky_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:>)
target_link_libraries(cholesky_cuda ${CUDA_LIBRARIES})
set_property(TARGET cholesky_cuda PROPERTY CXX_STANDARD "17")

pybind11_add_module(_cholesky_core MODULE src/triangle_solve.h src/main.cpp)
target_link_libraries(_cholesky_core PUBLIC cholesky_cuda)

#target_include_directories(_cholesky_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
target_compile_definitions(_cholesky_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS _cholesky_core DESTINATION .)
