cmake_minimum_required(VERSION 3.15...3.19)

project(trianglesolver LANGUAGES CXX CUDA)

add_executable(cutrsolve triangle_solve.h triangle_solve.cu main.cpp)
