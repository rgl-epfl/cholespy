#include "triangle_solve.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <typename Float>
void declare_solver(py::module &m, std::string typestr) {
    using Class = SparseTriangularSolver<Float>;
    std::string class_name = std::string("SparseTriangularSolver") + typestr;
    py::class_<Class>(m, class_name.c_str())
        .def(py::init([](uint n_rows, uint n_elements, uintptr_t row_ptr, uintptr_t col_ptr, uintptr_t data_ptr, bool lower){
            return new Class(n_rows, n_elements, (uint*) row_ptr, (uint*) col_ptr, (Float*) data_ptr, lower);
        }))
        .def("solve", [](Class &self, uintptr_t b_ptr){return self.solve((Float*) b_ptr);});
}


PYBIND11_MODULE(_cholesky_core, m) {

    declare_solver<float>(m, "F");
    declare_solver<double>(m, "D");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
