#include "cholesky_solver.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

template <typename Float>
void declare_cholesky(nb::module_ &m, std::string typestr) {
    using Class = CholeskySolver<Float>;
    std::string class_name = std::string("CholeskySolver") + typestr;
    nb::class_<Class>(m, class_name.c_str())
        .def("__init__", [](Class *self, uint nrhs, uint n_verts, uint n_faces, uintptr_t faces, double lambda){
            new (self) Class(nrhs, n_verts, n_faces, (uint *)faces, lambda);
        })
        .def("solve", [](Class &self, uintptr_t b_ptr){return self.solve((Float*) b_ptr);});
}

NB_MODULE(_cholesky_core, m) {

    declare_cholesky<float>(m, "F");
    declare_cholesky<double>(m, "D");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
