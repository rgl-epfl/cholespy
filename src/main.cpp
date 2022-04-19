#include "cholesky_solver.h"
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

template <typename Float>
void declare_cholesky(nb::module_ &m, std::string typestr) {
    using Class = CholeskySolver<Float>;
    std::string class_name = std::string("CholeskySolver") + typestr;
    nb::class_<Class>(m, class_name.c_str())
        .def("__init__", [](Class *self,
                            uint nrhs,
                            uint n_verts,
                            uint n_faces,
                            nb::tensor<int32_t, nb::shape<nb::any, 3>, nb::device::cpu, nb::c_contig> faces,
                            double lambda){
            new (self) Class(nrhs, n_verts, n_faces, (int *)faces.data(), lambda);
        })
        .def("solve", [](Class &self, nb::tensor<Float, nb::shape<nb::any, nb::any>, nb::device::cpu, nb::c_contig> b){
            Float *data = self.solve((Float *)b.data());
            // Delete 'data' when the 'owner' capsule expires
            nb::capsule owner(data, [](void *p) {
                delete[] (Float *) p;
            });
            size_t shape[2] = {b.shape(0), b.shape(1)};
            return nb::tensor<nb::numpy, Float>(data, 2, shape, owner);
        });
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
