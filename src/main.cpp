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
                            uint n_rhs,
                            uint n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> coo_i,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> coo_j,
                            nb::tensor<double, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> coo_x){

            std::vector<int> ii((int *)coo_i.data(), (int *)coo_i.data()+coo_i.shape(0));
            std::vector<int> jj((int *)coo_j.data(), (int *)coo_j.data()+coo_j.shape(0));
            std::vector<double> data((double *)coo_x.data(), (double *)coo_x.data()+coo_x.shape(0));
            new (self) Class(n_rhs, n_rows, ii, jj, data);
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
