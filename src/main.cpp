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
                            uint n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> ii,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> jj,
                            nb::tensor<double, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> x,
                            MatrixType type){

            std::vector<int> indices_a((int *)ii.data(), (int *)ii.data()+ii.shape(0));
            std::vector<int> indices_b((int *)jj.data(), (int *)jj.data()+jj.shape(0));
            std::vector<double> data((double *)x.data(), (double *)x.data()+x.shape(0));
            new (self) Class(n_rows, indices_a, indices_b, data, type);
        })
        .def("solve", [](Class &self, nb::tensor<Float, nb::shape<nb::any, nb::any>, nb::device::cpu, nb::c_contig> b){
            Float *data = self.solve(b.shape(1), (Float *)b.data());
            // Delete 'data' when the 'owner' capsule expires
            nb::capsule owner(data, [](void *p) {
                delete[] (Float *) p;
            });
            size_t shape[2] = {b.shape(0), b.shape(1)};
            return nb::tensor<nb::numpy, Float>(data, 2, shape, owner);
        });
}

NB_MODULE(_cholesky_core, m) {

    nb::enum_<MatrixType>(m, "MatrixType")
        .value("CSC", MatrixType::CSC)
        .value("CSR", MatrixType::CSR)
        .value("COO", MatrixType::COO);

    declare_cholesky<float>(m, "F");
    declare_cholesky<double>(m, "D");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
