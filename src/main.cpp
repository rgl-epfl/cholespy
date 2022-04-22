#include "cholesky_solver.h"
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line);
namespace nb = nanobind;

// void cuda_check_impl(CUresult errval, const char *file, const int line);

template <typename Float>
void declare_cholesky(nb::module_ &m, std::string typestr) {
    using Class = CholeskySolver<Float>;
    std::string class_name = std::string("CholeskySolver") + typestr;
    nb::class_<Class>(m, class_name.c_str())
        // CPU init
        .def("__init__", [](Class *self,
                            uint n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> ii,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> jj,
                            nb::tensor<double, nb::shape<nb::any>, nb::device::cpu, nb::c_contig> x,
                            MatrixType type){

            std::vector<int> indices_a(ii.shape(0));
            std::vector<int> indices_b(jj.shape(0));
            std::vector<double> data(x.shape(0));

            // TODO: Avoid unnecessary copies here
            std::copy((int *)ii.data(), (int *)ii.data()+ii.shape(0), indices_a.begin());
            std::copy((int *)jj.data(), (int *)jj.data()+jj.shape(0), indices_b.begin());
            std::copy((double *)x.data(), (double *)x.data()+x.shape(0), data.begin());

            new (self) Class(n_rows, indices_a, indices_b, data, type, true);
        })
        // GPU init
        .def("__init__", [](Class *self,
                            uint n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cuda, nb::c_contig> ii,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::device::cuda, nb::c_contig> jj,
                            nb::tensor<double, nb::shape<nb::any>, nb::device::cuda, nb::c_contig> x,
                            MatrixType type){

            std::vector<int> indices_a(ii.shape(0));
            std::vector<int> indices_b(jj.shape(0));
            std::vector<double> data(x.shape(0));

            cuda_check(cuMemcpyDtoHAsync(&indices_a[0], (CUdeviceptr) ii.data(), ii.shape(0)*sizeof(int), 0));
            cuda_check(cuMemcpyDtoHAsync(&indices_b[0], (CUdeviceptr) jj.data(), jj.shape(0)*sizeof(int), 0));
            cuda_check(cuMemcpyDtoHAsync(&data[0], (CUdeviceptr) x.data(), x.shape(0)*sizeof(double), 0));

            new (self) Class(n_rows, indices_a, indices_b, data, type, false);
        })
        // CPU solve
        .def("solve", [](Class &self,
                        nb::tensor<Float, nb::device::cpu, nb::c_contig> b,
                        nb::tensor<Float, nb::device::cpu, nb::c_contig> x){
            if (b.ndim() != 1 && b.ndim() != 2)
                throw std::invalid_argument("Expected 1D or 2D tensors as input.");
            if (b.shape(0) != x.shape(0) || (b.ndim() == 2 && b.shape(1) != x.shape(1)))
                throw std::invalid_argument("x and b should have the same dimensions.");

            if (!self.is_cpu())
                throw std::invalid_argument("Input device is CPU but the solver was initialized for CUDA.");

            self.solve_cpu(b.ndim()==2 ? b.shape(1) : 1, (Float *) b.data(), (Float *) x.data());
        })
        // CUDA solve
        .def("solve", [](Class &self,
                        nb::tensor<Float, nb::device::cuda, nb::c_contig> b,
                        nb::tensor<Float, nb::device::cuda, nb::c_contig> x){
            if (b.ndim() != 1 && b.ndim() != 2)
                throw std::invalid_argument("Expected 1D or 2D tensors as input.");
            if (b.shape(0) != x.shape(0) || (b.ndim() == 2 && b.shape(1) != x.shape(1)))
                throw std::invalid_argument("x and b should have the same dimensions.");

            if (self.is_cpu())
                throw std::invalid_argument("Input device is CUDA but the solver was initialized for CPU.");

            self.solve_cuda(b.ndim()==2 ? b.shape(1) : 1, (CUdeviceptr) b.data(), (CUdeviceptr) x.data());
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
