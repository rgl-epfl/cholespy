#include "cholesky_solver.h"
#include "docstr.h"
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

void cuda_check_impl(CUresult errval, const char *file, const int line);
namespace nb = nanobind;

// void cuda_check_impl(CUresult errval, const char *file, const int line);

template <typename Float>
void declare_cholesky(nb::module_ &m, const std::string &typestr, const char *docstr) {
    using Class = CholeskySolver<Float>;
    std::string class_name = std::string("CholeskySolver") + typestr;
    nb::class_<Class>(m, class_name.c_str(), docstr)
        .def("__init__", [](Class *self,
                            uint32_t n_rows,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> ii,
                            nb::tensor<int32_t, nb::shape<nb::any>, nb::c_contig> jj,
                            nb::tensor<double, nb::shape<nb::any>, nb::c_contig> x,
                            MatrixType type) {

            if (type == MatrixType::COO){
                if (ii.shape(0) != jj.shape(0))
                    throw std::invalid_argument("Sparse COO matrix: the two index arrays should have the same size.");
                if (ii.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse COO matrix: the index and data arrays should have the same size.");
            } else if (type == MatrixType::CSR) {
                if (jj.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse CSR matrix: the column index and data arrays should have the same size.");
                if (ii.shape(0) != n_rows+1)
                    throw std::invalid_argument("Sparse CSR matrix: Invalid size for row pointer array.");
            } else {
                if (jj.shape(0) != x.shape(0))
                    throw std::invalid_argument("Sparse CSC matrix: the row index and data arrays should have the same size.");
                if (ii.shape(0) != n_rows+1)
                    throw std::invalid_argument("Sparse CSC matrix: Invalid size for column pointer array.");
            }
            if (ii.device_type() != jj.device_type() || ii.device_type() != x.device_type())
                throw std::invalid_argument("All input tensors should be on the same device!");

            if (ii.device_type() == nb::device::cuda::value) {
                // GPU init

                // Initialize CUDA and load the kernels if not already done
                init_cuda();

                scoped_set_context guard(cu_context);

                int *indices_a = (int *) malloc(ii.shape(0)*sizeof(int));
                int *indices_b = (int *) malloc(jj.shape(0)*sizeof(int));
                double *data = (double *) malloc(x.shape(0)*sizeof(double));

                cuda_check(cuMemcpyAsync((CUdeviceptr) indices_a, (CUdeviceptr) ii.data(), ii.shape(0)*sizeof(int), 0));
                cuda_check(cuMemcpyAsync((CUdeviceptr) indices_b, (CUdeviceptr) jj.data(), jj.shape(0)*sizeof(int), 0));
                cuda_check(cuMemcpyAsync((CUdeviceptr) data, (CUdeviceptr) x.data(), x.shape(0)*sizeof(double), 0));

                new (self) Class(n_rows, x.shape(0), indices_a, indices_b, data, type, false);

                free(indices_a);
                free(indices_b);
                free(data);
            } else if (ii.device_type() == nb::device::cpu::value) {
                // CPU init
                new (self) Class(n_rows, x.shape(0), (int *) ii.data(), (int *) jj.data(), (double *) x.data(), type, true);
            } else
                throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
        },
        nb::arg("n_rows"),
        nb::arg("ii"),
        nb::arg("jj"),
        nb::arg("x"),
        nb::arg("type"),
        doc_constructor)
        .def("solve", [](Class &self,
                        nb::tensor<Float, nb::c_contig> b,
                        nb::tensor<Float, nb::c_contig> x){
            if (b.ndim() != 1 && b.ndim() != 2)
                throw std::invalid_argument("Expected 1D or 2D tensors as input.");
            if (b.shape(0) != x.shape(0) || (b.ndim() == 2 && b.shape(1) != x.shape(1)))
                throw std::invalid_argument("x and b should have the same dimensions.");
            if (b.device_type() != x.device_type())
                throw std::invalid_argument("x and b should be on the same device.");

            // CPU solve
            if (b.device_type() == nb::device::cpu::value) {
                if (!self.is_cpu())
                    throw std::invalid_argument("Input device is CPU but the solver was initialized for CUDA.");

                self.solve_cpu(b.ndim()==2 ? b.shape(1) : 1, (Float *) b.data(), (Float *) x.data());
            }
            // CUDA solve
            else if (b.device_type() == nb::device::cuda::value) {
                if (self.is_cpu())
                    throw std::invalid_argument("Input device is CUDA but the solver was initialized for CPU.");

                scoped_set_context guard(cu_context);
                self.solve_cuda(b.ndim()==2 ? b.shape(1) : 1, (CUdeviceptr) b.data(), (CUdeviceptr) x.data());
            }
            else
                throw std::invalid_argument("Unsupported input device! Only CPU and CUDA arrays are supported.");
        },
        nb::arg("b").noconvert(),
        nb::arg("x").noconvert(),
        doc_solve);
}

NB_MODULE(_cholespy_core, m_) {
    (void) m_;

    nb::module_ m = nb::module_::import_("cholespy");

    nb::enum_<MatrixType>(m, "MatrixType", doc_matrix_type)
        .value("CSC", MatrixType::CSC)
        .value("CSR", MatrixType::CSR)
        .value("COO", MatrixType::COO);

    declare_cholesky<float>(m, "F", doc_cholesky_f);
    declare_cholesky<double>(m, "D", doc_cholesky_d);

    // Custom object to gracefully shutdown CUDA when unloading the module
    nb::detail::keep_alive(m.ptr(),
                           (void *) 1, // Unused payload
                           [](void *p) noexcept { shutdown_cuda(); });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
