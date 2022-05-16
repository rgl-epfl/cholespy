const char *doc_constructor= R"(
Precompute a sparse Cholesky factor for the given input matrix in COO, CSR or CSC format.

Parameters
----------

- n_rows - The number of rows in the (sparse) matrix.
- ii - The first array of indices in the sparse matrix representation. If
       type is COO, then this is the array of row indices. If it is CSC (resp.
       CSR), then it is the array of column (resp. row) indices, such that row
       (resp. column) indices for column (resp. row) k are stored in
       jj[ii[k]:ii[k+1]] and the corresponding entries are in x[ii[k]:ii[k+1]].
- jj - The second array of indices in the sparse matrix representation. If
       type is COO, then this is the array of column indices. If it is CSC
       (resp. CSR), then it is the array of row (resp. column) indices.
- x - The array of nonzero entries.
- type - The matrix representation type, of type MatrixType. Available types
         are MatrixType.COO, MatrixType.CSC and MatrixType.CSR.
)";

const char *doc_matrix_type =
    "This enumeration is used to distinguish between different sparse matrix "
    "formats when constructing CholeskySolverD/CholeskySolverF instances";

const char *doc_cholesky_f =
    "Single-precision solver implementation";

const char *doc_cholesky_d =
    "Double-precision solver implementation";

const char *doc_solve = R"(

Solve the linear system for a given set of right-hand sides.

Parameters
----------

- `b` - Right-hand side of the equation to solve. Can be a vector or a matrix.
        If it is a matrix, it must be of shape `(n_rows, n_rhs)`. It must be on the
        same device as the tensors passed to the solver constructor. If using CUDA
        arrays, then the maximum supported value for `n_rhs` is `128`.
- `x` - Placeholder for the solution. It must be on the same device and have the
          same shape as `b`.

`x` and `b` **must** have the same dtype as the solver used, i.e. `float32` for
`CholeskySolverF` or `float64` for `CholeskySolver64`. Since `x` is modified in
place, implicit type conversion is not supported.
)";
