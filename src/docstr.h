const char *doc_constructor= R"(
Precompute a sparse Cholesky factor for the given input sparse matrix in CSC format.

Parameters
----------

- n_rows - The number of rows in the (sparse) matrix.
- ii - The first array of indices in the sparse matrix representation.
       The array of column (resp. row) indices, such that row
       (resp. column) indices for column (resp. row) k are stored in
       jj[ii[k]:ii[k+1]] and the corresponding entries are in x[ii[k]:ii[k+1]].
- jj - The second array of indices in the sparse matrix representation.
       The array of row (resp. column) indices.
- x - The array of nonzero entries.
)";

const char *doc_cholesky_d =
    "Double-precision solver implementation";

const char *doc_solve = R"(

Solve the linear system for a given set of right-hand sides.

Parameters
----------

- `b` - Right-hand side of the equation to solve. Can be a vector or a matrix.
        If it is a matrix, it must be of shape `(n_rows, n_rhs)`. It must be on the
        same device as the tensors passed to the solver constructor.
- `x` - Placeholder for the solution. It must be on the same device and have the
          same shape as `b`.
)";
