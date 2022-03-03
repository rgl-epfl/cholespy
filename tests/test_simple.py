import drjit as dr
from drjit.llvm import Float32, Float64, Int
from cholesky import SparseTriangularSolverD, SparseTriangularSolverF
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular

def test_solver_float():
    data = np.arange(1.0, 19.0, dtype=np.float32)
    rows = np.array([0, 1, 2, 3, 5, 7, 9, 11, 14, 18])
    cols = np.array([0, 1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 4, 7, 2, 3, 4, 8])

    M = sp.csr_matrix((data, cols, rows))
    b = np.arange(1.0, 10.0, dtype=np.float32)

    solver = SparseTriangularSolverF(M.shape[0], len(data), Int(rows).data_(), Int(cols).data_(), Float32(data).data_(), True)

    assert(np.allclose(solver.solve(Float32(b).data_()), spsolve_triangular(M, b, lower=True)))

    Mt = sp.csr_matrix(M.T)
    solver = SparseTriangularSolverF(Mt.shape[0], len(data), Int(Mt.indptr).data_(), Int(Mt.indices).data_(), Float32(Mt.data).data_(), False)

    assert(np.allclose(solver.solve(Float32(b).data_()), spsolve_triangular(Mt, b, lower=False)))

def test_solver_double():
    data = np.arange(1.0, 19.0, dtype=np.float64)
    rows = np.array([0, 1, 2, 3, 5, 7, 9, 11, 14, 18])
    cols = np.array([0, 1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 4, 7, 2, 3, 4, 8])

    M = sp.csr_matrix((data, cols, rows))
    b = np.arange(1.0, 10.0, dtype=np.float64)

    solver = SparseTriangularSolverD(M.shape[0], len(data), Int(rows).data_(), Int(cols).data_(), Float64(data).data_(), True)

    assert(np.allclose(solver.solve(Float64(b).data_()), spsolve_triangular(M, b, lower=True)))

    Mt = sp.csr_matrix(M.T)
    solver = SparseTriangularSolverD(Mt.shape[0], len(data), Int(Mt.indptr).data_(), Int(Mt.indices).data_(), Float64(Mt.data).data_(), False)

    assert(np.allclose(solver.solve(Float64(b).data_()), spsolve_triangular(Mt, b, lower=False)))
