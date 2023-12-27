import numpy as np
from scipy import sparse
from sksparse_minimal._cholespy_core import CholeskySolver

CHOLMOD_A = 0
CHOLMOD_LDLt = 1
CHOLMOD_LD = 2
CHOLMOD_DLt = 3
CHOLMOD_L = 4
CHOLMOD_Lt = 5
CHOLMOD_D = 6
CHOLMOD_P = 7
CHOLMOD_Pt = 8


class SparseCholesky:
    def __init__(self, A):
        if not sparse.isspmatrix_csc(A):
            raise ValueError("Expected CSC matrix")

        self._shape = A.shape[0]

        self._solver = CholeskySolver(self._shape, A.indptr, A.indices, A.data)

    def _solve(self, b, method):
        res = np.zeros_like(b, dtype=np.float64)
        self._solver.solve(b.astype(np.float64), res, method)
        return res

    def solve_A(self, b):
        return self._solve(b, CHOLMOD_A)

    def solve_D(self, b):
        return self._solve(b, CHOLMOD_D)

    def solve_DLt(self, b):
        return self._solve(b, CHOLMOD_DLt)

    def solve_L(self, b):
        return self._solve(b, CHOLMOD_L)

    def solve_LD(self, b):
        return self._solve(b, CHOLMOD_LD)

    def solve_LDLt(self, b):
        return self._solve(b, CHOLMOD_LDLt)

    def solve_Lt(self, b):
        return self._solve(b, CHOLMOD_Lt)

    def apply_P(self, b):
        return self._solve(b, CHOLMOD_P)

    def apply_Pt(self, b):
        return self._solve(b, CHOLMOD_Pt)

    def P(self):
        return self.apply_P(np.arange(self._shape))
