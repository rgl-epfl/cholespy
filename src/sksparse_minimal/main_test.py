import numpy as np
from sksparse_minimal import SparseCholesky
from scipy.sparse import csc_matrix


def test_results_compared_to_sksparse():
    M = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype=np.float64)

    M = csc_matrix(M)

    sparse_cholesky = SparseCholesky(M)

    b = np.array([1, 2, 3], dtype=np.float64)

    x = sparse_cholesky.solve_A(b)
    np.testing.assert_array_almost_equal(
        x, np.array([28.58333333, -7.66666667, 1.33333333])
    )

    x = sparse_cholesky.solve_Lt(b)
    np.testing.assert_array_almost_equal(x, np.array([52.0, -13.0, 3.0]))

    x = sparse_cholesky.solve_L(b)
    np.testing.assert_array_almost_equal(x, np.array([1.0, -1.0, 12.0]))

    x = sparse_cholesky.solve_D(b)
    np.testing.assert_array_almost_equal(x, np.array([0.25, 2.0, 0.33333333]))

    x = sparse_cholesky.solve_LDLt(b)
    np.testing.assert_array_almost_equal(
        x, np.array([28.58333333, -7.66666667, 1.33333333])
    )

    x = sparse_cholesky.solve_DLt(b)
    np.testing.assert_array_almost_equal(
        x, np.array([0.58333333, 0.33333333, 0.33333333])
    )

    x = sparse_cholesky.P()
    np.testing.assert_array_almost_equal(x, np.array([0, 1, 2]))

    x = sparse_cholesky.apply_P(b)
    np.testing.assert_array_almost_equal(x, np.array([1, 2, 3]))

    x = sparse_cholesky.apply_Pt(b)
    np.testing.assert_array_almost_equal(x, np.array([1, 2, 3]))
