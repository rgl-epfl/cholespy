from cholesky import LaplacianD, LaplacianF
from drjit.llvm import Int
import numpy as np

def test_cube_float():
	n_verts = 8
	n_faces = 12

	faces = Int([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])

	L = LaplacianF(n_verts, n_faces, faces.data_(), 1.0)

	data = np.array([7, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, 7])
	assert(np.all(L.data() == data))

	rows = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 3, 5, 7, 0, 2, 3, 6, 7, 0, 1, 2, 3, 7, 0, 4, 5, 6, 7, 0, 1, 4, 5, 7, 0, 2, 4, 6, 7, 1, 2, 3, 4, 5, 6, 7])
	assert(np.all(L.rows() == rows))

	col_ptr = np.array([0, 7, 12, 17, 22, 27, 32, 37, 44])
	assert(np.all(L.col_ptr() == col_ptr))


def test_cube_double():
	n_verts = 8
	n_faces = 12

	faces = Int([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])

	L = LaplacianD(n_verts, n_faces, faces.data_(), 1.0)

	data = np.array([7, -1, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, -1, 5, -1, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, 7])
	assert(np.all(L.data() == data))

	rows = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 3, 5, 7, 0, 2, 3, 6, 7, 0, 1, 2, 3, 7, 0, 4, 5, 6, 7, 0, 1, 4, 5, 7, 0, 2, 4, 6, 7, 1, 2, 3, 4, 5, 6, 7])
	assert(np.all(L.rows() == rows))

	col_ptr = np.array([0, 7, 12, 17, 22, 27, 32, 37, 44])
	assert(np.all(L.col_ptr() == col_ptr))
