import pytest
from cholesky import CholeskySolverD, CholeskySolverF
from cholesky import LaplacianD, LaplacianF
from drjit.llvm import Int, Float32, Float64
import numpy as np
import sksparse.cholmod as cholmod
import scipy.sparse as sp

def test_cube_float():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces_numpy = np.array([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])
    faces = Int(faces_numpy)

    L = LaplacianF(n_verts, n_faces, faces.data_(), lambda_)
    L_sp = sp.csc_matrix((L.data(), L.rows(), L.col_ptr()))
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverF(n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float32(b).data_()), factor.solve_A(b)))

def test_cube_double():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces_numpy = np.array([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])
    faces = Int(faces_numpy)

    L = LaplacianD(n_verts, n_faces, faces.data_(), lambda_)
    L_sp = sp.csc_matrix((L.data(), L.rows(), L.col_ptr()))
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverD(n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float64(b).data_()), factor.solve_A(b)))

def test_ico_float():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    faces_numpy = f.flatten()
    faces = Int(faces_numpy)

    L = LaplacianF(n_verts, n_faces, faces.data_(), lambda_)
    L_sp = sp.csc_matrix((L.data(), L.rows(), L.col_ptr()))
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverF(n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float32(b).data_()), factor.solve_A(b)))

def test_ico_double():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    faces_numpy = f.flatten()
    faces = Int(faces_numpy)

    L = LaplacianD(n_verts, n_faces, faces.data_(), lambda_)
    L_sp = sp.csc_matrix((L.data(), L.rows(), L.col_ptr()))
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverD(n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float64(b).data_()), factor.solve_A(b)))
