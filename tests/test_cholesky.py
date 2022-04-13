import pytest
from cholesky import CholeskySolverD, CholeskySolverF
from drjit.llvm import Int, Float32, Float64
import numpy as np
import sksparse.cholmod as cholmod
import scipy.sparse as sp

def build_matrix(n_verts, faces, lambda_):

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = np.concatenate((adj, np.stack((diag_idx, diag_idx), axis=0)), axis=1)
    values = np.concatenate((-adj_values, adj_values))

    L = sp.csc_matrix((values, idx))
    eye = sp.eye(n_verts).tocsc()

    return L * lambda_ + eye

def test_cube_float():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces_numpy = np.array([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])
    faces = Int(faces_numpy)

    L_sp = build_matrix(n_verts, faces_numpy.reshape((-1, 3)), lambda_)
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    # Test with a single RHS
    solver = CholeskySolverF(1, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float32(b).data_()), factor.solve_A(b)))

    # Test with several RHS
    n_rhs = 32
    solver = CholeskySolverF(n_rhs, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_rhs*n_verts)
    sol = solver.solve(Float32(b.flatten()).data_())
    sol_ref = factor.solve_A(b.reshape((n_rhs, n_verts)).T).T.flatten()

    assert(np.allclose(sol, sol_ref))

def test_cube_double():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces_numpy = np.array([0, 1, 3, 0, 3, 2, 2, 3, 7, 2, 7, 6, 4, 5, 7, 4, 7, 6, 0, 1, 5, 0, 5, 4, 1, 5, 7, 1, 7, 3, 0, 4, 6, 0, 6, 2])
    faces = Int(faces_numpy)

    L_sp = build_matrix(n_verts, faces_numpy.reshape((-1, 3)), lambda_)
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    # Test with a single RHS
    solver = CholeskySolverD(1, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float64(b).data_()), factor.solve_A(b)))

    # Test with several RHS
    n_rhs = 3
    solver = CholeskySolverD(n_rhs, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_rhs*n_verts)
    sol = solver.solve(Float64(b.flatten()).data_())
    sol_ref = factor.solve_A(b.reshape((n_rhs, n_verts)).T).T.flatten()

    assert(np.allclose(sol, sol_ref))

def test_ico_float():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    faces_numpy = f.flatten()
    faces = Int(faces_numpy)

    L_sp = build_matrix(n_verts, faces_numpy.reshape((-1, 3)), lambda_)
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    # Test with a single RHS
    solver = CholeskySolverF(1, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float32(b).data_()), factor.solve_A(b)))

    # Test with several RHS
    n_rhs = 32
    solver = CholeskySolverF(n_rhs, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_rhs*n_verts)
    sol = solver.solve(Float32(b.flatten()).data_())
    sol_ref = factor.solve_A(b.reshape((n_rhs, n_verts)).T).T.flatten()

    assert(np.allclose(sol, sol_ref))

def test_ico_double():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    faces_numpy = f.flatten()
    faces = Int(faces_numpy)

    L_sp = build_matrix(n_verts, faces_numpy.reshape((-1, 3)), lambda_)
    factor = cholmod.cholesky(L_sp, ordering_method='amd', mode='simplicial')

    # Test with a single RHS
    solver = CholeskySolverD(1, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_verts)

    assert(np.allclose(solver.solve(Float64(b).data_()), factor.solve_A(b)))

    # Test with several RHS
    n_rhs = 32
    solver = CholeskySolverD(n_rhs, n_verts, n_faces, faces.data_(), lambda_)

    np.random.seed(45)
    b = np.random.random(size=n_rhs*n_verts)
    sol = solver.solve(Float64(b.flatten()).data_())
    sol_ref = factor.solve_A(b.reshape((n_rhs, n_verts)).T).T.flatten()

    assert(np.allclose(sol, sol_ref))
