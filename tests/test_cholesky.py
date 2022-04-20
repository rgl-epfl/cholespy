import pytest
from cholesky import CholeskySolverD, CholeskySolverF
import numpy as np
import sksparse.cholmod as cholmod
import scipy.sparse as sp

def get_coo_arrays(n_verts, faces, lambda_):

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float) * lambda_

    # Diagonal indices, duplicated as many times as the connectivity of each index
    diag_idx = np.stack((adj[0], adj[0]), axis=0)

    diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

    # Build the sparse matrix
    idx = np.concatenate((adj, diag_idx, diag), axis=1)
    values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

    return values, idx

def test_cube_float():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces = np.array([[0, 1, 3],
                     [0, 3, 2],
                     [2, 3, 7],
                     [2, 7, 6],
                     [4, 5, 7],
                     [4, 7, 6],
                     [0, 1, 5],
                     [0, 5, 4],
                     [1, 5, 7],
                     [1, 7, 3],
                     [0, 4, 6],
                     [0, 6, 2]])

    values, idx = get_coo_arrays(n_verts, faces, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverF(n_verts, idx[0], idx[1], values)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=(n_verts,1)).astype(np.float32)

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(32, n_verts)).astype(np.float32).T

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

def test_cube_double():
    n_verts = 8
    n_faces = 12
    lambda_ = 2.0

    faces = np.array([[0, 1, 3],
                     [0, 3, 2],
                     [2, 3, 7],
                     [2, 7, 6],
                     [4, 5, 7],
                     [4, 7, 6],
                     [0, 1, 5],
                     [0, 5, 4],
                     [1, 5, 7],
                     [1, 7, 3],
                     [0, 4, 6],
                     [0, 6, 2]])

    values, idx = get_coo_arrays(n_verts, faces, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverD(n_verts, idx[0], idx[1], values)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=(n_verts,1)).astype(np.float64)

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(32, n_verts)).astype(np.float64).T

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

def test_ico_float():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    values, idx = get_coo_arrays(n_verts, f, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverF(n_verts, idx[0], idx[1], values)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=(n_verts,1)).astype(np.float32)

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(32, n_verts)).astype(np.float32).T

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

def test_ico_double():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_faces = len(f)
    n_verts = len(v)

    lambda_ = 2.0

    values, idx = get_coo_arrays(n_verts, f, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverD(n_verts, idx[0], idx[1], values)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=(n_verts,1)).astype(np.float64)

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(32, n_verts)).astype(np.float64).T

    assert(np.allclose(solver.solve(b), factor.solve_A(b)))
