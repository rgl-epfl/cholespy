import pytest
from cholesky import CholeskySolverD, CholeskySolverF, MatrixType
import numpy as np
import sksparse.cholmod as cholmod
import scipy.sparse as sp
import torch

def get_coo_arrays(n_verts, faces, lambda_):

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = np.unique(np.stack([np.concatenate([ii, jj]), np.concatenate([jj, ii])], axis=0), axis=1)
    adj_values = np.ones(adj.shape[1], dtype=np.float64) * lambda_

    # Diagonal indices, duplicated as many times as the connectivity of each index
    diag_idx = np.stack((adj[0], adj[0]), axis=0)

    diag = np.stack((np.arange(n_verts), np.arange(n_verts)), axis=0)

    # Build the sparse matrix
    idx = np.concatenate((adj, diag_idx, diag), axis=1)
    values = np.concatenate((-adj_values, adj_values, np.ones(n_verts)))

    return values, idx

def test_cube_float():
    n_verts = 8
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

    solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=n_verts).astype(np.float32)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

def test_cube_double():
    n_verts = 8
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

    solver = CholeskySolverD(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=n_verts).astype(np.float64)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(n_verts, 32)).astype(np.float64)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

def test_ico_float():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_verts = len(v)

    lambda_ = 2.0

    values, idx = get_coo_arrays(n_verts, f, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=n_verts).astype(np.float32)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

def test_ico_double():
    import igl
    import os
    v, f = igl.read_triangle_mesh(os.path.join(os.path.dirname(__file__), "ico.ply"))

    n_verts = len(v)

    lambda_ = 2.0

    values, idx = get_coo_arrays(n_verts, f, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    solver = CholeskySolverD(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

    np.random.seed(45)

    # Test with a single RHS
    b = np.random.random(size=n_verts).astype(np.float64)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))

    # Test with several RHS
    b = np.random.random(size=(n_verts, 32)).astype(np.float64)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), factor.solve_A(b)))
