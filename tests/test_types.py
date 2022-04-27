import pytest
from cholesky import CholeskySolverF, MatrixType
import numpy as np
import sksparse.cholmod as cholmod
import scipy.sparse as sp
import torch
from test_cholesky import get_coo_arrays

def test_matrices():
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
    L_csr = sp.csr_matrix((values, idx))
    factor = cholmod.cholesky(L_csc, ordering_method='amd', mode='simplicial')

    np.random.seed(45)
    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)

    x_ref = factor.solve_A(b)

    # Test with COO input
    solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    # Test with CSR input
    solver = CholeskySolverF(n_verts, torch.tensor(L_csr.indptr, device='cuda'), torch.tensor(L_csr.indices, device='cuda'), torch.tensor(L_csr.data, device='cuda'), MatrixType.CSR)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    # Test with CSC input
    solver = CholeskySolverF(n_verts, torch.tensor(L_csc.indptr, device='cuda'), torch.tensor(L_csc.indices, device='cuda'), torch.tensor(L_csc.data, device='cuda'), MatrixType.CSC)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))
