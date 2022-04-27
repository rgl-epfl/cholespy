import pytest
from cholesky import CholeskySolverF, MatrixType
import numpy as np
import torch
import drjit
import sksparse.cholmod as cholmod
import scipy.sparse as sp
from test_cholesky import get_coo_arrays

def test_frameworks():

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

    np.random.seed(45)
    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    x_ref = factor.solve_A(b)


    # Test with Numpy
    solver = CholeskySolverF(n_verts, idx[0], idx[1], values, MatrixType.COO)

    x = np.zeros_like(b)
    solver.solve(b, x)
    assert(np.allclose(x, x_ref))

    # Test with PyTorch - CUDA
    solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

    b_torch = torch.tensor(b, device='cuda')
    x_torch = torch.zeros_like(b_torch)
    solver.solve(b_torch, x_torch)
    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    # Test with PyTorch - CPU
    solver = CholeskySolverF(n_verts, torch.tensor(idx[0]), torch.tensor(idx[1]), torch.tensor(values), MatrixType.COO)

    b_torch = torch.tensor(b)
    x_torch = torch.zeros_like(b_torch)
    solver.solve(b_torch, x_torch)
    assert(np.allclose(x_torch.numpy(), x_ref))

    # Test with DrJIT - CUDA
    a = drjit.cuda.TensorXi(idx[0])
    solver = CholeskySolverF(n_verts, drjit.cuda.TensorXi(idx[0]), drjit.cuda.TensorXi(idx[1]), drjit.cuda.TensorXf64(values), MatrixType.COO)

    b_drjit = drjit.cuda.TensorXf(b)
    x_drjit = drjit.zero(drjit.cuda.TensorXf, b.shape)
    solver.solve(b_drjit, x_drjit)
    assert(np.allclose(x_drjit.numpy(), x_ref))

    # Test with DrJIT - CPU
    solver = CholeskySolverF(n_verts, drjit.llvm.TensorXi(idx[0]), drjit.llvm.TensorXi(idx[1]), drjit.llvm.TensorXf64(values), MatrixType.COO)

    b_drjit = drjit.llvm.TensorXf(b)
    x_drjit = drjit.zero(drjit.llvm.TensorXf, b.shape)
    solver.solve(b_drjit, x_drjit)
    assert(np.allclose(x_drjit.numpy(), x_ref))
