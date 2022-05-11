import pytest
from cholespy import CholeskySolverD, CholeskySolverF, MatrixType
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from utils import get_coo_arrays, get_icosphere, get_cube

RHS = [1, 4, 16, 64, 128, 256]

@pytest.mark.parametrize("n_verts, faces", [get_cube(), get_icosphere(3), get_icosphere(5)])
@pytest.mark.parametrize("variant", ["float", "double"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_solver(device, variant, n_verts, faces):
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch was not found!")

    if device=='cuda' and torch.cuda.device_count() == 0:
        pytest.skip("No GPU found.")

    np.random.seed(45)
    CholeskySolver = CholeskySolverF if variant == "float" else CholeskySolverD
    dtype = np.float32 if variant == "float" else np.float64

    lambda_ = 2.0
    values, idx = get_coo_arrays(n_verts, faces, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    b = np.random.random(size=(n_verts, max(RHS))).astype(dtype)
    x= spsolve(L_csc, b)
    solver = CholeskySolver(n_verts, torch.tensor(idx[0], device=device), torch.tensor(idx[1], device=device), torch.tensor(values, device=device), MatrixType.COO)

    # Test with different RHS
    for n_rhs in RHS:
        b_crop = b[:, :n_rhs]
        b_torch = torch.tensor(b_crop, device=device)
        x_torch = torch.zeros_like(b_torch)
        try:
            solver.solve(b_torch, x_torch)
            assert(np.allclose(x_torch.cpu().numpy(), x[:, :n_rhs]))
        except ValueError as e:
            assert n_rhs > 128
            assert e.__str__() == "The number of RHS should be less than 128."

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_matrices(device):
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("PyTorch was not found!")

    if device=='cuda' and torch.cuda.device_count() == 0:
        pytest.skip("No GPU found.")

    np.random.seed(45)
    lambda_ = 2.0
    n_verts, faces = get_cube()
    values, idx = get_coo_arrays(n_verts, faces, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    L_csr = sp.csr_matrix((values, idx))

    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    b_torch = torch.tensor(b, device=device)
    x_torch = torch.zeros_like(b_torch)

    x_ref = spsolve(L_csc, b)

    # Test with COO input
    solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device=device), torch.tensor(idx[1], device=device), torch.tensor(values, device=device), MatrixType.COO)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    # Test with CSR input
    solver = CholeskySolverF(n_verts, torch.tensor(L_csr.indptr, device=device), torch.tensor(L_csr.indices, device=device), torch.tensor(L_csr.data, device=device), MatrixType.CSR)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    # Test with CSC input
    solver = CholeskySolverF(n_verts, torch.tensor(L_csc.indptr, device=device), torch.tensor(L_csc.indices, device=device), torch.tensor(L_csc.data, device=device), MatrixType.CSC)
    solver.solve(b_torch, x_torch)

    assert(np.allclose(x_torch.cpu().numpy(), x_ref))

@pytest.mark.parametrize("framework", ["numpy", "torch", "tensorflow", "jax", "cupy", "drjit"])
def test_frameworks(framework):
    import importlib
    # Disable tests if the module is not installed
    try:
        importlib.import_module(framework)
    except ModuleNotFoundError:
        pytest.skip(f"Module {framework} is not installed.")

    np.random.seed(45)
    lambda_ = 2.0

    n_verts, faces = get_cube()
    values, idx = get_coo_arrays(n_verts, faces, lambda_)

    L_csc = sp.csc_matrix((values, idx))
    b = np.random.random(size=(n_verts, 32)).astype(np.float32)
    x_ref = spsolve(L_csc, b)

    # Test with Numpy
    if framework == "numpy":
        solver = CholeskySolverF(n_verts, idx[0], idx[1], values, MatrixType.COO)

        x = np.zeros_like(b)
        solver.solve(b, x)
        assert(np.allclose(x, x_ref))

    elif framework == "torch":
        import torch
        # Test with PyTorch - CPU
        solver = CholeskySolverF(n_verts, torch.tensor(idx[0]), torch.tensor(idx[1]), torch.tensor(values), MatrixType.COO)

        b_torch = torch.tensor(b)
        x_torch = torch.zeros_like(b_torch)
        solver.solve(b_torch, x_torch)
        assert(np.allclose(x_torch.numpy(), x_ref))

        # Test with PyTorch - CUDA
        if torch.cuda.device_count() > 0:
            solver = CholeskySolverF(n_verts, torch.tensor(idx[0], device='cuda'), torch.tensor(idx[1], device='cuda'), torch.tensor(values, device='cuda'), MatrixType.COO)

            b_torch = torch.tensor(b, device='cuda')
            x_torch = torch.zeros_like(b_torch)
            solver.solve(b_torch, x_torch)
            assert(np.allclose(x_torch.cpu().numpy(), x_ref))

    elif framework == "tensorflow":
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        # Prevent TF from allocating all GPU mrmory for itself
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Test with TensorFlow - CPU
        with tf.device('/device:cpu:0'):
            solver = CholeskySolverF(n_verts, tf.convert_to_tensor(idx[0]), tf.convert_to_tensor(idx[1]), tf.convert_to_tensor(values), MatrixType.COO)

            b_tf = tf.convert_to_tensor(b)
            x_tf = tf.zeros_like(b_tf)
            solver.solve(b_tf, x_tf)
            assert(np.allclose(x_tf.numpy(), x_ref))

        if len(gpus) > 0:
            # Test with TensorFlow - CUDA
            with tf.device('/device:gpu:0'):
                solver = CholeskySolverF(n_verts, tf.convert_to_tensor(idx[0]), tf.convert_to_tensor(idx[1]), tf.convert_to_tensor(values), MatrixType.COO)

                b_tf = tf.convert_to_tensor(b)
                x_tf = tf.zeros_like(b_tf)
                solver.solve(b_tf, x_tf)
                assert(np.allclose(x_tf.numpy(), x_ref))

    elif framework == "jax":
        import os
        # Prevent JAX from allocating all GPU mrmory for itself
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
        import jax
        with jax.experimental.enable_x64():
            # Test with JAX
            solver = CholeskySolverF(n_verts, jax.numpy.array(idx[0]), jax.numpy.array(idx[1]), jax.numpy.array(values, dtype=np.float64), MatrixType.COO)

            b_jax= jax.numpy.array(b)
            x_jax = jax.numpy.zeros_like(b_jax)
            solver.solve(b_jax, x_jax)
            assert(np.allclose(x_jax, x_ref))

    elif framework == "cupy":
        import cupy as cp
        # Test with CuPy
        solver = CholeskySolverF(n_verts, cp.array(idx[0], dtype=cp.int32), cp.array(idx[1], dtype=cp.int32), cp.array(values, dtype=cp.float64), MatrixType.COO)

        b_cp = cp.array(b)
        x_cp = cp.zeros_like(b_cp)
        solver.solve(b_cp, x_cp)
        assert(np.allclose(cp.asnumpy(x_cp), x_ref))

    elif framework == "drjit":
        import drjit
        # Test with DrJIT - CUDA
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
