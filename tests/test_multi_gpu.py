import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from cholespy import CholeskySolverD, CholeskySolverF, MatrixType
from utils import get_coo_arrays, get_icosphere

torch = pytest.importorskip("torch")
N_GPUS = torch.cuda.device_count()


def build(n_verts, faces, device, Solver):
    values, idx = get_coo_arrays(n_verts, faces, 2.0)
    ii = torch.tensor(idx[0], dtype=torch.int32, device=device)
    jj = torch.tensor(idx[1], dtype=torch.int32, device=device)
    x = torch.tensor(values, dtype=torch.float64, device=device)
    return Solver(n_verts, ii, jj, x, MatrixType.COO), sp.csc_matrix((values, idx))


@pytest.mark.skipif(N_GPUS == 0, reason="no GPU")
@pytest.mark.parametrize("variant", ["float", "double"])
@pytest.mark.parametrize("index", range(N_GPUS))
def test_solver_runs_on_the_input_device(index, variant):
    Solver, dtype = (CholeskySolverF, torch.float32) if variant == "float" else (CholeskySolverD, torch.float64)
    device = torch.device(f"cuda:{index}")
    n_verts, faces = get_icosphere(4)
    solver, A = build(n_verts, faces, device, Solver)
    b = torch.rand(n_verts, 16, dtype=dtype, device=device)
    x = torch.zeros_like(b)
    solver.solve(b, x)
    torch.cuda.synchronize(device)
    expected = spsolve(A, b.cpu().double().numpy())
    assert np.allclose(x.cpu().numpy(), expected, atol=1e-4 if variant == "float" else 1e-10)


@pytest.mark.skipif(N_GPUS < 2, reason="needs two GPUs")
def test_solvers_on_different_devices_coexist():
    n_verts, faces = get_icosphere(3)
    solvers = [build(n_verts, faces, torch.device(f"cuda:{i}"), CholeskySolverF)[0] for i in range(N_GPUS)]
    for i, s in enumerate(solvers):
        b = torch.ones(n_verts, 3, device=f"cuda:{i}")
        x = torch.zeros_like(b)
        s.solve(b, x)
        torch.cuda.synchronize(i)
        assert torch.isfinite(x).all() and x.abs().sum() > 0


@pytest.mark.skipif(N_GPUS < 2, reason="needs two GPUs")
def test_solve_on_the_wrong_device_raises():
    n_verts, faces = get_icosphere(2)
    solver, _ = build(n_verts, faces, torch.device("cuda:1"), CholeskySolverF)
    b = torch.ones(n_verts, 3, device="cuda:0")
    with pytest.raises(ValueError, match="cuda:1"):
        solver.solve(b, torch.zeros_like(b))


@pytest.mark.skipif(N_GPUS < 2, reason="needs two GPUs")
def test_inputs_on_mixed_devices_raise():
    n_verts, faces = get_icosphere(2)
    values, idx = get_coo_arrays(n_verts, faces, 2.0)
    with pytest.raises(ValueError, match="same CUDA device"):
        CholeskySolverF(n_verts, torch.tensor(idx[0], dtype=torch.int32, device="cuda:0"),
                        torch.tensor(idx[1], dtype=torch.int32, device="cuda:1"),
                        torch.tensor(values, device="cuda:0"), MatrixType.COO)
