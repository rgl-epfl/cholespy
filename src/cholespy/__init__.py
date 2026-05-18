from importlib import import_module
import_module('cholespy._cholespy_core')
del import_module

# _cholespy_core registers CholeskySolverF, CholeskySolverD, and MatrixType
# directly on this module.  Grab references before we shadow the names.
_CholeskySolverF = CholeskySolverF  # noqa: F821
_CholeskySolverD = CholeskySolverD  # noqa: F821


def _jax_block_input(b):
    """If b is a JAX array, block until its GPU computation is complete.

    JAX dispatches kernels asynchronously on a non-blocking per-thread stream
    that does not synchronize with cholespy's stream 0.  Without this call,
    cholespy may read stale (e.g. zero) data from b.
    """
    try:
        import jax
        if isinstance(b, jax.Array):
            b.block_until_ready()
    except ImportError:
        pass


class CholeskySolverF(_CholeskySolverF):
    def solve(self, b, x):
        _jax_block_input(b)
        super().solve(b, x)


class CholeskySolverD(_CholeskySolverD):
    def solve(self, b, x):
        _jax_block_input(b)
        super().solve(b, x)
