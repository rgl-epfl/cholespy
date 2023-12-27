# sksparse_minimal

This project is forked from https://github.com/rgl-epfl/cholespy

Changes made:

- Add support for all solving modes (CHOLMOD_A, CHOLMOD_L, CHOLMOD_Lt, CHOLMOD_P, etc)
- Remove support for GPU solving (because I didn't want to bother implementing it for all modes and I didn't need it)
- Emulate sksparse API for ease of use
- Change CHOLMOD configuration to match sksparse
- Update build to match recommendations from https://nanobind.readthedocs.io/en/latest/building.html

# Installing

## With PyPI (recommended)

```bash
pip install sksparse_minimal
```

## From source

```bash
git clone --recursive https://github.com/tansey-lab/sksparse_minimal.git
pip install .
```

# Example usage

```python
import numpy as np
from sksparse_minimal import SparseCholesky
from scipy.sparse import csc_matrix

M = np.array([[4, 12, -16],
           [12, 37, -43],
           [-16, -43, 98]], dtype=np.float64)

M = csc_matrix(M)

sparse_cholesky = SparseCholesky(M)

b = np.array([1, 2, 3], dtype=np.float64)

sparse_cholesky.solve_A(b)
```