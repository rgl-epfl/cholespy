# What is this repo?

This project is forked from https://github.com/rgl-epfl/cholespy

Changes made:

- Add support for all solving modes (CHMOD_A, CHMOD_L, CHMOD_Lt, CHMOD_P, etc)
- Remove support for GPU solving (which we didnt need)
- Add python API to emulate sksparse library
- Change CHOLMOD configuration to match sksparse
- Update build to match recommendations from https://nanobind.readthedocs.io/en/latest/building.html


# Installing

## With PyPI (recommended)

```bash
pip install cholespy
```

## From source

```bash
git clone --recursive https://github.com/rgl-epfl/cholespy
pip install .
```

# Example usage

```python
import numpy as np
from cholespy.main import SparseCholesky
from scipy.sparse import csc_matrix

M = np.array([[4, 12, -16],
           [12, 37, -43],
           [-16, -43, 98]], dtype=np.float64)

M = csc_matrix(M)

sparse_cholesky = SparseCholesky(M)

b = np.array([1, 2, 3], dtype=np.float64)

sparse_cholesky.solve_A(b)
```