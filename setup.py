import sys

try:
    from skbuild import setup
    import nanobind
except ImportError:
    print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
          "install .'. If you wish to run the setup script directly, you must "
          "first install the build dependencies listed in pyproject.toml!",
          file=sys.stderr)
    raise

from setuptools import find_packages

setup(
    name="cholesky",
    version="0.1.0",
    description="A triangular backsubstitution solver",
    author="Baptiste Nicolet",
    license="BSD",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/cholesky",
    include_package_data=True,
    python_requires=">=3.8",
)
