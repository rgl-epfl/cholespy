import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
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
    python_requires=">=3.6",
)
