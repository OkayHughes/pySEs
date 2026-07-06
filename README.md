# Overview
[![numpy](https://github.com/OkayHughes/pySEs/actions/workflows/numpy.yml/badge.svg)](https://github.com/OkayHughes/pySEs/actions/workflows/numpy.yml)
[![jax](https://github.com/OkayHughes/pySEs/actions/workflows/jax.yml/badge.svg)](https://github.com/OkayHughes/pySEs/actions/workflows/jax.yml)
[![torch](https://github.com/OkayHughes/pySEs/actions/workflows/torch.yml/badge.svg)](https://github.com/OkayHughes/pySEs/actions/workflows/torch.yml)

The purpose of this project is to create a highly readable, well documented, well tested atmospheric dynamical core with support for variable resolution meshes, automatic differentiation, and GPU support. It is designed to be highly backward-compatible with NCAR CAM-SE and HOMME numerics.
This project prioritizes code readability and maintainability, in the sense that code that runs 10% slower but is much easier for a second-year graduate student to understand and modify is better than its inaccessible optimized counterpart.
We want to minimize external dependencies and, insofar as it is possible, create a codebase that is entirely written in python.
Given the constraints of these design decisions, it is unlikely that the resulting dynamical core will scale to hundreds or thousands of nodes on an HPC computing system. This aligns with our stated goal of making atmospheric modeling accessible.


# Top-level list of features that make pySEs stand out
* GPU support and automatic differentiation via JAX or PyTorch.
* Battle-tested high-order numerical discretization with discrete conservation properties (mathematically identical to NCAR CAM)
    * Dry-mass coordinate allows easy coupling
* Numerical stabilization mechanisms are near-identical to dynamical cores in CAM/EAM
* Highly validated non-hydrostatic dynamics that are near-identical to the Simple Cloud Resolving E3SM Atmosphere Model (SCREAM)

# Quickstart with `uv`

## Common steps
* Run `git clone git@github.com:OkayHughes/pyses.git` and navigate to the `pyses` directory.
* Install `uv`, e.g. `pip install uv` in your python environment. `uv` is a 
modern environment manager for Python that we use for development and testing.

## MPI-dependent steps
pySEs gates MPI behind two install *extras*: `mpi-local` bundles an MPICH from
PyPI (no system MPI needed), and `mpi-system` installs only `mpi4py`, built
against your system MPI.

If you don't have MPI installed, run:
* `uv sync --extra mpi-local --group dev`

If you have a system MPI installed (e.g. on HPC systems), run:
* `which mpicc`
* Create a `.env` file containing `export MPICC=${MPICC_LOCATION}`
* `uv sync --env-file .env --extra mpi-system --group dev`
* Note: modern `mpi4py` wheels bundle their own MPICH, so to actually link the
  system MPI force a source build, e.g.
  `UV_NO_BINARY_PACKAGE=mpi4py uv sync --env-file .env --extra mpi-system --group dev`.

(Installing from PyPI rather than this checkout uses the same extras:
`pip install "pyses[mpi-local]"`, or
`MPICC=$(which mpicc) pip install --no-binary mpi4py "pyses[mpi-system]"`.)

## Run tests
* Navigate to the `tests` directory and run `bash run_test.sh`. These should catch if there are problems
with CPU configurations of `pyses`, and test whether there are issues with your MPI environment.


## How to enable the correct backend/device

Since pySEs can switch on-the-fly between CPU and GPU, and switch whether array operations are provided by Numpy, JAX, or PyTorch. 

When running from the command line, change what backend is used by exporting the environment variable `PYSES_BACKEND=[jax/torch/numpy]`. 
To use GPU, export `PYSES_USE_CPU=0`. 
pySEs defaultly uses CPU, as users must be careful to install and isolate the correct GPU-enabled JAX/PyTorch python environment that works with their system.
pySEs has only been numerically verified with 64 bit floating point arithmetic. 

If you are using an interactive environment, then run, e.g.,
```
from pyses._config import _reset_backend, get_backend
import os
os.environ["PYSES_BACKEND"] = "jax"
_reset_backend()
_be = get_backend()
bnp = _be.np # use this instead of importing numpy directly to defaultly work with device arrays.
```



# Accelerator support
Most scientific code can be easily written in a (nearly) [purely functional](https://en.wikipedia.org/wiki/Pure_function) programming style. Consequently, this means that the codebase can be written to satisfy the requirements of the [Jax](https://github.com/jax-ml/jax) library's just-in-time compilation and automatic differentiation, while retaining the ability to run with array/tensor operations provided by [PyTorch](https://pytorch.org/) or [Numpy](https://numpy.org/). Due to Google's history of abruptly discontinuing widely used software products, we have chosen to future proof this code base by ensuring that GPU parallelism (and ideally automatic differentiation) can be sourced from any library that provides an array implementation that resembles `Numpy.ndarray`, along with a list of array operations that is enumerated in the documentation.

# Jax support
The Jax configuration of the code is significantly more performant than either the Numpy or Torch configurations. 
This is because the allowable control flow constructs of `jax.jit` are significantly less constraining than,
e.g., `numba` or `torch.compile`. Consequently, the programming style of this project treats the idiosyncracies of the
Jax functional programming model as authorative. This means that Numpy and Torch performance suffers, as
Jax disallows assignment of slices of arrays after initial assignment, so code like 
```
x = np.eye((3, 3))
x[:, 0] = x[:, 1]
```
is entirely disallowed outside of initialization. Instead, this would be written
```
x = np.eye((3, 3))
x = np.stack((x[:, 1], x[:, 1:]), axis=1)
```
which appears almost deliberately tailored to force Numpy/Torch view semantics
to copy the data in `x`. 

# PyTorch support
PyTorch can also provide automatic differentiation capabilities and GPU parallelism. The recent switch to `torch.compile` allows 
JAX-style code to be JIT-compiled with PyTorch, once you write a shim library that reconstructs a `numpy`-like interface from the
tensor operations that `torch` provides. 

# Policy on intellectual property

The view of the maintainer is that since the training data 
used to train LLMs was obtained without the consent of the people who made it, 
LLMs trained on this data are structurally incapable
of determining when they are commiting plagiarism or theft. Contributing 
LLM generated code should be treated with the utmost care and consideration
for other people. 

LLMs cannot take responsibility for code, and the companies that train these models will not take responsibility for breaking the build.


<!-- Imagine that a guest at your dinner party shows up with a bottle of wine
that they stole from their brother-in-law's house. You won't know
that it was stolen unless they brag to you about it.
Analogously, we cannot necessarily know whether you used an LLM
to generate code that you're trying to commit, but you should feel
about the same way about contributing LLM-generated code as you
would showing up to a dinner party with something you stole from
a relative's house. -->

# Shoulders of giants disclaimer
The core functionality of this dynamical core is not primarily my work,
and is mostly a framework that helps unify the [Higher Order Methods Modeling Environment](https://doi.org/10.1029/2019MS001783) (HOMME) and the [Community Atmosphere Model-Spectral Element](https://doi.org/10.1029/2017MS001257) dynamical cores. I've been so fortunate to work with Drs. Mark Taylor, Oksana Guba, and Peter Lauritzen, and I would not have been able to write this codebase without their mentorship. None of this mentorship would have happened at all if my advisor, Professor Christiane Jablonowski, hadn't helped me find a place for myself in the atmospheric modeling community, and helped develop my taste for science-driven dynamical core development.