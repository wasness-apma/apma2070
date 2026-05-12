"""Spectral and PINN solvers for the fractional nonlinear Burgers equation."""

__version__ = "0.1.0"

from fracburgers.grid import FourierGrid
from fracburgers.solution import Solution

__all__ = ["FourierGrid", "Solution", "__version__"]
