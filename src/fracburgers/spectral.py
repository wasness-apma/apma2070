"""General-purpose spectral solver for the fractional Burgers equation.

Function-to-function design: ``solve`` takes an
``InitialCondition`` and returns a ``Solution`` — a TF callable
``(x, t) -> u(x, t)`` that's differentiable end-to-end.

Pipeline (closed-form in time, spectral in space):

    1.  u_0  --(Cole–Hopf)-->  θ_0   = exp(-I^α u_0 / (2ν))
                               (uses closed form when available)
    2.  θ_0  --(heat propagator)-->  θ(·, t) = F^{-1}[exp(-νk²t) F[θ_0]]
    3.  θ(·, t)  --(inverse Cole–Hopf)-->  u(·, t) = -2ν D^α ln θ

No time stepping is required — the heat equation has an exact
Fourier-space evolution. The returned ``Solution`` defers the
heat-evolve and Cole–Hopf-back steps until query time, so calling
``u(x, t)`` for a new ``t`` does one full FFT pipeline and returns
trig-interpolated values at the requested ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.initial_conditions import InitialCondition
from fracburgers.solution import Solution, GridEvaluator
from fracburgers.cole_hopf import u_to_theta_0, theta_to_u
import fracburgers.operators as operators

@dataclass
class SpectralSolver:
    """Fractional Burgers solver via the Cole–Hopf transform.

    Parameters
    ----------
    grid : FourierGrid
    nu : float
    alpha : float
        Fractional order in ``(0, 1)``.
    """

    grid: FourierGrid
    nu: float
    alpha: float

    def solve(self, ic: InitialCondition) -> Solution:
        """Solve the PDE; return a TF-callable solution.

        Uses ``ic.theta_0`` (closed form) if available, else computes
        ``θ_0`` from ``ic.u_0`` via the spectral fractional integral.
        """

        # Track the conserved k=0 mode explicitly. The Cole-Hopf spectral
        # pipeline reconstructs only the zero-mean component.
        u0_mean = tf.reduce_mean(ic.u_0(self.grid.x_tf), axis=-1, keepdims=True)

        # Step 1: Cole–Hopf forward transform on initial data.
        if ic.theta_0 is None:
            theta_0 = lambda x: u_to_theta_0(ic.u_0(x), self.alpha, self.nu, self.grid)
        else:
            theta_0 = lambda x: ic.theta_0(x, self.nu, self.alpha)

        # Step 2: Create GridEvaluator for the solution.
        def on_grid(t: tf.Tensor) -> tf.Tensor:
            """Evaluate u(x_grid, t) at the requested times."""
            # Heat-evolve θ_0 to get θ(·, t).
            theta_t = operators.heat_evolve(theta_0(self.grid.x), t, self.nu, self.grid)

            # Cole–Hopf back transform returns the zero-mean component; add back
            # the conserved spatial mean from the initial condition.
            u_centered = theta_to_u(theta_t, self.alpha, self.nu, self.grid)
            return u_centered + u0_mean

        return Solution(self.grid, on_grid)
