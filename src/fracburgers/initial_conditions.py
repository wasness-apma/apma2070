"""Preset initial conditions, exposed as TF callables.

Each preset is an ``InitialCondition`` with three pieces:

    .u_0(x)                           always defined; returns u_0(x)
    .log_theta_0(x, nu, alpha)        canonical form when a closed form
                                      for log θ_0 = -I^α u_0 / (2ν) is
                                      known, otherwise None
    .theta_0(x, nu, alpha)            optional; auto-derived as
                                      exp(log_theta_0) when log_theta_0
                                      is provided and theta_0 isn't

Prefer supplying ``log_theta_0`` — it has tighter dynamic range than
``θ_0`` (lives on a log scale, no exp overflow risk) and is what every
downstream consumer (PINN IC loss, Cole–Hopf inverse) actually needs.
``θ_0`` is kept for back-compat with consumers (spectral solver, viz,
references) that still want θ; it is derived once via ``exp`` and
cached on the dataclass.

All callables take and return ``tf.Tensor`` of dtype ``float64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf


UFunc = Callable[[tf.Tensor], tf.Tensor]
ThetaFunc = Callable[[tf.Tensor, float, float], tf.Tensor]


@dataclass(frozen=True)
class InitialCondition:
    """Bundle of an IC with optional closed-form log θ_0 and θ_0."""

    name: str
    u_0: UFunc
    log_theta_0: ThetaFunc | None = None
    theta_0: ThetaFunc | None = None

    def __post_init__(self):
        # Auto-derive θ_0 = exp(log θ_0) when only log_theta_0 was supplied.
        # frozen=True → must go through object.__setattr__.
        if self.theta_0 is None and self.log_theta_0 is not None:
            log_fn = self.log_theta_0
            object.__setattr__(
                self, "theta_0",
                lambda x, nu, alpha: tf.exp(log_fn(x, nu, alpha)),
            )


def sine() -> InitialCondition:
    """``u_0(x) = sin x``.

    Closed form: ``I^α [sin x] = sin(x - α π / 2)``, so
    ``log θ_0(x) = -sin(x - α π / 2) / (2ν)`` (bounded in ``[-1/(2ν),
    1/(2ν)]`` — much friendlier than θ_0 which spans up to
    ``exp(1/(2ν))``).  θ_0 is auto-derived via ``exp``.

    For periodicity make sure the grid's ``L`` is an integer multiple
    of ``π``.
    """
    return InitialCondition(
        name = "sine",
        u_0 = tf.sin,
        log_theta_0 = lambda x, nu, alpha: -tf.sin(x - alpha * np.pi / 2) / (2 * nu),
    )


def gaussian() -> InitialCondition:
    """``u_0(x) = exp(-x²)``.

    No elementary closed form for ``I^α``, so both ``log_theta_0`` and
    ``theta_0`` are None and the spectral pipeline computes them.
    """
    return InitialCondition(
        name = "gaussian",
        u_0 = lambda x: tf.exp(-x**2),
    )


REGISTRY: dict[str, Callable[[], InitialCondition]] = {
    "sine": sine,
    "gaussian": gaussian,
}


def get(name: str) -> InitialCondition:
    """Look up a preset by name; used by the CLI scripts."""
    try:
        return REGISTRY[name]()
    except KeyError as exc:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown initial condition {name!r}. Choose one of: {valid}.") from exc
