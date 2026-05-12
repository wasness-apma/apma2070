"""Physics-informed neural network for the heat equation.

The PINN models ``θ(x, t)`` as a feed-forward NN trained against the
heat-equation residual

    R(x, t) = θ_t(x, t) - ν θ_{xx}(x, t)

(both derivatives via TF autograd) plus an initial-condition penalty.
After training, ``HeatPINN.to_solution`` wraps the network in the
spectral Cole–Hopf post-processor and returns a ``Solution`` —
i.e., a TF callable with the same signature as the one produced by
``SpectralSolver``. Downstream code (plots, error metrics) is then
solver-agnostic.

Training on the heat equation rather than directly on the fractional
Burgers residual keeps the NN's job (smooth approximation of θ)
cleanly separate from the spectral pipeline's job (computing D^α),
which makes error attribution clean for the comparison study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from fracburgers.cole_hopf import theta_to_u, u_to_theta_0
from fracburgers.grid import FourierGrid
from fracburgers.initial_conditions import InitialCondition, ThetaFunc
from fracburgers.solution import Solution


class HeatPINN(tf.keras.Model):
    """Fully-connected NN ``(x, t) ↦ θ(x, t)``.

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Widths of the hidden layers.
    activation : str
        Hidden-layer activation. ``tanh`` is standard for PINNs.
    """

    def __init__(self, hidden_layers: tuple[int, ...] = (64, 64, 64, 64),
                 activation: str = "tanh", dtype: str = "float64", **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers must contain at least one width")
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        # +2 layers: explicit input projection (2 → first width) and output (→ 1)
        self._input_layer = tf.keras.layers.Dense(
            self.hidden_layers[0], activation=self.activation, dtype=dtype
        )
        self._hidden = [
            tf.keras.layers.Dense(width, activation=self.activation, dtype=dtype)
            for width in self.hidden_layers[1:]
        ]
        self._output_layer = tf.keras.layers.Dense(1, dtype=dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass: ``inputs`` shape ``(B, 2)`` with ``[:, 0]=x, [:, 1]=t``.

        Returns shape ``(B, 1)``.
        """
        x, t = inputs[:, 0:1], inputs[:, 1:2]  # each (B, 1)
        z = self._input_layer(tf.concat([x, t], axis=-1))
        for layer in self._hidden:
            z = layer(z)
        return self._output_layer(z)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "hidden_layers": list(self.hidden_layers),
                "activation": self.activation,
                "dtype": self.dtype,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        hidden_layers = tuple(config.pop("hidden_layers", [64, 64, 64, 64]))
        activation = config.pop("activation", "tanh")
        dtype = config.pop("dtype", "float64")
        return cls(hidden_layers=hidden_layers, activation=activation, dtype=dtype, **config)


@dataclass
class TrainingConfig:
    """Hyperparameters for PINN training."""

    n_collocation: int = 10_000
    n_initial: int = 256
    learning_rate: float = 1e-3
    epochs: int = 5_000
    pde_weight: float = 1.0
    ic_weight: float = 100.0
    log_every: int = 100
    seed: int | None = None
    prefer_gpu: bool = True
    verbose: bool = True


def configure_gpu(prefer_gpu: bool = True, verbose: bool = True) -> str:
    """Pick runtime device and enable GPU memory-growth when available.

    Returns a TensorFlow device string (``"/GPU:0"`` or ``"/CPU:0"``).
    """

    if not prefer_gpu:
        if verbose:
            print("GPU disabled by config; using CPU")
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if verbose:
            print("GPU not available, using CPU")
        return "/CPU:0"

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        if verbose:
            print(f"Could not set GPU memory growth: {exc}")

    if verbose:
        print(f"GPU available: {gpus[0]}")
        print("Using device: /GPU:0")
    return "/GPU:0"


def heat_residual(model: HeatPINN, x: tf.Tensor, t: tf.Tensor,
                  nu: float) -> tf.Tensor:
    """``θ_t - ν θ_{xx}`` at points ``(x, t)`` via autograd.

    Three nested tapes: the outer (t2) records ``θ_x`` so that
    ``d(θ_x)/dx`` is available; the inner persistent tape (t1) gives
    both ``θ_t`` and ``θ_x`` in one forward pass.
    """
    with tf.GradientTape() as t2:
        t2.watch(x)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch([x, t])
            theta = model(tf.concat([x, t], axis=-1))  # (B, 1)
        # computed inside t2 so t2 records theta_x's dependence on x
        theta_t = t1.gradient(theta, t)   # (B, 1)
        theta_x = t1.gradient(theta, x)   # (B, 1)
        del t1
    theta_xx = t2.gradient(theta_x, x)    # (B, 1)

    return theta_t - nu * theta_xx

def initial_condition_loss(model: HeatPINN, theta0: ThetaFunc, x_ic: tf.Tensor,
                           nu: float, alpha: float) -> tf.Tensor:
    """MSE penalty for deviation from the initial condition."""

    t0 = tf.zeros_like(x_ic)  # (N_ic, 1)
    inputs = tf.concat([x_ic, t0], axis=-1)  # (N_ic, 2)
    theta_pred = model(inputs)[:, 0]  # (N_ic,)
    theta_expected = tf.reshape(theta0(x_ic, nu, alpha), [-1])  # (N_ic,)
    return tf.reduce_mean((theta_pred - theta_expected) ** 2)


# Backward-compatible alias for earlier typo.
intial_condition_loss = initial_condition_loss

def train(model: HeatPINN, ic: InitialCondition, grid: FourierGrid,
          nu: float, alpha: float, t_max: float,
          config: TrainingConfig) -> dict:
    """Train ``model`` to satisfy the heat equation on ``[-L, L] × [0, t_max]``.

    ``θ_0`` for the IC loss is taken from ``ic.theta_0`` if available,
    otherwise computed spectrally from ``ic.u_0`` via the same
    pipeline used by ``SpectralSolver``.

    Returns a dict of per-epoch loss components for plotting.
    """
    if config.seed is not None:
        tf.random.set_seed(config.seed)

    theta0: Callable[[tf.Tensor, float, float], tf.Tensor]
    if ic.theta_0 is not None:
        theta0 = ic.theta_0
    else:
        theta0 = lambda x, nu_, alpha_: u_to_theta_0(ic.u_0(x), alpha_, nu_, grid)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    L = tf.constant(grid.L, dtype=tf.float64)
    t_max_tf = tf.constant(t_max, dtype=tf.float64)

    history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "ic": [],
    }

    device = configure_gpu(prefer_gpu=config.prefer_gpu, verbose=config.verbose)

    for epoch in range(1, config.epochs + 1):
        with tf.device(device):
            x_col = tf.random.uniform(
                shape=(config.n_collocation, 1),
                minval=-L,
                maxval=L,
                dtype=tf.float64,
            )
            t_col = tf.random.uniform(
                shape=(config.n_collocation, 1),
                minval=tf.constant(0.0, dtype=tf.float64),
                maxval=t_max_tf,
                dtype=tf.float64,
            )
            x_ic = tf.random.uniform(
                shape=(config.n_initial, 1),
                minval=-L,
                maxval=L,
                dtype=tf.float64,
            )

            with tf.GradientTape() as tape:
                residual = heat_residual(model, x_col, t_col, nu)
                pde_loss = tf.reduce_mean(tf.square(residual))
                ic_loss = initial_condition_loss(model, theta0, x_ic, nu, alpha)
                total_loss = config.pde_weight * pde_loss + config.ic_weight * ic_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_v = float(total_loss.numpy())
        pde_v = float(pde_loss.numpy())
        ic_v = float(ic_loss.numpy())
        history["total"].append(total_v)
        history["pde"].append(pde_v)
        history["ic"].append(ic_v)

        should_log = epoch == 1 or epoch == config.epochs
        if config.log_every > 0 and epoch % config.log_every == 0:
            should_log = True
        if should_log and config.verbose:
            print(
                f"epoch={epoch:5d} "
                f"total={total_v:.6e} pde={pde_v:.6e} ic={ic_v:.6e}"
            )

    return history


def to_solution(model: HeatPINN, grid: FourierGrid, nu: float,
                alpha: float) -> Solution:
    """Wrap a trained ``model`` as a ``Solution``.

    The wrapper samples ``θ`` on the grid for any requested ``t``,
    runs the spectral Cole–Hopf back-transform, and returns ``u``
    interpolated to arbitrary query points.
    """
    def on_grid(t: tf.Tensor) -> tf.Tensor:
        t = tf.cast(tf.convert_to_tensor(t), dtype=tf.float64)

        is_scalar_t = t.shape.rank == 0
        if is_scalar_t:
            t_batch = tf.reshape(t, [1, 1])
        elif t.shape.rank == 1:
            t_batch = t[:, None]
        else:
            t_batch = t

        n_times = tf.shape(t_batch)[0]
        x_batch = tf.broadcast_to(grid.x_tf[None, :], [n_times, grid.N])
        t_full = tf.broadcast_to(t_batch, [n_times, grid.N])

        inputs = tf.stack([x_batch, t_full], axis=-1)          # (T, N, 2)
        inputs_flat = tf.reshape(inputs, [-1, 2])              # (T*N, 2)
        theta_flat = model(inputs_flat)[:, 0]                  # (T*N,)
        theta_grid = tf.reshape(theta_flat, [n_times, grid.N]) # (T, N)

        u_grid = theta_to_u(theta_grid, alpha, nu, grid)       # (T, N)
        return u_grid[0] if is_scalar_t else u_grid

    return Solution(grid, on_grid)
