#!/usr/bin/env python3
"""Train the heat-equation PINN and save the model + diagnostic plots.

The PINN models θ(x, t) and is trained against:
  - PDE residual:  θ_t − ν θ_{xx} = 0   (autograd, collocation points)
  - IC penalty:    θ(x, 0) = θ_0(x)      (closed-form or spectral)

After training the Cole-Hopf back-transform gives u, which is compared
against the spectral solver at several time snapshots.

GPU efficiency: the entire training step (random sampling + forward +
backward) is compiled into a single tf.function, eliminating all
Python-level CPU/GPU round-trips inside the hot loop.

Example:
    python scripts/train_pinn.py --ic sine --alpha 0.5 --epochs 10000
    python scripts/train_pinn.py --ic gaussian --alpha 0.5 --epochs 20000 \\
        --n-collocation 16384 --hidden-layers 128,128,128,128
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import sys

_CACHE_DIR = Path(tempfile.gettempdir()) / "fracburgers-cache"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from fracburgers import initial_conditions


class _Tee:
    """Mirror writes to two file-like objects (e.g. stdout + a log file)."""
    def __init__(self, *fds):
        self._fds = fds
    def write(self, data):
        for fd in self._fds:
            fd.write(data)
    def flush(self):
        for fd in self._fds:
            fd.flush()
from fracburgers.cole_hopf import u_to_log_theta_0
from fracburgers.grid import FourierGrid
from fracburgers.interpolation import trig_interp
from fracburgers.pinn import HeatPINN, configure_gpu, to_solution
from fracburgers.result_naming import get_output_dir
from fracburgers.spectral import SpectralSolver


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _csv_ints(text: str) -> list[int]:
    values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _csv_floats(text: str) -> list[float]:
    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Problem
    p.add_argument("--ic", choices=sorted(initial_conditions.REGISTRY), default="gaussian")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--t-max", type=float, default=1.0)
    p.add_argument("--N", type=int, default=512, help="Fourier grid size (spectral reference)")
    p.add_argument("--L", type=float, default=np.pi, help="half-domain length")

    # Architecture
    p.add_argument(
        "--hidden-layers",
        type=_csv_ints,
        default=[64, 64, 64, 64],
        metavar="W1,W2,...",
        help="hidden layer widths",
    )
    p.add_argument("--activation", default="tanh")

    # Training
    p.add_argument("--epochs", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-decay", action="store_true", help="cosine learning rate decay to lr/100")
    p.add_argument("--n-collocation", type=int, default=8_192, help="PDE collocation points per step")
    p.add_argument("--n-initial", type=int, default=512, help="IC points per step")
    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--ic-weight", type=float, default=100.0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--uniform-grid", action="store_true",
                   help="use a fixed uniform meshgrid for collocation instead of random sampling")

    # Hardware
    p.add_argument("--device", default="/GPU:0", help="TF device string: /GPU:0 or /CPU:0")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                   help="floating-point precision (float32 is ~32x faster on T4 GPU)")

    # Output
    p.add_argument("--out-dir", type=Path, default=None, help="output directory (auto-generated from params if not specified)")
    p.add_argument("--out-model", type=Path, default=None, help="path to save model weights (default: <out-dir>/model.weights.h5)")
    p.add_argument(
        "--snap-times",
        type=_csv_floats,
        default=None,
        metavar="T1,T2,...",
        help="time slices for comparison plot (default: 4 evenly spaced up to t-max)",
    )
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    checks = [
        (0.0 < args.alpha < 1.0, "--alpha must be in (0, 1)"),
        (args.nu > 0.0, "--nu must be positive"),
        (args.t_max > 0.0, "--t-max must be positive"),
        (args.N >= 4, "--N must be >= 4"),
        (args.L > 0.0, "--L must be positive"),
        (args.epochs >= 1, "--epochs must be >= 1"),
        (args.lr > 0.0, "--lr must be positive"),
        (args.n_collocation >= 1, "--n-collocation must be >= 1"),
        (args.n_initial >= 1, "--n-initial must be >= 1"),
        (all(w >= 1 for w in args.hidden_layers), "hidden layer widths must be >= 1"),
        (len(args.hidden_layers) >= 1, "--hidden-layers must have at least one value"),
    ]
    for ok, msg in checks:
        if not ok:
            raise SystemExit(msg)


# ---------------------------------------------------------------------------
# log θ_0 interpolator — evaluated at arbitrary random IC points each step
# ---------------------------------------------------------------------------

def build_log_theta0_fn(ic, grid: FourierGrid, nu: float, alpha: float, fdtype=tf.float32):
    """Return a callable  x_ic (B, 1) → log θ_0 (B,)  usable inside tf.function.

    Prefers a closed-form ``ic.log_theta_0`` when available; otherwise
    pre-computes log θ_0 spectrally from u_0 (no exp/log roundtrip) and
    trig-interpolates to arbitrary points.
    """
    if ic.log_theta_0 is not None:
        # Cast input to float64 for the IC formula, cast result back to fdtype.
        def log_theta0_fn(x_ic: tf.Tensor) -> tf.Tensor:
            result = ic.log_theta_0(tf.cast(x_ic[:, 0], tf.float64), nu, alpha)
            return tf.cast(result, fdtype)
    else:
        # Spectral: compute in float64 (trig_interp requires it), cast output.
        u0_grid = ic.u_0(grid.x_tf)
        log_theta0_grid = u_to_log_theta_0(u0_grid, alpha, nu, grid)
        log_theta0_const = tf.constant(log_theta0_grid.numpy(), dtype=tf.float64)

        def log_theta0_fn(x_ic: tf.Tensor) -> tf.Tensor:
            # trig_interp: (1, N) × (B,) → (1, B), take row 0
            result = trig_interp(log_theta0_const[None, :], tf.cast(x_ic[:, 0], tf.float64), grid)[0]
            return tf.cast(result, fdtype)

    return log_theta0_fn


# ---------------------------------------------------------------------------
# GPU-efficient training step — compiled to a TF graph once
# ---------------------------------------------------------------------------

def make_train_step(
    model: HeatPINN,
    optimizer: tf.keras.optimizers.Optimizer,
    log_theta0_fn,
    grid: FourierGrid,
    nu: float,
    t_max: float,
    n_col: int,
    n_ic: int,
    pde_weight: float,
    ic_weight: float,
    fdtype=tf.float32,
    uniform_grid: bool = False,
):
    """Factory: returns a @tf.function compiled training step.

    When uniform_grid=False (default) collocation points are re-sampled
    randomly each step.  When True a fixed n_x × n_t meshgrid is used —
    same points every epoch, which can improve stability at the cost of
    some stochastic coverage.
    """
    nu_c = tf.constant(nu, dtype=fdtype)
    pde_w = tf.constant(pde_weight, dtype=fdtype)
    ic_w = tf.constant(ic_weight, dtype=fdtype)

    if uniform_grid:
        np_dtype = np.float32 if fdtype == tf.float32 else np.float64
        n_x = max(1, int(round(n_col ** 0.5)))
        n_t = max(1, (n_col + n_x - 1) // n_x)
        x_vals = np.linspace(-grid.L, grid.L, n_x, endpoint=False, dtype=np_dtype)
        t_vals = np.linspace(0.0, t_max, n_t, endpoint=False, dtype=np_dtype)
        XX, TT = np.meshgrid(x_vals, t_vals)
        x_col_c = tf.constant(XX.reshape(-1, 1), dtype=fdtype)
        t_col_c = tf.constant(TT.reshape(-1, 1), dtype=fdtype)
        x_ic_c = tf.constant(
            np.linspace(-grid.L, grid.L, n_ic, dtype=np_dtype).reshape(-1, 1),
            dtype=fdtype,
        )
        print(f"Uniform grid: {n_x} × {n_t} = {n_x * n_t} collocation points, "
              f"{n_ic} IC points")
    else:
        L_c = tf.constant(grid.L, dtype=fdtype)
        t_max_c = tf.constant(t_max, dtype=fdtype)
        zero = tf.constant(0.0, dtype=fdtype)

    @tf.function
    def step():
        if uniform_grid:
            x_col = x_col_c
            t_col = t_col_c
            x_ic = x_ic_c
        else:
            x_col = tf.random.uniform((n_col, 1), -L_c, L_c, dtype=fdtype)
            t_col = tf.random.uniform((n_col, 1), zero, t_max_c, dtype=fdtype)
            x_ic = tf.random.uniform((n_ic, 1), -L_c, L_c, dtype=fdtype)

        # ── forward + PDE residual + IC penalty ─────────────────────────
        with tf.GradientTape() as model_tape:
            # Three nested tapes are required for second derivatives:
            #   t2  records theta_x so that d(theta_x)/d(x) is available
            #   t1  (persistent) gives both theta_t and theta_x in one pass
            #   model_tape  gives d(loss)/d(weights) through everything
            with tf.GradientTape() as t2:
                t2.watch(x_col)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch([x_col, t_col])
                    theta_col = model(tf.concat([x_col, t_col], axis=-1))  # (n_col, 1)
                # computed INSIDE t2 so t2 records theta_x's dependence on x
                theta_t = t1.gradient(theta_col, t_col)   # (n_col, 1)
                theta_x = t1.gradient(theta_col, x_col)   # (n_col, 1)
                del t1
            theta_xx = t2.gradient(theta_x, x_col)        # (n_col, 1)

            residual = theta_t - nu_c * theta_xx
            pde_loss = tf.reduce_mean(tf.square(residual))

            t_zero = tf.zeros((n_ic, 1), dtype=fdtype)
            # Log-space IC MSE: compare log θ directly via the linear
            # pre-activation against the algebraic log θ_0 — no exp, no log,
            # naturally relative across the wide dynamic range of θ_0.
            log_theta_ic_pred = model.log_theta(tf.concat([x_ic, t_zero], axis=-1))[:, 0]
            log_theta_ic_true = log_theta0_fn(x_ic)
            ic_loss = tf.reduce_mean(tf.square(log_theta_ic_pred - log_theta_ic_true))

            total = pde_w * pde_loss + ic_w * ic_loss

        grads = model_tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, pde_loss, ic_loss

    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training(
    train_step,
    epochs: int,
    log_every: int,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {"total": [], "pde": [], "ic": []}

    print(f"Training for {epochs} epochs …")
    for epoch in range(1, epochs + 1):
        total, pde, ic = train_step()
        history["total"].append(float(total))
        history["pde"].append(float(pde))
        history["ic"].append(float(ic))

        log = epoch == 1 or epoch == epochs or (log_every > 0 and epoch % log_every == 0)
        if log:
            print(
                f"  epoch {epoch:6d}/{epochs}"
                f"  total={float(total):.4e}"
                f"  pde={float(pde):.4e}"
                f"  ic={float(ic):.4e}"
            )

    return history


# ---------------------------------------------------------------------------
# Heat-equation reference θ
# ---------------------------------------------------------------------------

def heat_theta_on_grid(theta0_fn, grid: FourierGrid, nu: float, t: float, fdtype) -> np.ndarray:
    """θ(x, t) from heat equation θ_t = ν θ_xx via spectral evolution.

    Evaluates θ_0 on the Fourier grid, advances with exp(-ν k² t) in
    Fourier space, and returns the real-space values as float64.
    """
    x_flat = tf.constant(grid.x.reshape(-1, 1), dtype=fdtype)
    theta0 = theta0_fn(x_flat).numpy().flatten().astype(np.float64)
    theta0_hat = np.fft.rfft(theta0)
    k = np.fft.rfftfreq(grid.N, d=grid.dx) * (2.0 * np.pi)
    theta_hat_t = theta0_hat * np.exp(-nu * k ** 2 * t)
    return np.fft.irfft(theta_hat_t, n=grid.N)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_loss_plot(history: dict[str, list[float]], out: Path) -> Path:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    fig.suptitle("PINN training loss", fontsize=11)

    for ax, key, color in zip(axes, ["total", "pde", "ic"], ["tab:blue", "tab:orange", "tab:green"]):
        vals = np.asarray(history[key])
        ax.semilogy(epochs, vals, color=color, linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_title(key)
        ax.grid(True, which="both", alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def save_heat_comparison_plot(
    model: "HeatPINN",
    theta0_fn,
    grid: FourierGrid,
    nu: float,
    snap_times: np.ndarray,
    out: Path,
    title: str = "",
    fdtype=tf.float32,
) -> Path:
    """Rows = snapshots, cols = (θ overlay, pointwise θ error).

    Compares the PINN's raw θ output against the heat-equation spectral
    reference.  Separates 'θ is wrong' from 'Cole–Hopf inversion is wrong'.
    """
    n_times = len(snap_times)
    ref_color = "#2980C7"
    err_color = "#5A0306"
    np_fdtype = np.float32 if fdtype == tf.float32 else np.float64

    x = grid.x
    x_flat = tf.constant(x.reshape(-1, 1), dtype=fdtype)

    fig, axes = plt.subplots(
        n_times, 2,
        figsize=(11.0, 2.6 * n_times),
        constrained_layout=True,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    if n_times == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle((title or "PINN θ vs heat-equation θ"), fontsize=10)
    axes[0, 0].set_title(r"$\theta(x,t)$", fontsize=10)
    axes[0, 1].set_title(r"PINN $-$ heat  (pointwise)", fontsize=10)

    for ti, t in enumerate(snap_times):
        t_flat = tf.constant(np.full((len(x), 1), t, dtype=np_fdtype), dtype=fdtype)
        theta_pinn = model(tf.concat([x_flat, t_flat], axis=-1))[:, 0].numpy()
        theta_heat = heat_theta_on_grid(theta0_fn, grid, nu, float(t), fdtype)
        err = theta_pinn.astype(np.float64) - theta_heat

        ax_t, ax_e = axes[ti, 0], axes[ti, 1]
        ax_t.set_facecolor("#FCFCFC")
        ax_e.set_facecolor("#FCFCFC")

        ax_t.plot(x, theta_heat, color=ref_color, linewidth=1.8, alpha=0.5, label="heat (spectral)")
        ax_t.plot(x, theta_pinn, color=err_color, linewidth=1.2, linestyle="--", alpha=0.95, label="PINN")
        ax_t.set_ylabel(f"t = {t:g}", fontsize=9)
        ax_t.grid(True, color="#B0B0B0", alpha=0.28)
        if ti == 0:
            ax_t.legend(fontsize=8, loc="upper right", framealpha=0.92, facecolor="white")

        ax_e.plot(x, err, color=err_color, linewidth=1.4)
        ax_e.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.45)
        ref_inf = max(float(np.max(np.abs(theta_heat))), np.finfo(float).tiny)
        rel = float(np.max(np.abs(err))) / ref_inf
        ax_e.set_ylabel(rf"$\epsilon$ (rel: {rel:.2e})", fontsize=8)
        ax_e.grid(True, color="#B0B0B0", alpha=0.28)

        if ti == n_times - 1:
            ax_t.set_xlabel("$x$", fontsize=9)
            ax_e.set_xlabel("$x$", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", edgecolor="white")
    plt.close(fig)
    return out


def save_comparison_plot(
    pinn_sol,
    spec_sol,
    x: np.ndarray,
    snap_times: np.ndarray,
    out: Path,
    title: str = "",
) -> Path:
    """Rows = time snapshots, cols = (u overlay, pointwise error)."""
    print(f"[DEBUG] save_comparison_plot: pinn_sol type={type(pinn_sol)}, n_times={len(snap_times)}")
    n_times = len(snap_times)
    ref_color = "#2980C7"
    err_color = "#5A0306"

    fig, axes = plt.subplots(
        n_times, 2,
        figsize=(11.0, 2.6 * n_times),
        constrained_layout=True,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    if n_times == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title or "PINN vs spectral", fontsize=10)
    axes[0, 0].set_title("$u(x,t)$", fontsize=10)
    axes[0, 1].set_title("PINN $-$ spectral  (pointwise)", fontsize=10)

    x_tf = tf.constant(x, dtype=tf.float64)

    for ti, t in enumerate(snap_times):
        t_tf = tf.constant(float(t), dtype=tf.float64)
        u_pinn = pinn_sol(x_tf, t_tf).numpy()
        u_spec = spec_sol(x_tf, t_tf).numpy()
        err = u_pinn - u_spec
        
        # Debug: check u_pinn values
        n_nan = np.sum(np.isnan(u_pinn))
        n_inf = np.sum(np.isinf(u_pinn))
        print(f"  t={t:g}: u_pinn shape={u_pinn.shape}, nan={n_nan}, inf={n_inf}, "
              f"min={np.nanmin(u_pinn):g}, max={np.nanmax(u_pinn):g}")

        ax_u, ax_e = axes[ti, 0], axes[ti, 1]
        ax_u.set_facecolor("#FCFCFC")
        ax_e.set_facecolor("#FCFCFC")
        ax_u.plot(x, u_spec, color=ref_color, linewidth=1.8, alpha=0.5, label="spectral")
        ax_u.plot(x, u_pinn, color=err_color, linewidth=1.2, linestyle="--", alpha=0.95, label="PINN")
        ax_u.set_ylabel(f"t = {t:g}", fontsize=9)
        ax_u.grid(True, color="#B0B0B0", alpha=0.28)
        if ti == 0:
            ax_u.legend(fontsize=8, loc="upper right", framealpha=0.92, facecolor="white")

        ax_e.plot(x, err, color=err_color, linewidth=1.4)
        ax_e.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.45)
        ref_inf = max(float(np.max(np.abs(u_spec))), np.finfo(float).tiny)
        rel = float(np.max(np.abs(err))) / ref_inf
        ax_e.set_ylabel(rf"$\epsilon$ ($\|u_{{spec}}\|_\infty$-rel: {rel:.2e})", fontsize=8)
        ax_e.grid(True, color="#B0B0B0", alpha=0.28)

        if ti == n_times - 1:
            ax_u.set_xlabel("$x$", fontsize=9)
            ax_e.set_xlabel("$x$", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", edgecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate(args)

    # Auto-generate output directory if not specified
    if args.out_dir is None:
        args.out_dir = get_output_dir(
            Path("results"),
            "train_pinn",
            {
                "ic": args.ic,
                "alpha": args.alpha,
                "nu": args.nu,
                "epochs": args.epochs,
                "__tags": ["ic", "alpha", "nu", "epochs"],
            },
        )
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.out_dir / "train.log"
    _log_f = log_path.open("w", buffering=1)
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _log_f)

    try:
        _main(args)
    finally:
        sys.stdout = _orig_stdout
        _log_f.close()
    print(f"Log saved to {log_path}")


def _main(args) -> None:
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    device = configure_gpu(prefer_gpu=args.device != "/CPU:0", verbose=True)
    fdtype = tf.float32 if args.dtype == "float32" else tf.float64
    print(f"Using dtype: {args.dtype}")

    grid = FourierGrid.make(N=args.N, L=args.L)
    ic = initial_conditions.get(args.ic)

    snap_times = np.asarray(
        args.snap_times if args.snap_times is not None
        else np.linspace(0.0, args.t_max, 5)[1:],  # 4 interior points
        dtype=np.float64,
    )

    # ── build model ──────────────────────────────────────────────────────
    model = HeatPINN(
        hidden_layers=tuple(args.hidden_layers),
        activation=args.activation,
        dtype=args.dtype,
        L=args.L,
    )
    # Warm-up call to build weights before any tf.function tracing
    with tf.device(device):
        _ = model(tf.zeros((1, 2), dtype=fdtype))
    print(f"Model parameters: {model.count_params():,}")

    # ── optimizer + LR schedule ──────────────────────────────────────────
    if args.lr_decay:
        schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.epochs,
            alpha=0.01,  # decays to lr * 0.01
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # ── log θ_0 interpolator (canonical) + θ_0 derived for heat plot ─────
    log_theta0_fn = build_log_theta0_fn(ic, grid, args.nu, args.alpha, fdtype=fdtype)
    theta0_fn = lambda x: tf.exp(log_theta0_fn(x))  # derived, used only for heat plot

    # ── compile training step ────────────────────────────────────────────
    with tf.device(device):
        train_step = make_train_step(
            model=model,
            optimizer=optimizer,
            log_theta0_fn=log_theta0_fn,
            grid=grid,
            nu=args.nu,
            t_max=args.t_max,
            n_col=args.n_collocation,
            n_ic=args.n_initial,
            pde_weight=args.pde_weight,
            ic_weight=args.ic_weight,
            fdtype=fdtype,
            uniform_grid=args.uniform_grid,
        )
        # Trace once so the first epoch doesn't include compile time
        print("Compiling tf.function (one-time) …")
        _ = train_step()

    # ── train ────────────────────────────────────────────────────────────
    with tf.device(device):
        history = run_training(train_step, args.epochs, args.log_every)

    # ── save model weights ────────────────────────────────────────────────
    model_path = args.out_model or args.out_dir / "model.weights.h5"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(model_path))
    print(f"Saved model weights to {model_path}")

    # ── loss plot ─────────────────────────────────────────────────────────
    loss_path = save_loss_plot(history, args.out_dir / "loss.png")
    print(f"Saved loss plot to {loss_path}")

    # ── spectral reference ────────────────────────────────────────────────
    print("Running spectral solver for comparison …")
    spec_sol = SpectralSolver(grid=grid, nu=args.nu, alpha=args.alpha).solve(ic)
    print(f"[DEBUG] spec_sol type={type(spec_sol)}")
    
    print(f"[DEBUG] Creating PINN solution wrapper with to_solution()…")
    pinn_sol = to_solution(model, grid, args.nu, args.alpha)
    print(f"[DEBUG] pinn_sol type={type(pinn_sol)}")
    
    # Quick test: evaluate PINN at a single point
    try:
        x_test = tf.constant([0.0], dtype=tf.float64)
        t_test = tf.constant(0.0, dtype=tf.float64)
        u_test = pinn_sol(x_test, t_test).numpy()
        print(f"[DEBUG] pinn_sol(x=0, t=0) = {u_test} (shape={u_test.shape})")
    except Exception as e:
        print(f"[ERROR] pinn_sol evaluation failed: {e}")


    label = (
        f"PINN vs spectral  —  ic={args.ic}, α={args.alpha:g}, "
        f"ν={args.nu:g}, N={args.N}, layers={args.hidden_layers}"
    )
    comp_path = save_comparison_plot(
        pinn_sol=pinn_sol,
        spec_sol=spec_sol,
        x=grid.x,
        snap_times=snap_times,
        out=args.out_dir / "comparison.png",
        title=label,
    )
    print(f"Saved comparison plot to {comp_path}")

    # ── θ comparison plot ─────────────────────────────────────────────────
    heat_path = save_heat_comparison_plot(
        model=model,
        theta0_fn=theta0_fn,
        grid=grid,
        nu=args.nu,
        snap_times=snap_times,
        out=args.out_dir / "heat_comparison.png",
        title=label,
        fdtype=fdtype,
    )
    print(f"Saved heat comparison plot to {heat_path}")

    # ── JSON report ───────────────────────────────────────────────────────
    final_total = history["total"][-1]
    final_pde = history["pde"][-1]
    final_ic = history["ic"][-1]
    report = {
        "config": {
            "ic": args.ic,
            "alpha": args.alpha,
            "nu": args.nu,
            "t_max": args.t_max,
            "N": args.N,
            "L": args.L,
            "hidden_layers": args.hidden_layers,
            "activation": args.activation,
            "dtype": args.dtype,
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_decay": args.lr_decay,
            "n_collocation": args.n_collocation,
            "n_initial": args.n_initial,
            "pde_weight": args.pde_weight,
            "ic_weight": args.ic_weight,
            "seed": args.seed,
            "uniform_grid": args.uniform_grid,
            "device": device,
        },
        "final_losses": {"total": final_total, "pde": final_pde, "ic": final_ic},
        "history": history,
        "artifacts": {
            "model_weights": str(model_path),
            "loss_plot": str(loss_path),
            "comparison_plot": str(comp_path),
        },
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
