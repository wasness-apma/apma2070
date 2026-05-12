"""Plotting helpers — consume ``Solution`` objects, produce matplotlib axes.

All helpers accept either a ``Solution`` (from any solver) or a raw
``tf.Tensor`` of grid samples. The ``Solution``-based path is the
common case and the one used by the comparison scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from fracburgers.solution import Solution


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _ensure_ax(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax


def _solution_snapshots(sol: Solution, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_tf = tf.constant(times[:, None], dtype=tf.float64)
    u = _to_numpy(sol.sample(t_tf))
    return sol.grid.x, u


def _as_spacetime_data(
    sol: Solution | np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    if isinstance(sol, Solution):
        return _solution_snapshots(sol, times)

    u = _to_numpy(sol)
    if u.ndim == 1 and times.size == 1:
        u = u[None, :]
    if u.ndim != 2:
        raise ValueError("For non-Solution input, expected shape (T, N).")
    if u.shape[0] != times.size:
        raise ValueError(f"Mismatch: got {u.shape[0]} snapshots but {times.size} times.")
    x = np.arange(u.shape[1], dtype=np.float64)
    return x, u


def _save_animation(anim, out: str | Path, fps: int, dpi: int) -> Path:
    import matplotlib.animation as mlanim

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".gif":
        writer = mlanim.PillowWriter(fps=fps)
    elif ext in {".mp4", ".m4v"}:
        writer = mlanim.FFMpegWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported movie extension: {ext!r}. Use .gif or .mp4.")

    anim.save(out_path, writer=writer, dpi=dpi)
    return out_path


def _finite_limits(arr: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
    arr_np = np.asarray(arr, dtype=np.float64)
    finite = arr_np[np.isfinite(arr_np)]
    if finite.size == 0:
        return -1.0, 1.0

    lo = float(np.min(finite))
    hi = float(np.max(finite))
    scale = max(abs(lo), abs(hi))
    pad = pad_frac * scale if scale > 0 else 1.0
    return lo - pad, hi + pad


def plot_snapshot(sol: Solution | np.ndarray, t: float, ax=None, **kwargs):
    """Plot ``u(x, t)`` at a single time on the spectral grid."""
    ax = _ensure_ax(ax)

    if isinstance(sol, Solution):
        x = sol.grid.x
        u = _to_numpy(sol.sample(tf.constant(float(t), dtype=tf.float64)))
        kwargs.setdefault("label", f"t={t:g}")
    else:
        u = _to_numpy(sol)
        if u.ndim != 1:
            raise ValueError(
                "For non-Solution input, plot_snapshot expects shape (N,)."
            )
        x = _to_numpy(kwargs.pop("x", np.arange(u.shape[0], dtype=np.float64)))

    ax.plot(x, u, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    return ax


def plot_evolution(sol: Solution | np.ndarray, times: np.ndarray, ax=None):
    """Overlay multiple time snapshots on one axis."""
    ax = _ensure_ax(ax)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    if isinstance(sol, Solution):
        t_tf = tf.constant(times[:, None], dtype=tf.float64)
        u = _to_numpy(sol.sample(t_tf))
        x = sol.grid.x
    else:
        u = _to_numpy(sol)
        if u.ndim == 1 and times.size == 1:
            u = u[None, :]
        if u.ndim != 2:
            raise ValueError(
                "For non-Solution input, plot_evolution expects shape (T, N)."
            )
        if u.shape[0] != times.size:
            raise ValueError(
                f"Mismatch: got {u.shape[0]} snapshots but {times.size} times."
            )
        x = np.arange(u.shape[1], dtype=np.float64)

    for i, t in enumerate(times):
        ax.plot(x, u[i], label=f"t={t:g}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    return ax


def plot_spacetime(sol: Solution | np.ndarray, times: np.ndarray, ax=None, **kwargs):
    """Heatmap of ``u(x, t)`` over the space-time plane."""
    ax = _ensure_ax(ax)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    if isinstance(sol, Solution):
        t_tf = tf.constant(times[:, None], dtype=tf.float64)
        u = _to_numpy(sol.sample(t_tf))
        x = sol.grid.x
    else:
        u = _to_numpy(sol)
        if u.ndim != 2:
            raise ValueError(
                "For non-Solution input, plot_spacetime expects shape (T, N)."
            )
        if u.shape[0] != times.size:
            raise ValueError(
                f"Mismatch: got {u.shape[0]} snapshots but {times.size} times."
            )
        x = _to_numpy(kwargs.pop("x", np.arange(u.shape[1], dtype=np.float64)))

    colorbar = kwargs.pop("colorbar", True)
    cmap = kwargs.pop("cmap", "RdBu_r")
    origin = kwargs.pop("origin", "lower")
    aspect = kwargs.pop("aspect", "auto")
    interpolation = kwargs.pop("interpolation", "nearest")

    if x.size > 1:
        dx = x[1] - x[0]
        x_min = x[0] - 0.5 * dx
        x_max = x[-1] + 0.5 * dx
    else:
        x_min = x[0] - 0.5
        x_max = x[0] + 0.5

    if times.size > 1:
        dt = times[1] - times[0]
        t_min = times[0] - 0.5 * dt
        t_max = times[-1] + 0.5 * dt
    else:
        t_min = times[0] - 0.5
        t_max = times[0] + 0.5

    im = ax.imshow(
        u,
        extent=[x_min, x_max, t_min, t_max],
        origin=origin,
        aspect=aspect,
        cmap=cmap,
        interpolation=interpolation,
        **kwargs,
    )
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label="u(x, t)")

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return ax


def plot_error(
    sol_ref: Solution | np.ndarray,
    sol_test: Solution | np.ndarray,
    times: np.ndarray,
    ax=None,
):
    """Per-time L²(x) error of ``sol_test`` against ``sol_ref``."""
    ax = _ensure_ax(ax)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    if isinstance(sol_ref, Solution) and isinstance(sol_test, Solution):
        x = sol_ref.grid.x_tf
        dx = float(sol_ref.grid.dx)
        errs = []
        for t in times:
            t_tf = tf.constant(float(t), dtype=tf.float64)
            u_ref = sol_ref(x, t_tf)
            u_test = sol_test(x, t_tf)
            diff = u_test - u_ref
            err_t = tf.sqrt(tf.reduce_sum(diff**2) * dx)
            errs.append(float(err_t.numpy()))
        errors = np.asarray(errs, dtype=np.float64)
    else:
        u_ref = _to_numpy(sol_ref)
        u_test = _to_numpy(sol_test)
        if u_ref.ndim == 1 and times.size == 1:
            u_ref = u_ref[None, :]
        if u_test.ndim == 1 and times.size == 1:
            u_test = u_test[None, :]
        if u_ref.shape != u_test.shape:
            raise ValueError(
                f"Reference/test snapshot shapes must match, got "
                f"{u_ref.shape} vs {u_test.shape}."
            )
        if u_ref.ndim != 2 or u_ref.shape[0] != times.size:
            raise ValueError(
                "For non-Solution input, expected shape (T, N) with T=len(times)."
            )
        errors = np.sqrt(np.sum((u_test - u_ref) ** 2, axis=-1))

    ax.plot(times, errors, marker="o")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|u_{\mathrm{test}}-u_{\mathrm{ref}}\|_{L^2_x}$")
    return ax


def animate_solution(
    sol: Solution | np.ndarray,
    times: np.ndarray,
    ax=None,
    *,
    title: str | None = None,
    interval: int = 100,
    fps: int = 12,
    dpi: int = 140,
    out: str | Path | None = None,
):
    """Animate ``u(x, t)`` over ``times`` for one solution.

    Parameters
    ----------
    sol : Solution | np.ndarray
        Solution object or raw array of shape ``(T, N)``.
    times : np.ndarray
        Time values of shape ``(T,)``.
    ax : matplotlib axes, optional
    title : str, optional
    interval : int
        Milliseconds per frame in the interactive animation.
    fps : int
        Frame-rate used only when ``out`` is provided.
    dpi : int
        DPI used only when ``out`` is provided.
    out : str | Path, optional
        If provided, saves animation to ``.gif`` or ``.mp4``.
    """

    import matplotlib.animation as mlanim

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    x, u = _as_spacetime_data(sol, times)
    ax = _ensure_ax(ax)

    y_min, y_max = _finite_limits(u)

    (line,) = ax.plot(x, u[0], color="tab:blue", linewidth=2.0)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(title or "Solution movie")

    def update(i: int):
        line.set_ydata(u[i])
        time_text.set_text(f"t={times[i]:.5g}")
        return line, time_text

    anim = mlanim.FuncAnimation(
        ax.figure, update, frames=len(times), interval=interval, blit=True
    )
    if out is not None:
        _save_animation(anim, out=out, fps=fps, dpi=dpi)
    return anim


def animate_comparison(
    sol_ref: Solution | np.ndarray,
    sol_test: Solution | np.ndarray,
    times: np.ndarray,
    axes=None,
    *,
    title: str | None = None,
    interval: int = 100,
    fps: int = 12,
    dpi: int = 140,
    out: str | Path | None = None,
):
    """Animate reference/test curves and their pointwise error.

    ``sol_ref`` and ``sol_test`` must represent the same spatial grid for
    array input. For ``Solution`` input, both are evaluated on ``sol_ref``'s
    grid for direct comparison.
    """

    import matplotlib.animation as mlanim
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    if isinstance(sol_ref, Solution):
        x = sol_ref.grid.x
        u_ref = _solution_snapshots(sol_ref, times)[1]
    else:
        x, u_ref = _as_spacetime_data(sol_ref, times)

    if isinstance(sol_test, Solution):
        if isinstance(sol_ref, Solution):
            x_tf = sol_ref.grid.x_tf
            u_test = []
            for t in times:
                t_tf = tf.constant(float(t), dtype=tf.float64)
                u_test.append(_to_numpy(sol_test(x_tf, t_tf)))
            u_test = np.asarray(u_test, dtype=np.float64)
        else:
            x_tf = tf.constant(x, dtype=tf.float64)
            u_test = []
            for t in times:
                t_tf = tf.constant(float(t), dtype=tf.float64)
                u_test.append(_to_numpy(sol_test(x_tf, t_tf)))
            u_test = np.asarray(u_test, dtype=np.float64)
    else:
        x_test, u_test = _as_spacetime_data(sol_test, times)
        if x_test.shape != x.shape or not np.allclose(x_test, x):
            raise ValueError("Reference/test x-grids must match for array inputs.")

    diff = u_test - u_ref

    if axes is None:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(9.5, 6.8), sharex=True, constrained_layout=True
        )
    else:
        if len(axes) != 2:
            raise ValueError("axes must be a length-2 sequence of matplotlib axes.")
        ax_top, ax_bot = axes
        fig = ax_top.figure

    u_min, u_max = _finite_limits(np.concatenate([u_ref, u_test], axis=0))
    d_min, d_max = _finite_limits(diff)

    (line_ref,) = ax_top.plot(x, u_ref[0], color="black", linewidth=2.0, label="reference")
    (line_test,) = ax_top.plot(
        x, u_test[0], color="tab:orange", linestyle="--", linewidth=2.0, label="test"
    )
    time_text = ax_top.text(0.02, 0.95, "", transform=ax_top.transAxes, va="top")
    ax_top.set_ylabel("u(x, t)")
    ax_top.set_ylim(u_min, u_max)
    ax_top.legend(loc="upper right")
    ax_top.set_title(title or "Solution comparison movie")

    (line_err,) = ax_bot.plot(x, diff[0], color="tab:red", linewidth=2.0)
    ax_bot.axhline(0.0, color="gray", linewidth=1.0)
    ax_bot.set_xlabel("x")
    ax_bot.set_ylabel("test - ref")
    ax_bot.set_ylim(d_min, d_max)

    def update(i: int):
        line_ref.set_ydata(u_ref[i])
        line_test.set_ydata(u_test[i])
        line_err.set_ydata(diff[i])
        time_text.set_text(f"t={times[i]:.5g}")
        return line_ref, line_test, line_err, time_text

    anim = mlanim.FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)
    if out is not None:
        _save_animation(anim, out=out, fps=fps, dpi=dpi)
    return anim


def build_theta_solution(ic, grid, nu: float, alpha: float) -> Solution:
    """Return the Cole-Hopf ``theta`` solution paired with a spectral run."""
    import fracburgers.operators as operators
    from fracburgers.cole_hopf import u_to_theta_0

    if ic.theta_0 is not None:
        theta0_fn = lambda x: ic.theta_0(x, nu=nu, alpha=alpha)
    else:
        theta0_fn = lambda x: u_to_theta_0(ic.u_0(x), alpha=alpha, nu=nu, grid=grid)

    def eval_theta(t):
        return operators.heat_evolve(theta0_fn(grid.x_tf), t=t, nu=nu, grid=grid)

    return Solution(grid, eval_theta)


def theta_mass_diagnostics(theta_sol: Solution, times: np.ndarray) -> dict[str, np.ndarray]:
    """Mass, absolute drift, and relative drift for a theta solution."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    _, theta = _solution_snapshots(theta_sol, times)
    mass = np.sum(theta, axis=1) * float(theta_sol.grid.dx)
    baseline = mass[0]
    abs_error = np.abs(mass - baseline)
    rel_error = abs_error / max(abs(float(baseline)), np.finfo(float).eps)
    return {"mass": mass, "abs_error": abs_error, "rel_error": rel_error}


def save_spectral_report(
    theta_sol: Solution,
    u_sol: Solution,
    times: np.ndarray,
    out: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save the standard theta/u solution report used by ``scripts/solve.py``."""
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    diagnostics = theta_mass_diagnostics(theta_sol, times)
    mass = diagnostics["mass"]
    abs_error = np.maximum(diagnostics["abs_error"], np.finfo(float).tiny)
    rel_error = np.maximum(diagnostics["rel_error"], np.finfo(float).tiny)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)
    if title:
        fig.suptitle(title)

    plot_spacetime(theta_sol, times, ax=axes[0, 0], cmap="viridis")
    axes[0, 0].set_title("theta spacetime")
    plot_spacetime(u_sol, times, ax=axes[0, 1])
    axes[0, 1].set_title("u spacetime")
    plot_evolution(theta_sol, times, ax=axes[1, 0])
    axes[1, 0].set_title("theta snapshots")
    axes[1, 0].set_ylabel("theta(x, t)")
    plot_evolution(u_sol, times, ax=axes[1, 1])
    axes[1, 1].set_title("u snapshots")

    axes[0, 2].plot(times, mass, marker="o")
    axes[0, 2].axhline(float(mass[0]), color="black", linewidth=1.0, alpha=0.6)
    axes[0, 2].set_title("theta mass")
    axes[0, 2].set_xlabel("t")
    axes[0, 2].set_ylabel(r"$\int \theta\,dx$")

    axes[1, 2].semilogy(times, abs_error, marker="o", label="absolute")
    axes[1, 2].semilogy(times, rel_error, marker="s", label="relative")
    axes[1, 2].set_title("theta mass drift")
    axes[1, 2].set_xlabel("t")
    axes[1, 2].set_ylabel("error")
    axes[1, 2].legend()

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_alpha_snapshot_grid(
    alphas: list[float],
    u_sols: list[Solution],
    times: np.ndarray,
    out: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Save a grid of u(x,t) snapshots: rows=times, columns=alpha values.

    Parameters
    ----------
    alphas : list[float]
        Alpha values, one per column.
    u_sols : list[Solution]
        Corresponding solved ``u`` solutions, same order as ``alphas``.
    times : np.ndarray
        Snapshot times, one per row.
    out : path
        Output PNG path.
    """
    import matplotlib.pyplot as plt

    if len(alphas) != len(u_sols):
        raise ValueError("alphas and u_sols must have the same length")

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(times), len(alphas)

    # Evaluate all snapshots up front: u_data[col][row] = (N,) array
    u_data: list[list[np.ndarray]] = []
    for sol in u_sols:
        t_tf = tf.constant(times[:, None], dtype=tf.float64)
        snaps = _to_numpy(sol.sample(t_tf))  # (T, N)
        u_data.append([snaps[ti] for ti in range(n_rows)])

    x = u_sols[0].grid.x

    # Shared y-limits per row for direct shape comparison across alpha
    row_ylims: list[tuple[float, float]] = []
    for ti in range(n_rows):
        vals = np.concatenate([u_data[ai][ti] for ai in range(n_cols)])
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.05 * (hi - lo), 1e-6)
        row_ylims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 2.4 * n_rows),
        constrained_layout=True,
        sharex=True,
    )
    # Normalise to always 2-D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    if title:
        fig.suptitle(title, fontsize=11)

    for ai, alpha in enumerate(alphas):
        axes[0, ai].set_title(rf"$\alpha={alpha:g}$", fontsize=10)
        for ti, t in enumerate(times):
            ax = axes[ti, ai]
            ax.plot(x, u_data[ai][ti], color="tab:blue", linewidth=1.4)
            ax.set_ylim(*row_ylims[ti])
            ax.grid(True, alpha=0.3)
            if ai == 0:
                ax.set_ylabel(f"t={t:g}", fontsize=9)
            if ti == n_rows - 1:
                ax.set_xlabel("x", fontsize=9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_theta_u_movie(
    theta_sol: Solution,
    u_sol: Solution,
    times: np.ndarray,
    out: str | Path,
    *,
    title: str | None = None,
    fps: int = 12,
    dpi: int = 120,
    interval: int = 100,
) -> Path:
    """Save a side-by-side movie of theta and u over the same time frames."""
    import matplotlib.animation as mlanim
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times must contain at least one value")

    x_theta, theta = _solution_snapshots(theta_sol, times)
    x_u, u = _solution_snapshots(u_sol, times)
    theta_min, theta_max = _finite_limits(theta)
    u_min, u_max = _finite_limits(u)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    if title:
        fig.suptitle(title)

    (theta_line,) = axes[0].plot(x_theta, theta[0], color="tab:green", linewidth=2.0)
    axes[0].set_xlim(float(x_theta[0]), float(x_theta[-1]))
    axes[0].set_ylim(theta_min, theta_max)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\theta(x,t)$")
    axes[0].set_title("theta")

    (u_line,) = axes[1].plot(x_u, u[0], color="tab:blue", linewidth=2.0)
    axes[1].set_xlim(float(x_u[0]), float(x_u[-1]))
    axes[1].set_ylim(u_min, u_max)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x,t)")
    axes[1].set_title("u")

    time_text = axes[1].text(0.02, 0.95, "", transform=axes[1].transAxes, va="top")

    def update(i: int):
        theta_line.set_ydata(theta[i])
        u_line.set_ydata(u[i])
        time_text.set_text(f"t={times[i]:.5g}")
        return theta_line, u_line, time_text

    anim = mlanim.FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)
    out_path = _save_animation(anim, out=out, fps=fps, dpi=dpi)
    plt.close(fig)
    return out_path


def solution_errors(
    sol_ref: Solution,
    sol_test: Solution,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-time L2 and Linf errors on the reference grid."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    x = sol_ref.grid.x_tf
    dx = float(sol_ref.grid.dx)
    l2_errors: list[float] = []
    linf_errors: list[float] = []

    for t in times:
        t_tf = tf.constant(float(t), dtype=tf.float64)
        diff = sol_test(x, t_tf) - sol_ref(x, t_tf)
        l2_errors.append(float(tf.sqrt(tf.reduce_sum(diff**2) * dx).numpy()))
        linf_errors.append(float(tf.reduce_max(tf.abs(diff)).numpy()))

    return np.asarray(l2_errors), np.asarray(linf_errors)


def _json_array(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _finite_stat(values: np.ndarray, fn) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(fn(finite))


def save_solution_comparison(
    sol_ref: Solution,
    sol_test: Solution,
    times: np.ndarray,
    out_dir: str | Path,
    *,
    title: str = "Solution comparison",
    prefix: str = "comparison",
    ref_label: str = "reference",
    test_label: str = "test",
    config: dict | None = None,
) -> tuple[Path, Path, dict]:
    """Save comparison plot plus JSON metrics for two ``Solution`` objects."""
    import matplotlib.pyplot as plt

    times = np.asarray(times, dtype=np.float64).reshape(-1)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / f"{prefix}.png"
    metrics_path = out_path / f"{prefix}_metrics.json"

    l2_errors, linf_errors = solution_errors(sol_ref, sol_test, times)
    metrics = {
        "times": _json_array(times),
        "l2_errors": _json_array(l2_errors),
        "linf_errors": _json_array(linf_errors),
        "max_l2_error": _finite_stat(l2_errors, np.max),
        "max_linf_error": _finite_stat(linf_errors, np.max),
        "mean_l2_error": _finite_stat(l2_errors, np.mean),
        "config": config or {},
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    fig.suptitle(title)

    plot_spacetime(sol_ref, times, ax=axes[0, 0])
    axes[0, 0].set_title(ref_label)
    plot_spacetime(sol_test, times, ax=axes[0, 1])
    axes[0, 1].set_title(test_label)

    snapshot_t = float(times[-1])
    plot_snapshot(sol_ref, snapshot_t, ax=axes[1, 0], color="black", label=ref_label)
    plot_snapshot(sol_test, snapshot_t, ax=axes[1, 0], color="tab:orange", linestyle="--", label=test_label)
    axes[1, 0].set_title(f"Final snapshot t={snapshot_t:g}")
    axes[1, 0].legend()

    axes[1, 1].semilogy(times, np.maximum(l2_errors, np.finfo(float).tiny), marker="o", label="L2")
    axes[1, 1].semilogy(times, np.maximum(linf_errors, np.finfo(float).tiny), marker="s", label="Linf")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("error")
    axes[1, 1].set_title("Error against reference")
    axes[1, 1].legend()

    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return fig_path, metrics_path, metrics


def save_convergence_plot(
    Ns: np.ndarray,
    times: np.ndarray,
    l2_errors: np.ndarray,
    linf_errors: np.ndarray,
    out: str | Path,
    *,
    title: str = "Reference convergence",
) -> Path:
    """Save a compact spectral-convergence diagnostic plot."""
    import matplotlib.pyplot as plt

    Ns = np.asarray(Ns, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    fig.suptitle(title)

    axes[0].loglog(Ns, np.max(l2_errors, axis=1), marker="o", label="max L2")
    axes[0].loglog(Ns, np.max(linf_errors, axis=1), marker="s", label="max Linf")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("error")
    axes[0].set_title("Worst-time error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for j, t in enumerate(times):
        axes[1].loglog(Ns, l2_errors[:, j], marker="o", label=f"t={t:g}")
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("L2 error")
    axes[1].set_title("L2 error by time")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_reference_visuals(
    x: np.ndarray,
    times: np.ndarray,
    u_ref: np.ndarray,
    u_test: np.ndarray,
    l2_errors: np.ndarray,
    linf_errors: np.ndarray,
    out: str | Path,
    *,
    title: str = "Reference solution",
    test_label: str = "spectral",
) -> Path:
    """Save spacetime/snapshot/error visuals for a trusted reference solution."""
    import matplotlib.pyplot as plt

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diff = np.asarray(u_test) - np.asarray(u_ref)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0), constrained_layout=True)
    fig.suptitle(title)

    plot_spacetime(u_ref, times, ax=axes[0, 0], x=x)
    axes[0, 0].set_title("reference spacetime")
    plot_spacetime(u_test, times, ax=axes[0, 1], x=x)
    axes[0, 1].set_title(f"{test_label} spacetime")
    plot_spacetime(diff, times, ax=axes[0, 2], x=x)
    axes[0, 2].set_title(f"{test_label} - reference")

    for idx in [0, len(times) // 2, len(times) - 1]:
        axes[1, 0].plot(x, u_ref[idx], label=f"ref t={times[idx]:g}")
        axes[1, 1].plot(x, u_test[idx], label=f"{test_label} t={times[idx]:g}")
        axes[1, 2].plot(x, diff[idx], label=f"t={times[idx]:g}")

    axes[1, 0].set_title("reference snapshots")
    axes[1, 1].set_title(f"{test_label} snapshots")
    axes[1, 2].set_title("pointwise error snapshots")
    for ax in axes[1]:
        ax.set_xlabel("x")
        ax.legend()

    inset = axes[1, 2].inset_axes([0.55, 0.55, 0.4, 0.38])
    inset.semilogy(times, np.maximum(l2_errors, np.finfo(float).tiny), marker="o", label="L2")
    inset.semilogy(times, np.maximum(linf_errors, np.finfo(float).tiny), marker="s", label="Linf")
    inset.set_title("errors", fontsize=9)
    inset.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
