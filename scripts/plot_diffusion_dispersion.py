#!/usr/bin/env python3
"""Visualize diffusion vs. dispersion in the linear fractional equation.

For the single-mode linearized problem starting from cos(kx):

    u(x, t) = exp(-γ t) · cos(kx - c·t)

where
    γ = ν |k|^α cos(α π/2)   [decay rate  — diffusion]
    c = ν |k|^α sin(α π/2)   [phase speed — dispersion]

At α→0: γ→ν, c→0  (pure diffusion, no drift)
At α→1: γ→0, c→ν  (pure dispersion, no decay)

Grid: rows = α values, columns = k values.
Each panel overlays time snapshots (light→dark = early→late).
Dashed gray lines show the ±envelope at t=0 and t=t_max.
Each panel is annotated with γ and c so the balance is explicit.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "fracburgers-cache"
_MPL_DIR = _CACHE_DIR / "matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as mlanim
import matplotlib.pyplot as plt
import numpy as np

from fracburgers.result_naming import get_output_dir


def csv_floats(text: str) -> list[float]:
    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument(
        "--alpha-list",
        type=csv_floats,
        default=csv_floats("0.1,0.3,0.5,0.7,0.9"),
        metavar="A1,A2,...",
        help="alpha values — one row per value",
    )
    parser.add_argument(
        "--k-list",
        type=csv_floats,
        default=csv_floats("1,2,4"),
        metavar="K1,K2,...",
        help="wavenumbers — one column per value",
    )
    parser.add_argument("--L", type=float, default=float(np.pi), help="half-domain length")
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument(
        "--times",
        type=csv_floats,
        default=csv_floats("0.0,0.25,0.5,1.0,2.0"),
        metavar="T1,T2,...",
        help="snapshot times to overlay",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (auto-generated from params if not specified)")
    parser.add_argument("--movie", type=Path, default=None, metavar="FILE")
    parser.add_argument("--movie-fps", type=int, default=15)
    parser.add_argument("--movie-t-max", type=float, default=None)
    parser.add_argument("--movie-frames", type=int, default=120)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    checks = [
        (args.nu > 0.0, "--nu must be positive"),
        (all(0.0 < a < 1.0 for a in args.alpha_list), "all --alpha-list values must be in (0,1)"),
        (all(k > 0.0 for k in args.k_list), "all --k-list values must be positive"),
        (args.L > 0.0, "--L must be positive"),
        (args.N >= 4, "--N must be at least 4"),
        (all(t >= 0.0 for t in args.times), "--times must be nonneg"),
        (args.movie_fps > 0, "--movie-fps must be positive"),
        (args.movie_frames > 0, "--movie-frames must be positive"),
    ]
    for ok, msg in checks:
        if not ok:
            raise SystemExit(msg)


def _gamma_c(k: float, alpha: float, nu: float) -> tuple[float, float]:
    """Decay rate γ and phase speed c for a single mode."""
    gamma = nu * k**alpha * np.cos(alpha * np.pi / 2)
    c = nu * k**alpha * np.sin(alpha * np.pi / 2)
    return gamma, c


def _u(x: np.ndarray, t: float, k: float, alpha: float, nu: float) -> np.ndarray:
    gamma, c = _gamma_c(k, alpha, nu)
    return np.exp(-gamma * t) * np.cos(k * x - c * t)


def _make_grid(n_rows: int, n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.6 * n_rows),
        constrained_layout=True,
        sharex=True,
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    return fig, axes


def main() -> None:
    args = parse_args()
    validate(args)

    # Auto-generate output directory if not specified
    if args.out_dir is None:
        args.out_dir = get_output_dir(
            Path("results"),
            "plot_diffusion_dispersion",
            {
                "nu": args.nu,
                "__tags": ["nu"],
            },
        )
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(args.alpha_list)
    ks = sorted(args.k_list)
    times = np.asarray(sorted(args.times), dtype=np.float64)
    x = np.linspace(-args.L, args.L, args.N, endpoint=False)

    n_rows, n_cols = len(alphas), len(ks)
    cmap = plt.get_cmap("plasma")
    t_colors = [cmap(i / max(len(times) - 1, 1)) for i in range(len(times))]

    # ── static snapshot grid ────────────────────────────────────────────────
    fig, axes = _make_grid(n_rows, n_cols)
    fig.suptitle(
        rf"Linear fractional eq. $u=e^{{-\gamma t}}\cos(kx-ct)$,  $\nu={args.nu:g}$"
        "\n"
        r"$\gamma=\nu|k|^\alpha\cos(\alpha\pi/2)$ (diffusion)   "
        r"$c=\nu|k|^\alpha\sin(\alpha\pi/2)$ (dispersion)",
        fontsize=10,
    )

    for ai, alpha in enumerate(alphas):
        for ki, k in enumerate(ks):
            ax = axes[ai, ki]
            gamma, c = _gamma_c(k, alpha, args.nu)

            for ti, t in enumerate(times):
                ax.plot(x, _u(x, t, k, alpha, args.nu), color=t_colors[ti], linewidth=1.3)

            # ±envelope at t=0 (light) and t=t_max (dark)
            for t_env, ls in [(times[0], "--"), (times[-1], ":")]:
                env = np.exp(-gamma * t_env)
                for sign in (+1, -1):
                    ax.axhline(sign * env, color="gray", linewidth=0.9, linestyle=ls, alpha=0.65)

            ax.set_ylim(-1.15, 1.15)
            ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.35)
            ax.grid(True, alpha=0.22)
            ax.text(
                0.97,
                0.96,
                rf"$\gamma={gamma:.3f}$" + "\n" + rf"$c={c:.3f}$",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
            )
            if ai == 0:
                ax.set_title(f"k = {k:g}", fontsize=10)
            if ki == 0:
                ax.set_ylabel(rf"$\alpha={alpha:g}$", fontsize=9)
            if ai == n_rows - 1:
                ax.set_xlabel("x", fontsize=9)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=float(times[0]), vmax=float(times[-1]))
    )
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.06,
        label="t",
    )

    out_path = args.out_dir / "diffusion_dispersion.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved diffusion/dispersion plot to {out_path}")

    # ── movie ────────────────────────────────────────────────────────────────
    if args.movie is None:
        return

    t_max = args.movie_t_max if args.movie_t_max is not None else float(times[-1])
    movie_times = np.linspace(0.0, t_max, args.movie_frames, dtype=np.float64)

    fig2, axes2 = _make_grid(n_rows, n_cols)
    fig2.suptitle(rf"Linear fractional eq., $\nu={args.nu:g}$", fontsize=10)

    # (ai, ki) → (wave_line, env_hi_line, env_lo_line, gamma, k, alpha)
    panels: dict[tuple[int, int], tuple] = {}
    for ai, alpha in enumerate(alphas):
        for ki, k in enumerate(ks):
            ax = axes2[ai, ki]
            gamma, c = _gamma_c(k, alpha, args.nu)
            (wave,) = ax.plot(x, _u(x, 0.0, k, alpha, args.nu), color="tab:blue", linewidth=1.5)
            env0 = np.exp(-gamma * 0.0)
            (eh,) = ax.plot(x, np.full_like(x, env0), color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
            (el,) = ax.plot(x, np.full_like(x, -env0), color="gray", linewidth=0.9, linestyle="--", alpha=0.7)
            ax.set_ylim(-1.15, 1.15)
            ax.axhline(0.0, color="black", linewidth=0.4, alpha=0.35)
            ax.grid(True, alpha=0.22)
            if ai == 0:
                ax.set_title(f"k = {k:g}", fontsize=10)
            if ki == 0:
                ax.set_ylabel(rf"$\alpha={alpha:g}$", fontsize=9)
            if ai == n_rows - 1:
                ax.set_xlabel("x", fontsize=9)
            panels[(ai, ki)] = (wave, eh, el, gamma, k, alpha)

    time_label = fig2.text(0.5, 0.005, "t = 0.000", ha="center", fontsize=10)

    def update(frame: int):
        t = movie_times[frame]
        artists: list = [time_label]
        for (ai, ki), (wave, eh, el, gamma, k, alpha) in panels.items():
            wave.set_ydata(_u(x, t, k, alpha, args.nu))
            env = np.exp(-gamma * t)
            eh.set_ydata(np.full_like(x, env))
            el.set_ydata(np.full_like(x, -env))
            artists += [wave, eh, el]
        time_label.set_text(f"t = {t:.4g}")
        return artists

    anim = mlanim.FuncAnimation(
        fig2, update, frames=args.movie_frames, interval=1000 // args.movie_fps, blit=True
    )

    movie_path = Path(args.movie)
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    ext = movie_path.suffix.lower()
    if ext == ".gif":
        writer = mlanim.PillowWriter(fps=args.movie_fps)
    elif ext in {".mp4", ".m4v"}:
        writer = mlanim.FFMpegWriter(fps=args.movie_fps)
    else:
        raise SystemExit(f"Unsupported movie extension: {ext!r}. Use .gif or .mp4.")
    anim.save(movie_path, writer=writer, dpi=120)
    plt.close(fig2)
    print(f"Saved movie to {movie_path}")


if __name__ == "__main__":
    main()
