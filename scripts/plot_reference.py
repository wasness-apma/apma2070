#!/usr/bin/env python3
"""Plot and animate u(x, t) for the cosine-mode reference solution.

Produces:
  <out-dir>/reference_snapshots.png   -- grid: rows=times, cols=alpha values
  <out-dir>/reference_movie.{gif,mp4} -- animation (if --movie is set; uses first alpha)
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
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from fracburgers.grid import FourierGrid
from fracburgers.references import CosineModeReference
from fracburgers.viz import animate_solution


def csv_floats(text: str) -> list[float]:
    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nu", type=float, default=0.1, help="diffusivity")
    parser.add_argument("--a", type=float, default=2.0, help="cosine offset (must exceed |b|)")
    parser.add_argument("--b", type=float, default=1.0, help="cosine amplitude")
    parser.add_argument("--k", type=float, default=1.0, help="wavenumber")
    parser.add_argument("--L", type=float, default=float(np.pi), help="half-domain length")
    parser.add_argument("--N", type=int, default=512, help="grid points for evaluation")
    parser.add_argument("--n-terms", type=int, default=120, help="series truncation")
    parser.add_argument(
        "--alpha-list",
        type=csv_floats,
        default=csv_floats("0.25,0.5,0.75"),
        metavar="A1,A2,...",
        help="comma-separated alpha values (columns)",
    )
    parser.add_argument(
        "--times",
        type=csv_floats,
        default=csv_floats("0.0,0.1,0.5,1.0,2.0"),
        metavar="T1,T2,...",
        help="comma-separated snapshot times (rows)",
    )
    parser.add_argument(
        "--movie",
        type=Path,
        default=None,
        metavar="FILE",
        help="save animation to .gif or .mp4 (uses first --alpha-list value)",
    )
    parser.add_argument("--movie-fps", type=int, default=15)
    parser.add_argument(
        "--movie-t-max",
        type=float,
        default=None,
        help="end time for movie frames (defaults to max of --times)",
    )
    parser.add_argument("--movie-frames", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    checks = [
        (args.nu > 0.0, "--nu must be positive"),
        (args.a > abs(args.b), "--a must exceed |b|"),
        (args.k > 0.0, "--k must be positive"),
        (args.L > 0.0, "--L must be positive"),
        (args.N >= 4, "--N must be at least 4"),
        (args.n_terms >= 1, "--n-terms must be at least 1"),
        (all(0.0 < al < 1.0 for al in args.alpha_list), "all --alpha-list values must be in (0,1)"),
        (len(args.alpha_list) >= 1, "--alpha-list must have at least one value"),
        (all(t >= 0.0 for t in args.times), "--times must be nonneg"),
        (len(args.times) >= 1, "--times must have at least one value"),
        (args.movie_fps > 0, "--movie-fps must be positive"),
        (args.movie_frames > 0, "--movie-frames must be positive"),
    ]
    for ok, msg in checks:
        if not ok:
            raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    validate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(args.alpha_list)
    times = np.asarray(sorted(args.times), dtype=np.float64)
    grid = FourierGrid.make(N=args.N, L=args.L)
    x = grid.x  # (N,) numpy, used as the aligned plotting coordinate

    # Pre-compute u[alpha_idx, time_idx, x_idx].
    # To align the n=1 mode across alpha, use aligned coordinate ξ and evaluate
    # u(ξ - α π/(2k), t), so cos(k(ξ - shift) + α π/2) = cos(kξ).
    u_data: list[np.ndarray] = []
    for alpha in alphas:
        shift = alpha * np.pi / (2.0 * args.k)
        x_eval_tf = grid.x_tf - tf.constant(shift, dtype=tf.float64)
        ref = CosineModeReference(
            a=args.a, b=args.b, k=args.k, nu=args.nu, alpha=alpha, n_terms=args.n_terms,
        )
        snapshots = []
        for t in times:
            t_tf = tf.constant(float(t), dtype=tf.float64)
            snapshots.append(ref._u_on_grid(x_eval_tf, t_tf).numpy())
        u_data.append(np.stack(snapshots, axis=0))  # (T, N)

    n_rows = len(times)
    n_cols = len(alphas)

    # Shared y-limits per row so shapes are comparable across alpha
    row_ylims = []
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
    # Ensure axes is always 2-D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        rf"Reference $u(x,t)=a+b\cos(\omega x)$ solution: "
        f"a={args.a:g}, b={args.b:g}, k={args.k:g}, ν={args.nu:g}",
        fontsize=11,
    )

    for ai, alpha in enumerate(alphas):
        shift = alpha * np.pi / (2.0 * args.k)
        axes[0, ai].set_title(rf"$\alpha={alpha:g}$  (eval at $x-{shift/np.pi:.3g}\pi$)", fontsize=10)
        for ti, t in enumerate(times):
            ax = axes[ti, ai]
            ax.plot(x, u_data[ai][ti], color="tab:blue", linewidth=1.4)
            ax.set_ylim(*row_ylims[ti])
            ax.grid(True, alpha=0.3)
            if ai == 0:
                ax.set_ylabel(f"t={t:g}", fontsize=9)
            if ti == n_rows - 1:
                ax.set_xlabel(r"aligned coordinate $\xi$", fontsize=9)

    snap_path = args.out_dir / "reference_snapshots.png"
    fig.savefig(snap_path, dpi=160)
    plt.close(fig)
    print(f"Saved snapshot grid to {snap_path}")

    # -- movie (first alpha only) ---------------------------------------------
    if args.movie is not None:
        alpha0 = alphas[0]
        ref0 = CosineModeReference(
            a=args.a, b=args.b, k=args.k, nu=args.nu, alpha=alpha0, n_terms=args.n_terms,
        )
        sol0 = ref0.reference_solution(grid)
        t_max = args.movie_t_max if args.movie_t_max is not None else float(times[-1])
        movie_times = np.linspace(0.0, t_max, args.movie_frames, dtype=np.float64)
        animate_solution(
            sol0,
            movie_times,
            title=(
                rf"Reference u(x,t): a={args.a:g}, b={args.b:g}, k={args.k:g}, "
                rf"$\alpha={alpha0:g}$, ν={args.nu:g}"
            ),
            fps=args.movie_fps,
            out=args.movie,
        )
        print(f"Saved movie to {args.movie}")


if __name__ == "__main__":
    main()
