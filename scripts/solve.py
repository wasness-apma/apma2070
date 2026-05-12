#!/usr/bin/env python3
"""Solve spectral Burgers for one or more alpha values and save theta/u visuals."""

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
import numpy as np

matplotlib.use("Agg")

from fracburgers import initial_conditions
from fracburgers.grid import FourierGrid
from fracburgers.result_naming import get_output_dir
from fracburgers.spectral import SpectralSolver
from fracburgers.viz import (
    build_theta_solution,
    save_alpha_snapshot_grid,
    save_spectral_report,
    save_theta_u_movie,
)


def csv_floats(text: str) -> list[float]:
    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic", choices=sorted(initial_conditions.REGISTRY), default="sine")
    parser.add_argument(
        "--alpha-list",
        type=csv_floats,
        default=csv_floats("0.5"),
        metavar="A1,A2,...",
        help="comma-separated alpha values to solve",
    )
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--n-times", type=int, default=6)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--L", type=float, default=float(np.pi))
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (auto-generated from params if not specified)")
    parser.add_argument("--out", type=Path, default=Path("solution.png"), help="output filename within out-dir")
    parser.add_argument("--movie", type=Path, default=None)
    parser.add_argument("--movie-fps", type=int, default=12)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    checks = [
        (all(al > 0.0 for al in args.alpha_list), "all --alpha-list values must be positive"),
        (args.nu > 0.0, "--nu must be positive"),
        (args.t_max >= 0.0, "--t-max must be nonnegative"),
        (args.n_times >= 2, "--n-times must be at least 2"),
        (args.N >= 2, "--N must be at least 2"),
        (args.L > 0.0, "--L must be positive"),
        (args.movie_fps > 0, "--movie-fps must be positive"),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


def _suffixed(path: Path, alpha: float) -> Path:
    """Insert _alpha<value> before the file extension."""
    return path.with_name(f"{path.stem}_alpha{alpha:g}{path.suffix}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Auto-generate output directory if not specified
    if args.out_dir is None:
        args.out_dir = get_output_dir(
            Path("results"),
            "solve",
            {
                "ic": args.ic,
                "nu": args.nu,
                "N": args.N,
                "__tags": ["ic", "nu", "N"],
            },
        )
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = FourierGrid.make(N=args.N, L=args.L)
    ic = initial_conditions.get(args.ic)
    times = np.linspace(0.0, args.t_max, args.n_times)
    alphas = sorted(args.alpha_list)
    multi = len(alphas) > 1

    u_sols = []
    for alpha in alphas:
        title = f"{args.ic}: alpha={alpha:g}, nu={args.nu:g}, N={args.N}"
        out_path = args.out_dir / (_suffixed(args.out, alpha) if multi else args.out)

        u_sol = SpectralSolver(grid=grid, nu=args.nu, alpha=alpha).solve(ic)
        theta_sol = build_theta_solution(ic, grid=grid, nu=args.nu, alpha=alpha)
        u_sols.append(u_sol)

        report_path = save_spectral_report(theta_sol, u_sol, times, out_path, title=title)
        print(f"Saved solution report to {report_path}")

        if args.movie is not None:
            movie_times = np.linspace(0.0, args.t_max, max(args.n_times, 80))
            movie_out = _suffixed(args.movie, alpha) if multi else args.movie
            movie_path = save_theta_u_movie(
                theta_sol,
                u_sol,
                movie_times,
                args.out_dir / movie_out,
                title=title,
                fps=args.movie_fps,
            )
            print(f"Saved theta/u movie to {movie_path}")

    grid_path = args.out_dir / args.out.with_name(f"{args.out.stem}_alpha_grid{args.out.suffix}")
    grid_title = f"{args.ic}: nu={args.nu:g}, N={args.N}"
    save_alpha_snapshot_grid(alphas, u_sols, times, grid_path, title=grid_title)
    print(f"Saved alpha snapshot grid to {grid_path}")


if __name__ == "__main__":
    main()
