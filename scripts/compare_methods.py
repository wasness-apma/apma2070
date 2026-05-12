#!/usr/bin/env python3
"""Compare a trained PINN checkpoint against a high-resolution spectral solve."""

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
import tensorflow as tf

matplotlib.use("Agg")

from fracburgers import initial_conditions
from fracburgers.grid import FourierGrid
from fracburgers.pinn import HeatPINN, to_solution
from fracburgers.result_naming import get_output_dir
from fracburgers.spectral import SpectralSolver
from fracburgers.viz import animate_comparison, save_solution_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic", choices=sorted(initial_conditions.REGISTRY), default="gaussian")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--n-times", type=int, default=21)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--N-ref", type=int, default=2048)
    parser.add_argument("--L", type=float, default=20.0)
    parser.add_argument("--pinn-checkpoint", type=Path, default=Path("checkpoints/heat_pinn.keras"))
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (auto-generated from params if not specified)")
    parser.add_argument("--movie", type=Path, default=None)
    parser.add_argument("--movie-fps", type=int, default=12)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    checks = [
        (0.0 < args.alpha <= 1.0, "--alpha must satisfy 0 < alpha <= 1"),
        (args.nu > 0.0, "--nu must be positive"),
        (args.t_max >= 0.0, "--t-max must be nonnegative"),
        (args.n_times >= 2, "--n-times must be at least 2"),
        (args.N >= 2, "--N must be at least 2"),
        (args.N_ref >= 2, "--N-ref must be at least 2"),
        (args.L > 0.0, "--L must be positive"),
        (args.movie_fps > 0, "--movie-fps must be positive"),
    ]
    for ok, message in checks:
        if not ok:
            raise SystemExit(message)


def load_pinn_model(path: Path) -> HeatPINN:
    if not path.exists():
        raise FileNotFoundError(
            f"PINN checkpoint not found at {path}. "
            "Run scripts/train_pinn.py first or pass --pinn-checkpoint."
        )
    model = tf.keras.models.load_model(
        path,
        custom_objects={"HeatPINN": HeatPINN},
        compile=False,
    )
    if not isinstance(model, HeatPINN):
        raise TypeError(f"Loaded {type(model).__name__}, expected HeatPINN.")
    return model


def main() -> None:
    args = parse_args()
    validate_args(args)

    # Auto-generate output directory if not specified
    if args.out_dir is None:
        args.out_dir = get_output_dir(
            Path("results"),
            "compare",
            {
                "ic": args.ic,
                "alpha": args.alpha,
                "nu": args.nu,
                "N": args.N,
                "__tags": ["ic", "alpha", "nu", "N"],
            },
        )
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    ic = initial_conditions.get(args.ic)
    grid = FourierGrid.make(N=args.N, L=args.L)
    grid_ref = FourierGrid.make(N=args.N_ref, L=args.L)
    times = np.linspace(0.0, args.t_max, num=args.n_times, dtype=np.float64)

    sol_pinn = to_solution(load_pinn_model(args.pinn_checkpoint), grid=grid, nu=args.nu, alpha=args.alpha)
    sol_ref = SpectralSolver(grid=grid_ref, nu=args.nu, alpha=args.alpha).solve(ic)

    title = (
        "PINN vs spectral "
        f"(IC={ic.name}, alpha={args.alpha:g}, nu={args.nu:g}, "
        f"N={args.N}, N_ref={args.N_ref})"
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    fig_path, metrics_path, metrics = save_solution_comparison(
        sol_ref,
        sol_pinn,
        times,
        args.out_dir,
        title=title,
        prefix="pinn_vs_spectral",
        ref_label="spectral reference",
        test_label="PINN",
        config=config,
    )

    print(f"Saved comparison figure to {fig_path}")
    print(f"Saved metrics to {metrics_path}")
    if metrics["max_l2_error"] is None or metrics["max_linf_error"] is None:
        print("Error summary: no finite error values found")
    else:
        print(
            "Error summary: "
            f"max L2={metrics['max_l2_error']:.3e}, "
            f"max Linf={metrics['max_linf_error']:.3e}"
        )

    if args.movie is not None:
        animate_comparison(
            sol_ref,
            sol_pinn,
            times,
            title=title,
            out=args.movie,
            fps=args.movie_fps,
            dpi=120,
        )
        print(f"Saved comparison movie to {args.movie}")


if __name__ == "__main__":
    main()
