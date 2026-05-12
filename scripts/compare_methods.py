#!/usr/bin/env python3
"""Compare a trained PINN checkpoint against a high-resolution spectral solve."""

from __future__ import annotations

import argparse
import json
import math
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


def _csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(tok.strip()) for tok in text.split(",") if tok.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


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
    parser.add_argument(
        "--hidden-layers",
        type=_csv_ints,
        default=None,
        metavar="W1,W2,...",
        help="PINN hidden widths for weights-only checkpoints (auto-read from report.json when omitted)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=None,
        help="PINN activation for weights-only checkpoints (auto-read from report.json when omitted)",
    )
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


def _read_arch_from_report(path: Path) -> tuple[tuple[int, ...] | None, str | None, float | None, str | None]:
    report_path = path.with_name("report.json")
    if not report_path.exists():
        return None, None, None, None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None, None

    config = report.get("config", {})
    hidden_layers = config.get("hidden_layers")
    activation = config.get("activation")
    L = config.get("L")
    dtype = config.get("dtype")
    if isinstance(hidden_layers, list) and all(isinstance(w, int) for w in hidden_layers):
        return (
            tuple(hidden_layers),
            activation if isinstance(activation, str) else None,
            float(L) if L is not None else None,
            dtype if isinstance(dtype, str) else None,
        )
    return None, None, None, None


def load_pinn_model(
    path: Path,
    *,
    hidden_layers: tuple[int, ...] | None = None,
    activation: str | None = None,
    L: float | None = None,
    dtype: str | None = None,
) -> HeatPINN:
    if not path.exists():
        raise FileNotFoundError(
            f"PINN checkpoint not found at {path}. "
            "Run scripts/train_pinn.py first or pass --pinn-checkpoint."
        )

    # Full serialized model path (.keras / legacy full .h5 model)
    if path.suffix == ".keras" or (path.suffix == ".h5" and not path.name.endswith(".weights.h5")):
        model = tf.keras.models.load_model(
            path,
            custom_objects={"HeatPINN": HeatPINN},
            compile=False,
        )
        if not isinstance(model, HeatPINN):
            raise TypeError(f"Loaded {type(model).__name__}, expected HeatPINN.")
        return model

    # Weights-only path (.weights.h5): reconstruct architecture from report.json.
    report_layers, report_activation, report_L, report_dtype = _read_arch_from_report(path)
    hidden_layers = hidden_layers or report_layers
    activation = activation or report_activation or "tanh"
    L = L if L is not None else (report_L if report_L is not None else math.pi)
    dtype = dtype or report_dtype or "float32"
    if hidden_layers is None:
        raise ValueError(
            "Weights checkpoint detected but model architecture is unknown. "
            "Either place report.json next to the weights file (from train_pinn.py), "
            "or pass --hidden-layers and optionally --activation."
        )

    fdtype = tf.float32 if dtype == "float32" else tf.float64
    model = HeatPINN(hidden_layers=hidden_layers, activation=activation, L=L, dtype=dtype)
    _ = model(tf.zeros((1, 2), dtype=fdtype))
    model.load_weights(str(path))
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

    # Cole–Hopf forward strips the spatial mean of u_0; restore it in
    # the PINN-side reconstruction so it matches the spectral reference.
    u0_mean = float(tf.reduce_mean(ic.u_0(grid.x_tf)).numpy())

    sol_pinn = to_solution(
        load_pinn_model(
            args.pinn_checkpoint,
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            L=args.L,
        ),
        grid=grid,
        nu=args.nu,
        alpha=args.alpha,
        u0_mean=u0_mean,
    )
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
