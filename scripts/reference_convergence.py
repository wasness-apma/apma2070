#!/usr/bin/env python3
"""Spectral convergence study for the tough cosine-mode reference (b ≈ a).

Fixed defaults: a=1, b=0.99, nu=0.1  →  |r(0)| ≈ 0.87, slow series convergence.

Sweeps alpha (--alpha-list) and k (--k-list) to show how L² and L∞ errors
vs. N change with time and fractional order.  Produces one convergence figure
per k value (rows = alpha, columns = L² / L∞, lines coloured by t).
"""

from __future__ import annotations

import argparse
import json
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

# Force a light plotting style regardless of user/system defaults.
matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFCFC",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "axes.edgecolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    }
)

from fracburgers.grid import FourierGrid
from fracburgers.references import CosineModeReference
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
    p.add_argument("--a", type=float, default=1.0, help="cosine offset")
    p.add_argument("--b", type=float, default=0.99, help="cosine amplitude (set close to a for tough solution)")
    p.add_argument("--nu", type=float, default=0.1, help="diffusivity")
    p.add_argument("--L", type=float, default=float(np.pi), help="half-domain length")
    p.add_argument(
        "--alpha-list",
        type=_csv_floats,
        default=_csv_floats("0.25,0.5,0.75"),
        metavar="A1,A2,...",
        help="fractional orders to sweep (rows)",
    )
    p.add_argument(
        "--k-list",
        type=_csv_floats,
        default=_csv_floats("1,2"),
        metavar="K1,K2,...",
        help="wavenumbers to sweep (one figure each)",
    )
    p.add_argument(
        "--N-list",
        type=_csv_ints,
        default=_csv_ints("32,64,128,256,512,1024"),
        metavar="N1,N2,...",
        help="grid sizes to test",
    )
    p.add_argument(
        "--times",
        type=_csv_floats,
        default=_csv_floats("0.01,0.1,0.5,1.0"),
        metavar="T1,T2,...",
        help="evaluation times (lines in convergence plot)",
    )
    p.add_argument("--n-terms", type=int, default=300, help="series truncation for reference (needs ~200 for b≈a)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    checks = [
        (args.a > 0.0, "--a must be positive"),
        (0.0 < args.b < args.a, "--b must satisfy 0 < b < a"),
        (args.nu > 0.0, "--nu must be positive"),
        (args.L > 0.0, "--L must be positive"),
        (all(0.0 < al < 1.0 for al in args.alpha_list), "all --alpha-list values must be in (0,1)"),
        (all(k > 0.0 for k in args.k_list), "all --k-list values must be positive"),
        (all(N >= 4 for N in args.N_list), "all --N-list values must be >= 4"),
        (len(set(args.N_list)) == len(args.N_list), "--N-list must not contain duplicates"),
        (all(t > 0.0 for t in args.times), "--times must be strictly positive (t=0 is singular for b≈a)"),
        (args.n_terms >= 1, "--n-terms must be at least 1"),
    ]
    for ok, msg in checks:
        if not ok:
            raise SystemExit(msg)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _r0(a: float, b: float) -> float:
    """Series ratio |r(t=0)|; controls difficulty of the solution."""
    return abs(-b / (a + np.sqrt(a**2 - b**2)))


def _r_of_t(a: float, b: float, k: float, nu: float, t: float) -> float:
    beta = b * np.exp(-nu * k**2 * t)
    return abs(-beta / (a + np.sqrt(a**2 - beta**2)))


def compute_errors(
    sol_ref,
    sol_spec,
    x_tf: tf.Tensor,
    times: np.ndarray,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative (l2, linf) arrays of shape (len(times),).

    Each error is divided by the corresponding norm of the reference solution
    at that time, so diffusion toward zero does not artificially inflate accuracy.
    """
    l2_list, linf_list = [], []
    for t in times:
        t_tf = tf.constant(float(t), dtype=tf.float64)
        u_ref = sol_ref(x_tf, t_tf).numpy()
        diff = sol_spec(x_tf, t_tf).numpy() - u_ref
        ref_l2 = float(np.sqrt(np.sum(u_ref**2) * dx))
        ref_linf = float(np.max(np.abs(u_ref)))
        l2_list.append(float(np.sqrt(np.sum(diff**2) * dx)) / max(ref_l2, np.finfo(float).tiny))
        linf_list.append(float(np.max(np.abs(diff))) / max(ref_linf, np.finfo(float).tiny))
    return np.asarray(l2_list), np.asarray(linf_list)


def run_sweep(
    a: float,
    b: float,
    k: float,
    nu: float,
    alpha: float,
    n_terms: int,
    Ns: np.ndarray,
    times: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run convergence sweep for one (k, alpha) pair.

    Returns
    -------
    l2_mat, linf_mat : shape (len(Ns), len(times))
        Relative L² and L∞ errors.
    x_dense : shape (dense_N,)
        Spatial grid used for all evaluations.
    u_ref_snaps : shape (len(times), dense_N)
        Reference solution snapshots.
    u_spec_snaps_by_N : dict[int, ndarray of shape (len(times), dense_N)]
        Spectral snapshots for each N, evaluated on the dense grid.
    """
    dense_N = max(8192, int(Ns[-1]) * 4)
    dense_grid = FourierGrid.make(N=dense_N, L=L)

    ref = CosineModeReference(a=a, b=b, k=k, nu=nu, alpha=alpha, n_terms=n_terms)
    ic = ref.initial_condition()
    sol_ref = ref.reference_solution(dense_grid)

    # Reference snapshots (computed once)
    u_ref_snaps = np.stack([
        sol_ref(dense_grid.x_tf, tf.constant(float(t), dtype=tf.float64)).numpy()
        for t in times
    ])

    l2_rows, linf_rows = [], []
    u_spec_snaps_by_N: dict[int, np.ndarray] = {}
    for N in Ns:
        sol_spec = SpectralSolver(
            grid=FourierGrid.make(N=int(N), L=L), nu=nu, alpha=alpha
        ).solve(ic)
        l2, linf = compute_errors(sol_ref, sol_spec, dense_grid.x_tf, times, dense_grid.dx)
        l2_rows.append(l2)
        linf_rows.append(linf)
        u_spec_snaps_by_N[int(N)] = np.stack([
            sol_spec(dense_grid.x_tf, tf.constant(float(t), dtype=tf.float64)).numpy()
            for t in times
        ])

    return (
        np.asarray(l2_rows),
        np.asarray(linf_rows),
        dense_grid.x,
        u_ref_snaps,
        u_spec_snaps_by_N,
    )


def estimate_orders(Ns: np.ndarray, errors: np.ndarray) -> list[float | None]:
    """Local convergence orders from consecutive N pairs."""
    orders: list[float | None] = [None]
    for i in range(1, len(Ns)):
        e0, e1 = float(errors[i - 1]), float(errors[i])
        if e0 <= 0.0 or e1 <= 0.0 or not np.isfinite(e0 + e1) or e1 >= e0:
            orders.append(None)
        else:
            orders.append(float(np.log(e0 / e1) / np.log(float(Ns[i]) / float(Ns[i - 1]))))
    return orders


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_convergence_grid(
    alphas: list[float],
    Ns: np.ndarray,
    times: np.ndarray,
    results: list[tuple[np.ndarray, np.ndarray]],
    k: float,
    a: float,
    b: float,
    nu: float,
    out: Path,
) -> Path:
    """One figure: rows=alpha, cols=(L², L∞), lines=t.

    results[ai] = (l2_mat, linf_mat) each of shape (len(Ns), len(times)).
    """
    n_rows = len(alphas)
    # Use distinct, high-contrast colors per time curve.
    t_colors = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # magenta
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
    ]
    t_styles = ["-", "--", ":", "-."]
    t_markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11.0, 3.0 * n_rows),
        constrained_layout=True,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    r0 = _r0(a, b)
    fig.suptitle(
        rf"Spectral convergence  —  $a={a:g},\ b={b:g},\ k={k:g},\ \nu={nu:g}$"
        rf"   $|r(0)|\approx{r0:.3f}$",
        fontsize=11,
    )

    for ai, alpha in enumerate(alphas):
        l2_mat, linf_mat = results[ai]
        for col, (mat, norm_tex) in enumerate([(l2_mat, r"rel. $L^2$"), (linf_mat, r"rel. $L^\infty$")]):
            ax = axes[ai, col]
            ax.set_facecolor("#FCFCFC")
            for ti, t in enumerate(times):
                rt = _r_of_t(a, b, k, nu, t)
                ax.loglog(
                    Ns,
                    mat[:, ti],
                    marker=t_markers[ti % len(t_markers)],
                    markersize=4,
                    linewidth=1.8,
                    linestyle=t_styles[ti % len(t_styles)],
                    color=t_colors[ti % len(t_colors)],
                    label=rf"$t={t:g}$  $|r|\approx{rt:.2f}$",
                )
            ax.set_ylim(bottom=1e-14)
            ax.grid(True, which="both", color="#B0B0B0", alpha=0.35, linewidth=0.6)
            if ai == 0:
                ax.set_title(norm_tex, fontsize=10)
            if ai == n_rows - 1:
                ax.set_xlabel("$N$", fontsize=9)
            # row label on left column only
            if col == 0:
                ax.set_ylabel(rf"$\alpha={alpha:g}$" + "\nrelative error", fontsize=9)
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9, facecolor="white")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", edgecolor="white")
    plt.close(fig)
    return out


def save_snapshot_comparison(
    x: np.ndarray,
    times: np.ndarray,
    u_ref: np.ndarray,
    u_spec_lo: np.ndarray,
    u_spec_hi: np.ndarray,
    N_lo: int,
    N_hi: int,
    k: float,
    alpha: float,
    a: float,
    b: float,
    nu: float,
    out: Path,
) -> Path:
    """Grid of time-slice snapshots: reference vs. coarsest and finest N.

    Layout: rows = time slices, 3 columns = (u curves, pointwise error N_lo,
    pointwise error N_hi).
    """
    n_times = len(times)
    ref_color = "#2980C7"
    err_color = "#5A0306"

    fig, axes = plt.subplots(
        n_times,
        3,
        figsize=(13.0, 2.6 * n_times),
        constrained_layout=True,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    if n_times == 1:
        axes = axes[np.newaxis, :]

    r0 = _r0(a, b)
    fig.suptitle(
        rf"Reference vs spectral snapshots  —  $a={a:g},\ b={b:g},\ k={k:g},\ "
        rf"\nu={nu:g},\ \alpha={alpha:g}$   $|r(0)|\approx{r0:.3f}$",
        fontsize=10,
    )
    axes[0, 0].set_title("$u(x,t)$", fontsize=10)
    axes[0, 1].set_title(rf"error  $N={N_lo}$", fontsize=10)
    axes[0, 2].set_title(rf"error  $N={N_hi}$", fontsize=10)

    for ti, t in enumerate(times):
        ref = u_ref[ti]
        lo = u_spec_lo[ti]
        hi = u_spec_hi[ti]

        ax_u, ax_lo, ax_hi = axes[ti, 0], axes[ti, 1], axes[ti, 2]
        ax_u.set_facecolor("#FCFCFC")
        ax_lo.set_facecolor("#FCFCFC")
        ax_hi.set_facecolor("#FCFCFC")

        ax_u.plot(x, ref, color=ref_color, linewidth=1.8, alpha = 0.5, label="reference")
        ax_u.plot(x, lo, color=err_color, linewidth=1.2, linestyle="--", alpha=0.95, label=f"N={N_lo}")
        ax_u.plot(x, hi, color=err_color, linewidth=1.2, linestyle=":", alpha=0.95, label=f"N={N_hi}")
        ax_u.set_ylabel(f"t={t:g}", fontsize=9)
        ax_u.grid(True, color="#B0B0B0", alpha=0.28)
        if ti == 0:
            ax_u.legend(fontsize=7, loc="upper right", framealpha=0.92, facecolor="white")

        for ax_e, u_spec in [(ax_lo, lo), (ax_hi, hi)]:
            err = u_spec - ref
            ref_inf = max(float(np.max(np.abs(ref))), np.finfo(float).tiny)
            ax_e.plot(x, err / ref_inf, color=err_color, linewidth=1.4)
            ax_e.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.45)
            ax_e.grid(True, color="#B0B0B0", alpha=0.28)
            ax_e.set_ylabel("$(u_{spec}-u_{ref})/\|u_{ref}\|_\infty$", fontsize=7)

        if ti == n_times - 1:
            for ax in axes[ti]:
                ax.set_xlabel("$x$", fontsize=9)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white", edgecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_table(
    k: float,
    alpha: float,
    Ns: np.ndarray,
    times: np.ndarray,
    l2_mat: np.ndarray,
    linf_mat: np.ndarray,
) -> None:
    print(f"\n  k={k:g}  alpha={alpha:g}")
    header = f"  {'N':>6}" + "".join(f"  relL2(t={t:g})   relLinf(t={t:g}) " for t in times)
    print(header)
    for i, N in enumerate(Ns):
        row = f"  {int(N):>6}"
        for ti in range(len(times)):
            row += f"  {l2_mat[i, ti]:.3e}    {linf_mat[i, ti]:.3e}  "
        print(row)
    # asymptotic order from last two N for max-over-time error
    l2_max = np.max(l2_mat, axis=1)
    linf_max = np.max(linf_mat, axis=1)
    orders_l2 = estimate_orders(Ns, l2_max)
    orders_linf = estimate_orders(Ns, linf_max)
    print(f"  {'N':>6}  order_L2(max-t)  order_Linf(max-t)")
    for i, N in enumerate(Ns):
        o2 = "—" if orders_l2[i] is None else f"{orders_l2[i]:6.2f}"
        oi = "—" if orders_linf[i] is None else f"{orders_linf[i]:6.2f}"
        print(f"  {int(N):>6}  {o2:>16}  {oi:>18}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alphas = sorted(args.alpha_list)
    ks = sorted(args.k_list)
    Ns = np.asarray(sorted(args.N_list), dtype=np.int64)
    times = np.asarray(sorted(args.times), dtype=np.float64)

    print(f"a={args.a}, b={args.b}, nu={args.nu},  |r(0)|={_r0(args.a, args.b):.4f}")
    print(f"alphas: {alphas}")
    print(f"ks:     {ks}")
    print(f"Ns:     {Ns.tolist()}")
    print(f"times:  {times.tolist()}")

    all_results: dict = {}

    for k in ks:
        k_results: list[tuple[np.ndarray, np.ndarray]] = []
        for alpha in alphas:
            print(f"\nRunning k={k:g}, alpha={alpha:g} ...", flush=True)
            l2_mat, linf_mat, x_dense, u_ref_snaps, u_spec_by_N = run_sweep(
                a=args.a,
                b=args.b,
                k=k,
                nu=args.nu,
                alpha=alpha,
                n_terms=args.n_terms,
                Ns=Ns,
                times=times,
                L=args.L,
            )
            k_results.append((l2_mat, linf_mat))
            print_table(k, alpha, Ns, times, l2_mat, linf_mat)

            snap_path = args.out_dir / f"snapshots_k{k:g}_alpha{alpha:g}.png"
            save_snapshot_comparison(
                x=x_dense,
                times=times,
                u_ref=u_ref_snaps,
                u_spec_lo=u_spec_by_N[int(Ns[0])],
                u_spec_hi=u_spec_by_N[int(Ns[-1])],
                N_lo=int(Ns[0]),
                N_hi=int(Ns[-1]),
                k=k,
                alpha=alpha,
                a=args.a,
                b=args.b,
                nu=args.nu,
                out=snap_path,
            )
            print(f"  Saved {snap_path}")

        out_path = args.out_dir / f"convergence_k{k:g}.png"
        save_convergence_grid(alphas, Ns, times, k_results, k, args.a, args.b, args.nu, out_path)
        print(f"\nSaved {out_path}")

        all_results[f"k{k:g}"] = [
            {
                "alpha": alpha,
                "rel_l2_by_N_time": l2.tolist(),
                "rel_linf_by_N_time": linf.tolist(),
                "rel_l2_max_over_time": np.max(l2, axis=1).tolist(),
                "rel_linf_max_over_time": np.max(linf, axis=1).tolist(),
                "order_rel_l2_max": estimate_orders(Ns, np.max(l2, axis=1)),
                "order_rel_linf_max": estimate_orders(Ns, np.max(linf, axis=1)),
                "|r_of_t|": [_r_of_t(args.a, args.b, k, args.nu, float(t)) for t in times],
            }
            for alpha, (l2, linf) in zip(alphas, k_results)
        ]

    report = {
        "config": {
            "a": args.a,
            "b": args.b,
            "nu": args.nu,
            "|r(0)|": _r0(args.a, args.b),
            "L": args.L,
            "n_terms": args.n_terms,
            "alphas": alphas,
            "ks": ks,
            "N_list": Ns.tolist(),
            "times": times.tolist(),
        },
        "results": all_results,
    }
    report_path = args.out_dir / "convergence.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
