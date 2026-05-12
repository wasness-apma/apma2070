#!/usr/bin/env python
"""Debug script: test PINN plotting with existing model."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fracburgers.grid import FourierGrid
from fracburgers.pinn import HeatPINN, to_solution
from fracburgers.spectral import SpectralSolver
from fracburgers.initial_conditions import get as get_ic
import json

# ─── Config ───────────────────────────────────────────────────────────────
MODEL_PATH = (
    Path(__file__).parent 
    / "results/train_pinn/ic_sine/alpha_0p5/nu_0p1/epochs_20000/model.weights.h5"
)
REPORT_PATH = MODEL_PATH.parent / "report.json"

print(f"Loading model from {MODEL_PATH}")
print(f"Loading config from {REPORT_PATH}")

# Load report to get config
with open(REPORT_PATH) as f:
    report = json.load(f)
    config = report["config"]

print(f"Config: {config}")

# Rebuild model from report
ic_name = config["ic"]
alpha = config["alpha"]
nu = config["nu"]
N = config["N"]
L = config["L"]
t_max = config["t_max"]
hidden_layers = config["hidden_layers"]
activation = config["activation"]

print(f"\nArchitecture: hidden_layers={hidden_layers}, activation={activation}")

# Create grid and initial condition
grid = FourierGrid.make(N=N, L=L)
ic = get_ic(ic_name)

# Build and load model
model = HeatPINN(hidden_layers=hidden_layers, activation=activation)
model.build((None, 2))  # (None, 2) for (x, t) input
model.load_weights(str(MODEL_PATH))
print(f"\nModel loaded. Trainable params: {model.count_params()}")

# Create Solution wrappers
print(f"\nCreating Solution wrappers...")
spec_sol = SpectralSolver(grid=grid, nu=nu, alpha=alpha).solve(ic)
pinn_sol = to_solution(model, grid, nu, alpha)

print(f"spec_sol type: {type(spec_sol)}")
print(f"pinn_sol type: {type(pinn_sol)}")

# Test evaluation at various times
snap_times = np.array([0.5, 1.0, 1.5, 2.0])
x_tf = tf.constant(grid.x[:20], dtype=tf.float64)  # Test with first 20 points

print(f"\n{'='*70}")
print("Evaluating at test points:")
print(f"{'='*70}")

for t in snap_times:
    t_tf = tf.constant(float(t), dtype=tf.float64)
    
    try:
        u_spec = spec_sol(x_tf, t_tf).numpy()
        print(f"\nt={t:g}:")
        print(f"  spec: shape={u_spec.shape}, min={np.nanmin(u_spec):g}, max={np.nanmax(u_spec):g}, "
              f"nan={np.sum(np.isnan(u_spec))}, inf={np.sum(np.isinf(u_spec))}")
    except Exception as e:
        print(f"\nt={t:g}: spec_sol FAILED: {e}")
    
    try:
        u_pinn = pinn_sol(x_tf, t_tf).numpy()
        print(f"  pinn: shape={u_pinn.shape}, min={np.nanmin(u_pinn):g}, max={np.nanmax(u_pinn):g}, "
              f"nan={np.sum(np.isnan(u_pinn))}, inf={np.sum(np.isinf(u_pinn))}")
        
        # Check if they're close
        err = u_pinn - u_spec
        print(f"  err:  min={np.nanmin(err):g}, max={np.nanmax(err):g}, l2={np.sqrt(np.mean(err**2)):g}")
    except Exception as e:
        print(f"  pinn_sol FAILED: {e}")
        import traceback
        traceback.print_exc()
