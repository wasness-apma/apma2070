#!/usr/bin/env python
"""Debug: trace NaN source in PINN evaluation."""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fracburgers.grid import FourierGrid
from fracburgers.pinn import HeatPINN
from fracburgers.initial_conditions import get as get_ic
from fracburgers.cole_hopf import theta_to_u
import json

# Load model
MODEL_PATH = (
    Path(__file__).parent 
    / "results/train_pinn/ic_sine/alpha_0p5/nu_0p1/epochs_20000/model.weights.h5"
)
REPORT_PATH = MODEL_PATH.parent / "report.json"

with open(REPORT_PATH) as f:
    report = json.load(f)
    config = report["config"]

alpha = config["alpha"]
nu = config["nu"]
N = config["N"]
L = config["L"]
hidden_layers = config["hidden_layers"]
activation = config["activation"]

print(f"Building model with hidden_layers={hidden_layers}, activation={activation}")
model = HeatPINN(hidden_layers=hidden_layers, activation=activation)
model.build((None, 2))
model.load_weights(str(MODEL_PATH))
print(f"Model loaded successfully")

# Create grid
grid = FourierGrid.make(N=N, L=L)

# Test 1: Direct model evaluation
print("\n" + "="*70)
print("TEST 1: Direct model evaluation")
print("="*70)

x_test = np.array([0.0, 1.0, 2.0])
t_test = np.array([0.5, 0.5, 0.5])

inputs = np.stack([x_test, t_test], axis=-1).astype(np.float64)
print(f"inputs shape: {inputs.shape}, dtype: {inputs.dtype}")
print(f"inputs:\n{inputs}")

try:
    theta_raw = model(inputs)
    print(f"theta_raw shape: {theta_raw.shape}, dtype: {theta_raw.dtype}")
    print(f"theta_raw: {theta_raw.numpy()}")
    theta_vals = theta_raw[:, 0].numpy()
    print(f"theta values: {theta_vals}")
    print(f"  nan: {np.sum(np.isnan(theta_vals))}, inf: {np.sum(np.isinf(theta_vals))}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: theta_to_u conversion
print("\n" + "="*70)
print("TEST 2: theta_to_u conversion")
print("="*70)

# Create a simple theta grid (ones and twos)
theta_grid_ones = np.ones((3, N), dtype=np.float64)
theta_grid_twos = 2.0 * np.ones((3, N), dtype=np.float64)

print(f"theta_grid_ones shape: {theta_grid_ones.shape}")
print(f"theta_grid_ones[0, :5]: {theta_grid_ones[0, :5]}")

try:
    u_ones = theta_to_u(tf.constant(theta_grid_ones), alpha, nu, grid)
    print(f"u_ones shape: {u_ones.shape}")
    u_ones_np = u_ones.numpy()
    print(f"u_ones[0, :5]: {u_ones_np[0, :5]}")
    print(f"  nan: {np.sum(np.isnan(u_ones_np))}, inf: {np.sum(np.isinf(u_ones_np))}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Full to_solution path
print("\n" + "="*70)
print("TEST 3: Checking to_solution internals")
print("="*70)

t_tf = tf.constant(0.5, dtype=tf.float64)
t_batch = tf.reshape(t_tf, [1, 1])
x_batch = tf.broadcast_to(grid.x_tf[None, :], [1, grid.N])
t_full = tf.broadcast_to(t_batch, [1, grid.N])

inputs = tf.stack([x_batch, t_full], axis=-1)  # (1, N, 2)
inputs_flat = tf.reshape(inputs, [-1, 2])      # (N, 2)

print(f"inputs_flat shape: {inputs_flat.shape}")
print(f"inputs_flat[0]: {inputs_flat[0].numpy()}")
print(f"inputs_flat[-1]: {inputs_flat[-1].numpy()}")

try:
    theta_flat = model(inputs_flat)[:, 0]
    print(f"theta_flat shape: {theta_flat.shape}")
    theta_np = theta_flat.numpy()
    print(f"theta_flat[:5]: {theta_np[:5]}")
    print(f"theta_flat[-5:]: {theta_np[-5:]}")
    print(f"  nan: {np.sum(np.isnan(theta_np))}, inf: {np.sum(np.isinf(theta_np))}")
    
    if np.sum(np.isnan(theta_np)) == 0:
        theta_grid = tf.reshape(theta_flat, [1, grid.N])
        u_grid = theta_to_u(theta_grid, alpha, nu, grid)
        u_np = u_grid.numpy()
        print(f"u_grid shape: {u_grid.shape}")
        print(f"u_grid[0, :5]: {u_np[0, :5]}")
        print(f"  nan: {np.sum(np.isnan(u_np))}, inf: {np.sum(np.isinf(u_np))}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
