# fracburgers

Version 0.1.0 — numerical solvers and PINN for the fractional nonlinear
Burgers equation

```
u_t + D^α_x [(D^{1-α} u)^2 / 2] = ν u_{xx},   0 < α < 1
```

via the fractional Cole–Hopf transform `u = -2ν D^α ln θ`, which
reduces the equation to the classical heat equation `θ_t = ν θ_{xx}`
(up to gauge).

## Design

Both the spectral solver and the PINN expose the same interface:
they consume an ``InitialCondition`` (a TF callable `x -> u_0(x)`)
and produce a ``Solution`` (a TF callable `(x, t) -> u(x, t)`,
differentiable end-to-end via autograd).

```python
ic       = initial_conditions.gaussian()
solver   = SpectralSolver(grid, nu=0.1, alpha=0.5)
sol_spec = solver.solve(ic)            # Solution

pinn     = train(HeatPINN(), ic, grid, nu=0.1, alpha=0.5, t_max=1.0, ...)
sol_pinn = to_solution(pinn, grid, nu=0.1, alpha=0.5)   # Solution

# Same interface for both:
sol_spec(x_query, t_query)
sol_pinn(x_query, t_query)
```

This means downstream code (plots, error metrics, comparison scripts)
is solver-agnostic — the comparison study is literally the same
callable applied to the same query points.

### Pipeline split

The PINN learns θ on the heat equation; the spectral side handles the
fractional algebra. Concretely:

- **PINN** is trained on `θ_t - ν θ_{xx} = 0` plus an IC penalty —
  pure heat equation, no D^α in the loss.
- The Cole–Hopf inversion `u = -2ν D^α ln θ` runs at query time inside
  ``to_solution``, using the same spectral D^α as the reference solver.

Training the NN against the heat residual (rather than the full
fractional Burgers residual) keeps the NN's job (smooth θ
approximation) cleanly separate from the spectral pipeline's job
(computing D^α). Error attribution is then clean: the
`scripts/compare_methods.py` log-θ residual spectrum plot decomposes
the PINN θ error in Fourier space and weights it by `|k|^α` to predict
the contribution to the u error.

### PINN architecture

- Inputs are the periodic features `(sin(π x / L), cos(π x / L), t)`,
  so periodicity in x is enforced by construction.
- The output layer is **linear** in pre-activation, and the model
  returns `θ = exp(linear)`. This guarantees `θ > 0` for all weights
  and lets `log θ` be read off as the pre-activation algebraically —
  no `log()` call in the Cole–Hopf inverse, no float32 underflow →
  `log(0)` NaN trap.
- The IC loss is MSE on `log θ` (pre-activation vs. closed-form or
  spectral `log θ_0`), which is naturally relative across the wide
  dynamic range of `θ_0`.
- The training step is compiled into a single ``tf.function`` (random
  sampling + forward + backward) so the hot loop never round-trips to
  Python.

### Gauge / k=0 mode

`I^α` and `D^α` annihilate constants, so the forward Cole–Hopf
transform fixes a gauge by removing `mean(u_0)` and the back-transform
reconstructs only the zero-mean component of u. Both ``SpectralSolver``
and ``to_solution`` track `mean(u_0)` separately and add it back at
query time.

## Layout

```
src/fracburgers/
  grid.py                  FourierGrid: x, k, np + tf views
  operators.py             FFT operators (TF): D^α, I^α, heat propagator
  cole_hopf.py             u ↔ θ_0 transforms (TF), log-space variants
  initial_conditions.py    InitialCondition presets: u_0 + closed-form θ_0 / log θ_0
  interpolation.py         Trig interpolation on the spectral grid (TF)
  solution.py              Solution wrapper: TF callable (x, t) -> u
  spectral.py              SpectralSolver -> Solution
  pinn.py                  HeatPINN, train, to_solution -> Solution
  references.py            Closed-form cosine-mode reference solutions
  result_naming.py         Parameter-driven output-directory naming
  viz.py                   Plotting helpers, take Solution objects

scripts/
  solve.py                 Run the spectral solver, save plot.
  train_pinn.py            Train + checkpoint a PINN.
  compare_methods.py       Spectral vs. PINN at matched query points,
                           plus the log-θ residual spectrum diagnostic.
  plot_reference.py        Plot/animate the cosine-mode reference solution.
  reference_convergence.py Spectral convergence study (alpha × k × N × t).

tests/
  test_operators.py        FFT operators vs. analytic identities
  test_cole_hopf.py        Round-trip + closed-form-vs-spectral
  test_interpolation.py    Trig interp correctness + autograd
  test_solution.py         Solution interface, both solvers, classical limit
  test_references.py       Cosine-mode closed form vs. spectral pipeline
```

## Conventions

- Spatial grid: uniform on `[-L, L)` with `N` points, periodic.
- All FFT operators assume periodicity; this is a wide-domain
  approximation to the problem on `R`. Pick `L` so that the solution
  decays to ~0 well before `|x| = L`.
- Wavenumber convention: `k = 2π * fftfreq(N, dx)`.
- Fractional symbol: `D^α ↔ (ik)^α` (principal branch); at `k=0` the
  symbol is set to 0.
- **Spectral pipeline** uses `tf.float64` / `tf.complex128`
  throughout. Fractional derivatives high-pass-amplify, so single
  precision is too noisy for D^α.
- **PINN** defaults to `tf.float32` for ~32× speedup on T4-class GPUs
  (selectable via `--dtype`). The PINN output is cast back to float64
  before entering the spectral D^α, so the fractional algebra stays in
  double precision regardless of training dtype.
- All operators take ``tf.Tensor`` and return ``tf.Tensor``. Inputs
  may carry leading batch dimensions; FFTs act on the last axis.

## Closed-form θ_0 / log θ_0 shortcut

When ``InitialCondition.theta_0`` (or ``log_theta_0``) is provided —
e.g. ``sine``, where `I^α[sin x] = sin(x - α π / 2)` admits a closed
form — the spectral solver and the PINN IC loss use it directly and
skip their own ``I^α`` step. This removes one source of spectral
error from the validation pipeline. For ICs without a closed form
(e.g. Gaussian), the solver computes ``θ_0`` spectrally from
``ic.u_0``.

The log-form variant `u_to_log_theta_0` is preferred internally
wherever the consumer ultimately wants `log θ` (PINN IC loss,
Cole–Hopf inverse), since it avoids the wasted `exp` → `log`
round-trip and the float32 underflow risk.

## Install

```
pip install -e .
pip install -e .[dev]    # adds pytest, ruff
```

## References

- Mao & Karniadakis, *A fractional nonlinear Burgers equation*,
  J. Comput. Phys. (2017).
- Podlubny, *Fractional Differential Equations*, Academic Press, 1999.
