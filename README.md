# fracburgers

Numerical solvers and PINN for the fractional nonlinear Burgers equation

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
is solver-agnostic — the comparison study in Task 6 is literally the
same callable applied to the same query points.

## Layout

```
src/fracburgers/
  grid.py                  FourierGrid: x, k, np + tf views
  operators.py             FFT operators (TF): D^α, I^α, heat propagator
  cole_hopf.py             u ↔ θ_0 transforms (TF)
  initial_conditions.py    InitialCondition presets: u_0 + closed-form θ_0
  interpolation.py         Trig interpolation on the spectral grid (TF)
  solution.py              Solution wrapper: TF callable (x, t) -> u
  spectral.py              SpectralSolver -> Solution
  pinn.py                  HeatPINN, train, to_solution -> Solution
  viz.py                   Plotting helpers, take Solution objects

scripts/
  solve.py                 Run the spectral solver, save plot.
  train_pinn.py            Train + checkpoint a PINN.
  compare_methods.py       Spectral vs. PINN at matched query points.

tests/
  test_operators.py        FFT operators vs. analytic identities
  test_cole_hopf.py        Round-trip + closed-form-vs-spectral
  test_interpolation.py    Trig interp correctness + autograd
  test_solution.py         Solution interface, both solvers, classical limit
```

## Conventions

- Spatial grid: uniform on `[-L, L)` with `N` points, periodic.
- All FFT operators assume periodicity; this is a wide-domain
  approximation to the problem on `R`. Pick `L` so that the solution
  decays to ~0 well before `|x| = L`.
- Wavenumber convention: `k = 2π * fftfreq(N, dx)`.
- Fractional symbol: `D^α ↔ (ik)^α` (principal branch); at `k=0` the
  symbol is set to 0.
- Real fields: `tf.float64`. Spectral coefficients: `tf.complex128`.
  Fractional derivatives high-pass-amplify; single precision is too
  noisy.
- All operators take ``tf.Tensor`` and return ``tf.Tensor``. Inputs
  may carry leading batch dimensions; FFTs act on the last axis.

## Closed-form θ_0 shortcut

When ``InitialCondition.theta_0`` is provided (e.g. ``sine``, where
`I^α[sin x] = sin(x - α π / 2)` admits a closed form), the spectral
solver uses it directly and skips its own ``I^α`` step. This removes
one source of spectral error from the validation pipeline. For ICs
without a closed form (e.g. Gaussian), the solver computes ``θ_0``
spectrally from ``ic.u_0``.

## Install

```
pip install -e .
pip install -e .[dev]    # adds pytest, ruff
```

## References

- Mao & Karniadakis, *A fractional nonlinear Burgers equation*,
  J. Comput. Phys. (2017).
- Podlubny, *Fractional Differential Equations*, Academic Press, 1999.
