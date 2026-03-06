# Plan: vmap-Accelerated Column Model & Optimizer

## Context

Currently `multi_optim.py` evaluates each NSGA2 individual **sequentially** via `ElementwiseProblem`, calling `run_model` ~2000 times per optimization. By switching to `jax.vmap`, the entire population (40 individuals) is evaluated **simultaneously** per generation.

## Already Done (by user)

- `SpatialDiscretisation.x0/x_final` → non-static pytree leaves
- `binop` validation check removed
- `@jax.jit(static_argnums=2)` removed from `column_ode`
- `@eqx.filter_jit` added to `run_model`
- Switched to `Kvaerno5()` + `PIDController`
- `get_finish_state` partially converted to JAX (has bugs to fix)

## Files Modified

| File | Summary |
|------|---------|
| `Model/diffrax_based/diffrax_column_model.py` | Fix `get_finish_state`; remove numpy/cache side effects from `run_model` |
| `Model/diffrax_based/multi_optim.py` | Switch `ElementwiseProblem` → `Problem`; use `jax.vmap(run_model)` |

---

## Steps

### Step 1: Fix `get_finish_state` in `diffrax_column_model.py`

Current code (lines 123-133) has two issues:
- **Typo**: `solution.ys.colmulative_out` → `solution.ys.cumulative_out`
- **Dead code**: `t = np.array(solution.ts)` on line 126 is unused and imports numpy unnecessarily

Fix to pure JAX:
```python
def get_finish_state(solution):
    valid = jnp.isfinite(solution.ts)
    idx = jnp.sum(valid) - 1
    return (
        solution.ts[idx],
        solution.ys.C.vals[idx],
        solution.ys.n.vals[idx],
        solution.ys.cumulative_out[idx],
    )
```

### Step 2: Remove `cache.clear_cache()` from `run_model`

Line 230: `diffrax.diffeqsolve._cached.clear_cache()` is a side effect that breaks JIT tracing. Remove it — with `filter_jit`, the compilation cache is managed by JAX/XLA.

### Step 3: Remove `import numpy as np`

After steps 1-2, numpy is no longer used in `diffrax_column_model.py`. Remove the import (line 9).

### Step 4: Vectorize `multi_optim.py`

**4a.** Switch from `ElementwiseProblem` to `Problem`:
```python
from pymoo.core.problem import Problem
```

**4b.** In `__init__`, pre-extract constants and build vmapped function:
```python
class ColumnOptimizationProblem(Problem):
    def __init__(self, study, C_in, L_bounds, u_s_bounds,
                 des_threshold_bounds, loss_fraction=0.01):
        super().__init__(n_var=3, n_obj=2, xl=[...], xu=[...])
        self.epsilon = study.column_experiments.column_properties.porosity
        self.rho_p = study.sorbent_properties.density_si
        self.d_p = study.sorbent_properties.particle_diameter_si
        self.C_in = C_in
        self.loss_fraction = loss_fraction
        self.mu = 1.002e-3
        self.rho_water = 1000

        kin = study.get_kinetics_experiment_from_concentration(C_in)
        self.k_s = kin.kinetics_params.k2_si
        self.base_isotherm = study.isotherm.isotherm_fit

        self._vmapped_run = jax.vmap(
            lambda params, dt: run_model(params, loss_fraction, dt),
            in_axes=(0, 0),
        )
```

**4c.** New `_evaluate` handles the whole population at once:
```python
def _evaluate(self, X, out, *args, **kwargs):
    N = X.shape[0]
    L_vals = jnp.array(X[:, 0])
    u_super_vals = jnp.array(X[:, 1])
    des_thresh_vals = jnp.array(X[:, 2])

    u_inter_vals = u_super_vals / self.epsilon
    params_batch = ColumnParameters(
        u_inter=u_inter_vals,
        k_s=jnp.full(N, self.k_s),
        epsilon=jnp.full(N, self.epsilon),
        C_in=jnp.full(N, self.C_in),
        L=L_vals,
        rho_p=jnp.full(N, self.rho_p),
        isotherm=self.base_isotherm,
    )

    (t_ads, C_ads, n_ads), (t_des, C_des, n_des), frac_lost = \
        self._vmapped_run(params_batch, des_thresh_vals)

    sec, li_recovered = self._sec_batch(u_super_vals, n_ads, n_des, t_des)
    productivity = li_recovered / t_des

    out["F"] = np.column_stack([np.array(sec), -np.array(productivity)])
```

**4d.** Vectorized SEC computation:
```python
def _sec_batch(self, u_super, n_ads_final, n_des_final, t_des):
    term1 = 150 * self.mu * (1-self.epsilon)**2 * u_super / (self.epsilon**3 * self.d_p**2)
    term2 = 1.75 * self.rho_water * (1-self.epsilon) * u_super**2 / (self.epsilon**3 * self.d_p)
    dP_dL = term1 + term2
    pumping_power = dP_dL * u_super * t_des

    n_start = jnp.mean(n_ads_final, axis=1)
    n_end = jnp.mean(n_des_final, axis=1)
    li_recovered = (n_start - n_end) * (1-self.epsilon) * self.rho_p

    return pumping_power / li_recovered, li_recovered
```

**4e.** Add `n_max_gen` parameter to `run_optimization()`.

---

## Key Details

| Concern | Resolution |
|---------|------------|
| `SpatialDiscretisation.x0/x_final` | Non-static leaves — vary freely across vmap |
| `get_finish_state` uses numpy | Fix to pure JAX with `jnp.isfinite` indexing |
| `cache.clear_cache()` side effect | Remove — incompatible with JIT |
| `desorption_threshold` varies per individual | Passed as second vmap axis |
| Isotherm shared across population | Scalar-field pytree; vmap broadcasts correctly |
| Events with vmap | diffrax events are pure JAX functions; should trace through vmap |
| **Fallback if events + vmap fail** | "Soft events": add `stopped` flag to `ColumnState`, multiply derivatives by `(1 - stopped)` |

---

## Verification

1. Fix typo and run `run_model` for a single parameter set — confirm it produces valid results
2. `jax.vmap(run_model)(batch_of_2_params, batch_of_2_thresholds)` — verify shapes
3. Run optimization and compare Pareto front to previous results