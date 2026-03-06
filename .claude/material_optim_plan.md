# Plan: Extending Optimization to Include Material Parameters

## Current State

The existing optimizer (`multi_optim.py`) solves a **3-variable, 2-objective** problem:

| Variable | Bounds | Type |
|---|---|---|
| `L` (column length) | 0.5–500 m | Operating/design |
| `u_super` (superficial velocity) | 5e-5–0.0099 m/s | Operating |
| `des_threshold` | 0.02–0.5 | Operating |

**Objectives:** minimize SEC (J/mol), maximize Productivity (mol/m²/s)

All material properties are frozen from the Jiang 2020 study:
- `rho_p = 681 kg/m³`, `d_p = 0.8 mm`, `epsilon = 0.35`
- Temkin isotherm: `A = 0.008428 L/mg`, `B = 2.4118 kJ/mol`
- PSO kinetics: `k2` picked from 3 discrete concentration-matched experiments

---

## Proposed Material Decision Variables

| Parameter | Symbol | Current Value | Proposed Bounds | Rationale |
|---|---|---|---|---|
| Particle diameter | `d_p` | 0.8 mm | 0.1–5.0 mm | Smaller particles → faster kinetics + higher ΔP; larger → slower mass transfer but lower ΔP. Strong trade-off with SEC. |
| Bed porosity | `epsilon` | 0.35 | 0.25–0.50 | Depends on particle shape/packing; affects capacity ratio, pressure drop, and interstitial velocity. |
| Sorbent density | `rho_p` | 681 kg/m³ | 400–1500 kg/m³ | Varies across adsorbent families (ion-exchange resins ~700, zeolites ~1200, activated alumina ~1500). Directly scales solid-phase capacity. |
| Isotherm capacity (Temkin B) | `B` | 2.4118 kJ/mol | 0.5–10 kJ/mol | Controls steepness of isotherm; lower B → higher equilibrium loading. Represents "how good" the sorbent's thermodynamics are. |
| Isotherm affinity (Temkin A) | `A` | 0.008428 L/mg | 0.001–0.1 L/mg | Controls onset of adsorption; higher A → adsorption starts at lower C. Represents binding affinity. |
| Kinetics rate constant | `k_s` | ~0.0236 g/mg/min | 0.005–0.5 g/mg/min | Faster kinetics → shorter breakthrough, better bed utilization. Depends on particle size, pore structure, surface chemistry. |

### Which to include?

**Recommended Tier 1 (high impact, physically meaningful):**
- `d_p` — directly affects both Ergun ΔP and mass transfer; strong interaction with `u_super`
- `k_s` — controls kinetic limitation; Sobol analysis likely shows high sensitivity
- `B` (Temkin) — controls equilibrium capacity, the thermodynamic ceiling

**Tier 2 (moderate impact, physically constrained):**
- `epsilon` — narrower feasible range, but important for pressure drop
- `rho_p` — scales capacity linearly; if exploring different sorbent families

**Tier 3 (can be derived or is less independent):**
- `A` (Temkin) — often correlated with `B`; may add noise without new insight

---

## Implementation Plan

### Step 1: Extend `ColumnOptimizationProblem` to Accept Material Variables

Currently the constructor freezes material params from `Study`. Change to:

```python
class ColumnOptimizationProblem(ElementwiseProblem):
    def __init__(self, study, C_in, bounds_dict, loss_fraction=0.01):
        # bounds_dict = {
        #     'L': (0.5, 500),
        #     'u_super': (5e-5, 0.0099),
        #     'des_threshold': (0.02, 0.5),
        #     # Material params (optional — if key absent, use study default)
        #     'd_p': (1e-4, 5e-3),
        #     'k_s': (0.005, 0.5),       # in g/mg/min, converted internally
        #     'B_temkin': (0.5, 10.0),    # kJ/mol
        # }
        self.var_names = list(bounds_dict.keys())
        n_var = len(self.var_names)
        xl = [bounds_dict[k][0] for k in self.var_names]
        xu = [bounds_dict[k][1] for k in self.var_names]
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu)
        ...
```

In `_evaluate`, unpack decision variables by name:

```python
def _evaluate(self, x, out, *args, **kwargs):
    vals = dict(zip(self.var_names, x))
    L = vals['L']
    u_super = vals['u_super']
    des_threshold = vals['des_threshold']

    # Override material params if they are decision variables
    d_p = vals.get('d_p', self.d_p_default)
    k_s = vals.get('k_s', self.k_s_default)
    epsilon = vals.get('epsilon', self.epsilon_default)
    rho_p = vals.get('rho_p', self.rho_p_default)

    # Rebuild isotherm if A or B are decision variables
    B_temkin = vals.get('B_temkin', None)
    A_temkin = vals.get('A_temkin', None)
    isotherm = self._build_isotherm(A_temkin, B_temkin)

    # Build ColumnParameters directly (bypass study.to_column_parameter)
    params = ColumnParameters(
        L=L, u_inter=u_super/epsilon, C_in=self.C_in,
        k_s=convert_k2(k_s), rho_p=rho_p, epsilon=epsilon,
        isotherm=isotherm
    )
    ...
```

### Step 2: Update Pressure Drop Calculation

The Ergun equation already uses `self.d_p` and `self.epsilon`. These need to use the per-evaluation values instead of stored constants:

```python
def _pressure_drop_per_unit_length(self, u_super, d_p, epsilon):
    term1 = (150 * self.mu * (1-epsilon)**2 * u_super) / (epsilon**3 * d_p**2)
    term2 = (1.75 * self.rho_water * (1-epsilon) * u_super**2) / (epsilon**3 * d_p)
    return term1 + term2
```

Similarly `_li_recovered` needs `rho_p` and `epsilon` as arguments.

### Step 3: Handle Kinetics–Concentration Coupling

Currently `k_s` is looked up from discrete experiments matched to `C_in`. When `k_s` is a decision variable, skip the lookup and use the optimized value directly. This represents "what if we had a sorbent with this rate constant at this concentration."

### Step 4: Build Isotherm on the Fly

When `A_temkin` or `B_temkin` are decision variables, construct a new `TemkinIsothermFit` each evaluation:

```python
def _build_isotherm(self, A_override=None, B_override=None):
    A = A_override if A_override is not None else self.A_default
    B = B_override if B_override is not None else self.B_default
    return TemkinIsothermFit(A=A, B=B, T=self.T, A_units="L/mg", B_units="kJ/mol")
```

### Step 5: Update CSV Output and Pareto Plotting

The output CSV header and column stack need to reflect the variable number of decision variables:

```python
header = ",".join(problem.var_names + ["SEC_J_per_mol", "Productivity_mol_per_m2_per_s"])
pareto_data = np.column_stack([res.X, res.F[:, 0], -res.F[:, 1]])
```

### Step 6: Add CLI Arguments for Material Bounds

```python
parser.add_argument("--optimize-materials", nargs="*",
    choices=["d_p", "k_s", "B_temkin", "A_temkin", "epsilon", "rho_p"],
    help="Material parameters to include as decision variables")
parser.add_argument("--dp-bounds", nargs=2, type=float, default=[1e-4, 5e-3])
parser.add_argument("--ks-bounds", nargs=2, type=float, default=[0.005, 0.5])
# etc.
```

---

## Considerations

### Population Size
More decision variables → larger search space. Rule of thumb for NSGA2: `pop_size ≈ 10 * n_var`. For 6 variables, use `pop_size=60` minimum (current: 40).

### Correlation Constraints
Some material parameters are physically correlated:
- Smaller `d_p` → faster kinetics (`k_s` increases). If both are free, the optimizer may find unrealistic combos (tiny particles with slow kinetics). Options:
  1. **Ignore it** — treat the Pareto front as "what properties would an ideal sorbent need?" (material screening perspective)
  2. **Add a correlation constraint** — e.g., `k_s ∝ 1/d_p²` from mass-transfer scaling, penalize deviations
  3. **Replace `k_s` with an effectiveness factor** — keep intrinsic kinetics fixed, vary `d_p` and compute effective `k_s` from internal diffusion limitations

Option 1 is simplest and most useful for identifying target material properties. Option 3 is most physically rigorous.

### Interpretation of Results
The Pareto front will now answer: *"What combination of column design AND sorbent properties minimizes SEC and maximizes productivity?"* This is useful for:
- **Material screening** — which sorbent properties matter most for performance?
- **Design targets** — what `d_p`, `k_s`, `B` should a new sorbent aim for?
- **Sensitivity** — how much does the Pareto front shift when material params change?

### Computational Cost
Each model evaluation takes ~0.1–1s. With 6 variables and pop_size=60, expect ~60 evals/generation × ~100 generations = ~6000 evaluations. At 0.5s each → ~50 minutes. The planned vmap vectorization would cut this significantly.

---

## Suggested Phased Approach

**Phase 1:** Add `d_p` and `k_s` only (5 total variables). These have the strongest interaction with pressure drop and breakthrough dynamics. Run and compare Pareto fronts.

**Phase 2:** Add `B_temkin` (6 variables). This opens up the thermodynamic dimension. Analyze which isotherm properties the optimizer exploits.

**Phase 3 (optional):** Add `epsilon` and `rho_p` if Phase 1–2 results suggest they matter. Consider correlation constraints between `d_p` and `k_s`.

---

## Files to Modify

| File | Changes |
|---|---|
| `Model/diffrax_based/multi_optim.py` | Extend problem class, CLI args, CSV output |
| `utils/Dataclasses.py` | May need `TemkinIsothermFit` constructor flexibility (unit-aware rebuild) |
| No new files needed | |
