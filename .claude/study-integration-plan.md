# Plan: Integrate Study Class with Diffrax Column Model

## Goal
Add `to_column_parameters` method to `Study` class that creates a `ColumnParameters` object for use with the diffrax column model, with flexible isotherm support.

---

## Current State

### Already Done
- [x] `ColumnParameters` moved to `utils/Dataclasses.py` as `eqx.Module`
- [x] `get_kinetics_experiment_from_concentration(C)` added to Study (finds closest match)
- [x] `to_column_parameter(L, D, C_in, Q)` partially implemented in Study
- [x] Directory renamed: `diffrax based` → `diffrax_based`
- [x] Added `import equinox as eqx` to Dataclasses.py

### TODO
- [ ] Create missing `__init__.py` files for package structure
- [ ] Convert isotherm classes to `eqx.Module` for JAX compatibility
- [ ] Update `ColumnParameters` to accept isotherm module instead of `q_max`/`b`
- [ ] Complete `to_column_parameter` with isotherm integration
- [ ] Fix bugs in `diffrax_column_model.py`

---

## Implementation

### Step 1: Fix Package Structure

**Problem:** Relative imports won't work without proper `__init__.py` files.

**Actions:**
1. Create `/home/janlc/AdsorbentColumnModeling/__init__.py` (root package)
2. Create `/home/janlc/AdsorbentColumnModeling/Model/diffrax_based/__init__.py`

### Step 2: Convert Isotherms to eqx.Module

**File:** `/home/janlc/AdsorbentColumnModeling/utils/Dataclasses.py`

Convert isotherm classes from `@dataclass` to `eqx.Module`. Key changes:
- Use `eqx.field(static=True)` for string fields (units)
- Add `__call__(C)` method with unified signature
- For temperature-dependent isotherms, store `T` inside the module

```python
class LangmuirIsotherm(eqx.Module):
    Q_max: float
    b: float
    Q_max_units: str = eqx.field(static=True)
    b_units: str = eqx.field(static=True)

    @property
    def Q_max_si(self) -> float:
        match self.Q_max_units:
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(self.Q_max)
            case "mol/kg":
                return self.Q_max
            case _:
                raise ValueError(f"Unsupported Q_max units: {self.Q_max_units}")

    @property
    def b_si(self) -> float:
        match self.b_units:
            case "L/mg":
                return self.b * Li_MW
            case "m3/mol":
                return self.b
            case _:
                raise ValueError(f"Unsupported b units: {self.b_units}")

    def __call__(self, C: float) -> float:
        """JAX-compatible isotherm evaluation."""
        return self.Q_max_si * self.b_si * C / (1 + self.b_si * C)

    @classmethod
    def from_dict(cls, data: dict) -> "LangmuirIsotherm":
        return cls(
            Q_max=data["Q_max"],
            b=data["b"],
            Q_max_units=data["Q_max_units"],
            b_units=data["b_units"]
        )


class TemkinIsotherm(eqx.Module):
    A: float
    B: float
    T: float  # Temperature baked in for unified __call__(C) signature
    A_units: str = eqx.field(static=True)
    B_units: str = eqx.field(static=True)

    @property
    def A_si(self) -> float:
        match self.A_units:
            case "L/mg":
                return self.A * Li_MW
            case "m3/mol":
                return self.A
            case _:
                raise ValueError(f"Unsupported A units: {self.A_units}")

    @property
    def B_si(self) -> float:
        match self.B_units:
            case "kJ/mol":
                return self.B * 1000 * Li_MW
            case "J/mol":
                return self.B
            case _:
                raise ValueError(f"Unsupported B units: {self.B_units}")

    def __call__(self, C: float) -> float:
        """JAX-compatible isotherm evaluation."""
        import jax.numpy as jnp
        R = 8.314
        return jnp.maximum(R * self.T / self.B_si * jnp.log(self.A_si * C), 0)

    @classmethod
    def from_dict(cls, data: dict, T: float) -> "TemkinIsotherm":
        return cls(
            A=data["A"],
            B=data["B"],
            T=T,
            A_units=data["A_units"],
            B_units=data["B_units"]
        )


class SipsIsotherm(eqx.Module):
    Q_max: float
    K_s: float
    n: float
    Q_max_units: str = eqx.field(static=True)
    K_s_units: str = eqx.field(static=True)

    @property
    def Q_max_si(self) -> float:
        match self.Q_max_units:
            case "mg/g":
                return convert_mg_per_g_to_mol_per_kg(self.Q_max)
            case "mol/kg":
                return self.Q_max
            case _:
                raise ValueError(f"Unsupported Q_max units: {self.Q_max_units}")

    @property
    def K_s_si(self) -> float:
        match self.K_s_units:
            case "L/mg":
                return self.K_s * Li_MW
            case "m3/mol":
                return self.K_s
            case _:
                raise ValueError(f"Unsupported K_s units: {self.K_s_units}")

    def __call__(self, C: float) -> float:
        """JAX-compatible isotherm evaluation."""
        return self.Q_max_si * (self.K_s_si * C)**self.n / (1 + (self.K_s_si * C)**self.n)

    @classmethod
    def from_dict(cls, data: dict) -> "SipsIsotherm":
        return cls(
            Q_max=data["Q_max"],
            K_s=data["K_s"],
            n=data["n"],
            Q_max_units=data["Q_max_units"],
            K_s_units=data["K_s_units"]
        )
```

### Step 3: Update ColumnParameters

**File:** `/home/janlc/AdsorbentColumnModeling/utils/Dataclasses.py`

Replace `q_max` and `b` with a generic isotherm module:

```python
class ColumnParameters(eqx.Module):
    u_inter: float
    k_s: float
    epsilon: float
    C_in: float
    L: float
    rho_p: float
    isotherm: eqx.Module  # Any isotherm with __call__(C) -> q
```

### Step 4: Update diffrax_column_model.py

**File:** `/home/janlc/AdsorbentColumnModeling/Model/diffrax_based/diffrax_column_model.py`

1. Remove local `langmuir_isotherm` function
2. Update `column_ode` to use the isotherm from params:

```python
def column_ode(t, state: ColumnState, args: ColumnParameters):
    C = state.C
    n = state.n  # Fixed: was state.q

    q_star = jax.vmap(args.isotherm)(C.vals)  # Use passed isotherm
    dq_dt = args.k_s * (q_star - n.vals)

    C_prev = jnp.roll(C.vals, shift=1)
    C_prev = C_prev.at[0].set(args.C_in)

    advection_dc_dx = (C.vals - C_prev) / C.δx
    sorption = (1 - args.epsilon) / args.epsilon * args.rho_p * dq_dt  # Fixed: was rho_s

    dC_dt = -args.u_inter * advection_dc_dx - sorption

    return ColumnState(
        C=SpatialDiscretisation(C.x0, C.x_final, dC_dt),
        n=SpatialDiscretisation(n.x0, n.x_final, dq_dt)  # Fixed: was q
    )
```

### Step 5: Complete to_column_parameter in Study

**File:** `/home/janlc/AdsorbentColumnModeling/utils/Dataclasses.py`

```python
def to_column_parameter(self, L: float, D: float, C_in: float, Q: float) -> ColumnParameters:
    """
    Create ColumnParameters for the diffrax model.

    Args:
        L: Column length (m) - optimizer variable
        D: Column diameter (m) - optimizer variable
        C_in: Inlet concentration (mol/m³) - case input
        Q: Volumetric flowrate (m³/s) - case input

    Returns:
        ColumnParameters for diffrax simulation
    """
    u_inter = Q / (np.pi * (D/2)**2 * self.column_experiments.column_properties.porosity)
    kinetics_experiment = self.get_kinetics_experiment_from_concentration(C_in)

    # Create JAX-compatible isotherm module from study data
    isotherm = self.isotherm.to_jax_isotherm()

    return ColumnParameters(
        L=L,
        u_inter=u_inter,
        C_in=C_in,
        k_s=kinetics_experiment.kinetics_params.k2_si,
        rho_p=self.sorbent_properties.density_si,
        epsilon=self.column_experiments.column_properties.porosity,
        isotherm=isotherm,
    )
```

Add helper to `Isotherm` class:

```python
def to_jax_isotherm(self) -> eqx.Module:
    """Convert to JAX-compatible eqx.Module isotherm."""
    match self.fit_type:
        case "Langmuir":
            return LangmuirIsotherm.from_dict(self.isotherm_fit.__dict__)
        case "Temkin":
            return TemkinIsotherm.from_dict(self.isotherm_fit.__dict__, T=self.T)
        case "Sips":
            return SipsIsotherm.from_dict(self.isotherm_fit.__dict__)
        case _:
            raise ValueError(f"Unsupported isotherm type: {self.fit_type}")
```

---

## Files to Modify/Create

| Action | File |
|--------|------|
| Create | `/home/janlc/AdsorbentColumnModeling/__init__.py` |
| Create | `/home/janlc/AdsorbentColumnModeling/Model/diffrax_based/__init__.py` |
| Edit | `/home/janlc/AdsorbentColumnModeling/Model/diffrax_based/diffrax_column_model.py` |
| Edit | `/home/janlc/AdsorbentColumnModeling/utils/Dataclasses.py` |

---

## Bug Fixes in diffrax_column_model.py

1. `state.q` → `state.n` (ColumnState uses `n`, not `q`)
2. `args.rho_s` → `args.rho_p` (ColumnParameters uses `rho_p`)
3. Return statement: `q=...` → `n=...`

---

## Design Decisions

### Why store T inside temperature-dependent isotherms?
All isotherms have a unified `__call__(C)` signature. Temperature is "curried" at creation time for Temkin, so the ODE code stays simple:
```python
q_star = jax.vmap(args.isotherm)(C.vals)
```

### Why use eqx.Module for isotherms?
- JAX compatibility (pytree structure)
- Works with `jax.jit`, `jax.grad`, `jax.vmap`
- Immutable by default
- Preserves existing class structure (properties, classmethods)

---

## Verification

1. Test import works:
   ```python
   from utils.Dataclasses import Study, ColumnParameters
   from Model.AlLDH.diffrax_column_model import run_wrapper
   ```

2. Load study and create parameters:
   ```python
   study = Study.from_json('LiteratureReview/isotherm_kinetics.json', 'study_key')
   params = study.to_column_parameter(L=0.6, D=0.02, C_in=7.2, Q=1e-6)
   ```

3. Run simulation:
   ```python
   solution = run_wrapper(params)
   ```
