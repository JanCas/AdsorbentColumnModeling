# Task: IX non-dim model — two related changes

File: `Model/IX/NonDim/ix_nondim.py` (the uploaded `ix_nondim.py`).
Apply BOTH changes below. They are independent but touch the same params,
so do Change 1 first, then Change 2.

---

## Change 1 — revert the isotherm from `theta` to bare `K*`

### Goal
Switch the isotherm from the feed-favorability form

    n*_eq = (1 + theta) * A* / (h + theta * A*),   h = H*/H*_in,  theta = K*/H*_in

to the bare mass-action form scaled by Q_sites, using a single concentration
scale (A_in) everywhere:

    n*_eq = K* * A* / (H* + K* * A*)

This removes the hidden `H*_in` proton scale, restores `K*` as the sampled
group, and puts loading on fractional-occupancy scaling (n* in [0, f], with
feed loading f = K*/(H*_in + K*) < 1). It is algebraically the same isotherm —
no physics change — just consistently A_in-normalized.

### Edits

1. `n_eq_star` (~lines 166–170): take `H` (not `h`) and use `K_star`:

   ```python
   def n_eq_star(A: jax.Array, H: jax.Array, args: NonDimParams) -> jax.Array:
       """Bare mass-action isotherm, A_in-consistent (Q_sites loading scale).
       At feed (A*=1, H*=H*_in): n*_eq = K*/(H*_in + K*) = f < 1.
       """
       return args.K_star * A / (H + args.K_star * A)
   ```

2. `vector_field` (~lines 179–181): drop the `h = H/H*_in` line and pass `H`:

   ```python
   H = h_star(T, args.omega)
   ne = n_eq_star(A, H, args)
   ```

3. `NonDimParams`: rename field `theta` -> `K_star` (~line 107), update the
   field-name tuple in `__post_init__` (~line 122). Update the class docstring
   note that explained the beta->theta rename: it is now `K_star`, the bare
   dimensionless mass-action constant (no feed scaling folded in).

4. `default_params`: rename arg `theta: float = 1.0` -> `K_star`, pass
   `K_star=K_star` into `NonDimParams`, fix the docstring.

5. Sobol adapter (`Sensitivity/adapters/ix_nondim.py`) and any spec/QoI driver:
   rename the sampled axis `theta` -> `K_star`. Pick the K* sample range
   directly (it is no longer divided by H*_in) — confirm the intended range
   before running.

6. Update stale comments/docstrings that assert "n*=1 at feed by construction"
   (module header ~lines 18–20 and `n_eq_star`): feed loading is now f < 1.

---

## Change 2 — rename the inlet proton parameters to H_in_ads / H_in_des

### Rationale
`H_star_in` (load/feed inlet proton level) and `Pi_el` (desorb/eluent inlet
proton level) are the SAME physical quantity in two phases: the dimensionless
inlet proton level H* = [H+]/A_in. Rename so this is obvious:

    H_star_in  ->  H_in_ads   (adsorption-phase inlet proton level, the feed pH)
    Pi_el      ->  H_in_des   (desorption-phase inlet proton level, the strip pH)

### Edits

1. `NonDimParams`: rename fields `H_star_in -> H_in_ads` and `Pi_el ->
   H_in_des`. Update the `__post_init__` field-name tuple and docstrings.

2. Module constants: rename `H_STAR_IN_DEFAULT -> H_IN_ADS_DEFAULT`. Add
   `H_IN_DES_DEFAULT = 1.0e2` (the old `Pi_el` default) so both inlets have a
   named default.

3. Inlet boundary values — make BOTH phases use the same H* -> T* conversion so
   the naming is honest (T* is the transported variable; the exact positive-root
   inverse of the water quadratic is T* = H* - omega/H*):

   ```python
   # load (adsorption) inlet
   T_load = nd.H_in_ads - nd.omega / nd.H_in_ads
   # desorb inlet  (was: des T_in = Pi_el directly, ~line 405)
   T_des  = nd.H_in_des - nd.omega / nd.H_in_des
   ```
   Then set `des_phase` T_in to `T_des`.

   NOTE: tiny numerical change for the desorb inlet. The old code set
   T*_in = Pi_el directly; since the eluent is strongly acidic (H_in_des in
   [1e2, 1e3]) the omega/H term is ~1e-10, so T_des == H_in_des to ~10 sig figs.
   To keep byte-identical desorb behavior instead, set `T_des = nd.H_in_des`
   directly and skip the inverse for the desorb phase only.

4. `default_params`: rename kwargs `H_star_in -> H_in_ads`, `Pi_el -> H_in_des`,
   update defaults to the renamed constants, fix the docstring.

5. Sobol adapter, spec HTML, QoI/driver code: rename the sampled/labelled axes
   `Pi_el -> H_in_des` and `H_star_in -> H_in_ads`. Keep the existing sample
   ranges (H_in_ads ~ [1e-9, 1e-8], H_in_des ~ [1e2, 1e3]); only names change.

6. Update comments referencing "Pi_el" / "acid eluent strength" / "feed proton
   reference" to the new names.

   NOTE: `H_in_ads` (formerly `H_star_in`) now appears ONLY in the load-phase
   inlet — after Change 1 it is no longer in the isotherm.

---

## Verify (after both changes)
- `default_params()` builds and `qois(default_params())` runs without NaN.
- Isotherm: at feed (A*=1, T* set so H*=H_in_ads) `n_eq_star` returns
  K*/(H_in_ads + K*), NOT 1.
- Inlets: H*(T_load) == H_in_ads and H*(T_des) == H_in_des to float64 precision.
- Mass-balance check on A*+T* (pure advection) still holds.
