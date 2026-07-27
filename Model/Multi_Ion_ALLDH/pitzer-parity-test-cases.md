# Differential test — Python port of `pitzer_mix` vs the MATLAB original

**Principle: same inputs, same outputs.** The MATLAB is the oracle. No expected values are specified
anywhere in this document — every assertion compares the two implementations to each other.

```matlab
function [lngamma_LiCl, lngamma_NaCl, lngamma_MgCl2, osmo_w] = pitzer_mix(m_LiCl, m_NaCl, m_MgCl2)
```

Three scalar molalities in (mol/kg water), four scalars out. Pure, deterministic, no state.

**Translate faithfully.** Reproduce the MATLAB's behaviour including anything that looks wrong. A
port that "improves" on the original cannot be tested against it, and every mismatch becomes
unattributable.

Compare `lngamma`, never `gamma` — the logs are the actual output, and exponentiating amplifies
discrepancies non-uniformly.

---

## 1. Compare intermediates, not just the four outputs

Endpoint-only comparison leaves you with one wrong number and twenty candidate causes. Expose these
on both sides, **in this order** — it is dependency order, so the first mismatch is the cause and
everything after it is downstream contamination.

```
I_tot, Z_sum, m_Cl, m_tot, A_phi,
thetaE_LiMg, dthetaE_LiMg, Phi_LiMg, Phiphi_LiMg,
g_chi, dg_chi, B_LiCl, dB_LiCl, C_LiCl, BC_LiCl,
phi_DH, bin_phi, tert_phi, F_mix, osmo_w,
lngamma_LiCl, lngamma_NaCl, lngamma_MgCl2
```

Assert in order. Stop at the first failure. Report only that one.

On the MATLAB side you can get all of these without editing the original: copy `pitzer_mix.m` to
`pitzer_body.m` and delete two lines — the `function` declaration and the trailing `end`. It becomes
a script, so every local lands in the base workspace and is readable by name.

Also compare `J0(x)` and `J1(x)` **on their own, before anything else.** They are pure one-argument
functions and everything else depends on them; if either differs, the rest of the suite tells you
nothing about where the problem is. Sweep x ∈ {0.01, 0.05, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100}.

---

## 2. Input grid

~150 triples. Every intermediate compared at every point.

| Subgroup | Cases | What a failure here isolates |
|---|---|---|
| Single salt | `(m,0,0)`, `(0,m,0)`, `(0,0,m)` for m ∈ {0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 6} | One salt's block; all mixing terms are zero |
| Binary | all three pairs at {0.1, 1, 3} × {0.1, 1, 3} | One θ/ψ pair at a time |
| Ternary | 3-factor grid at {0.05, 0.5, 2, 4} | Cross terms only active here |
| Target brine | `(0.1, 2.0, 1.0)`, `(0.05, 3.0, 2.0)`, `(0.02, 4.0, 1.5)` | The regime that actually matters |
| Trace | `(1e-4, 3, 2)`, `(3, 1e-4, 1e-4)` | One species near zero while others are high |
| Low I | `(1e-3, 0, 0)`, `(1e-4, 1e-4, 1e-4)` | Ill-conditioned regime — see §5 |
| High I | `(0, 4, 2)`, `(1, 3, 3)` | Upper end of the parameterisation |
| Degenerate | `(0,0,0)`, and all pairs with two zeros | NaN behaviour |

The grid structure is the point. Single-salt cases zero out the mixing terms entirely, binaries
activate exactly one θ/ψ pair, ternaries turn everything on. A failure confined to one subgroup names
the broken term before you read any code.

Generate the triples once in Python and feed the same values to both sides. Do not generate them
independently in each language.

---

## 3. What counts as "the same"

| Class | rtol | atol |
|---|---|---|
| Closed-form algebra: `I_tot`, `Z_sum`, `m_Cl`, `m_tot`, `g_chi`, `dg_chi`, `B*`, `dB*`, `C*`, `BC*`, `A_phi`, `phi_DH`, `bin_phi` | `1e-13` | `1e-15` |
| Quadrature-dependent (`J*`, `thetaE*`, `Phi*`, `F_mix`, `osmo_w`, `lngamma*`), I ≥ 0.1 | `1e-8` | `1e-14` |
| same, 0.01 ≤ I < 0.1 | `1e-6` | `1e-13` |
| same, I < 0.01 | `1e-4` | `1e-10` |

The split matters. The first tier involves identical IEEE operations on both sides with nothing lost
in transfer, so agreement should be near-exact — **anything in that tier off by more than `1e-13` is
a transcription error, not floating-point noise, and widening the tolerance to make it pass hides the
bug the suite exists to find.**

The looser tiers exist only because the two languages use different adaptive quadrature routines for
`J0`/`J1`. That is the sole legitimate source of disagreement in this function.

---

*Prepared by Claude (Anthropic, Cowork) for the multi-ion ALLDH isotherm model. Session
[01CLfqLmqYT1tgyQyyXViAKW](https://claude.ai/code/session_01CLfqLmqYT1tgyQyyXViAKW) · 2026-07-27.*
