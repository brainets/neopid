# SPID Repository — Incongruencies & Bugs Audit

> Generated: 2026-04-15  
> Scope: cross-check of `paper/spid_v2.pdf` against all source files and notebooks.

---

## Summary table

| Severity | Issue | Location |
|---|---|---|
| **Critical** | `simulate_delayed` crashes on call (8-value unpack into 7 variables) | `src/models.py:283` |
| **Major** | Coupling window 0–0.4 s in code vs. paper's stated 0.1–0.3 s | All notebooks |
| **Major** | `C₁₂ = 10` in code vs. `C₁₂ = 1` in Fig. 3 caption | `notebooks/Figure3.ipynb` |
| **Major** | Delay matrix loaded and converted, then silently discarded in Fig. 4 | `notebooks/Figure4.ipynb` |
| **Moderate** | "0.4 ms" unit typo in Figs. 3–4 captions (should be "0.4 s") | Paper |
| **Moderate** | Visual circuit delays zeroed via `* 0` factor | `notebooks/Figure2.ipynb` |
| **Moderate** | `stim_mode="both"` is unimplemented (falls through as no-op) | `src/models.py` |
| **Minor** | Hardcoded absolute home-directory paths in Figs. 3–4 notebooks | `notebooks/Figure3.ipynb`, `Figure4.ipynb` |
| **Minor** | Unused import `mean_squared_error` | `notebooks/Figure4.ipynb` |
| **Minor** | `ypos` array has 7 elements for 6 axes in Fig. 1 panel labels | `notebooks/Figure1.ipynb` |
| **Minor** | Real and imaginary noise components correlated at `t=0` | `src/models.py:128–129` |
| **Minor** | `_set_nodes()` docstring says "Kuramoto" but is used for SL too | `src/models_setup.py:8` |
| **Minor** | Author TODO/red-text comments left in PDF manuscript | Paper |

---

## 1. CRITICAL BUG — `simulate_delayed()` crashes on call

**File:** `src/models.py:283`

`_set_nodes_delayed()` returns **8** values:

```python
# src/models_setup.py:126
return N, A, D, max_delay, omegas, jnp.asarray(phases)..., dt, a
#       1  2  3  4          5       6                        7   8
```

`simulate_delayed()` only unpacks **7**:

```python
# src/models.py:283
N, A, D, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)
```

This raises `ValueError: too many values to unpack (expected 7)` whenever called.
The variable assignments are also completely wrong from position 4 onwards:
`omegas` receives `max_delay` (an int), `phases_history` receives `omegas` (a float array),
`dt` receives the complex phases array, and `a` from the return is not captured at all.

`max_delay` was likely added to the return of `_set_nodes_delayed` (needed by
`simulate_kuramoto`) after `simulate_delayed` was written, without updating the latter.

**Currently unexploded:** both `Figure3.ipynb` and `Figure4.ipynb` import `simulate_delayed`
but never call it. Figure 3 uses `jax.vmap(simulate, ...)` instead.

**Fix:** add `max_delay` to the unpacking in `simulate_delayed`:

```python
N, A, D, max_delay, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)
```

---

## 2. MAJOR — Coupling time window mismatch (paper vs. code)

**Affects:** Figures 1, 2, 3, 4

| | Paper states | Code uses |
|---|---|---|
| Fig. 1 caption | "transiently non-zero from **0.1 to 0.3 s**" | `time_start=0, time_end=0.4` → **0 to 0.4 s** |
| Fig. 2 caption | "from **0.1 to 0.3 s**" | same 0–0.4 s |
| Fig. 3/4 caption | "from **0 to 0.4 ms**" | `time_end=0.4` (seconds) |

The Figs. 3 and 4 captions also contain a **unit typo**: "0.4 ms" should be "0.4 s".
At 40 Hz, a 0.4 ms coupling window is shorter than a single oscillation cycle and
makes no physical sense.

**Relevant code (same pattern in all notebooks):**

```python
# e.g. notebooks/Figure1.ipynb, cell 6
time_start = 0
time_end = 0.4    # seconds, not ms; and not 0.1–0.3 as stated in paper
```

---

## 3. MAJOR — Stuart-Landau coupling strength mismatch (paper vs. code)

**File:** `notebooks/Figure3.ipynb` (cell 11)

```python
# Paper Figure 3 caption: "coupling strength C₁₂ = 1"
out = simulate_loop(10 * C, ...)   # C₁₂ = 10 in code
```

The paper states `C₁₂ = 1`; the code passes the adjacency matrix scaled by `10`.
This is a factor-of-10 discrepancy in the coupling strength used for Figure 3.

For comparison, Figure 1 (Kuramoto) also uses `10 * A` and states `C₁₂ = 10` — consistent.
The Figure 3 caption value appears to be wrong.

---

## 4. MAJOR — Figure 4 delay matrix computed but silently discarded

**File:** `notebooks/Figure4.ipynb` (cells 10–12 vs. cell 16)

```python
# Cell 10: distances loaded and converted to delay in seconds
D = data["Distances"] * 1e-3 / 3.5   # axonal velocity = 3.5 m/s

# Cell 12: converted to integer timesteps
D = (D / dt).astype(int)              # D is ready to use

# Cell 16: simulation — D is never passed
out += [simulate(
    flnMat,
    coupling[trial],
    f, -5, fsamp, beta, Npoints,
    None,            # Iext[trial] is commented out
    seeds[trial],
    decim=decim,
    stim_mode="both",
)]
```

`simulate()` (no-delay version) is called instead of `simulate_delayed()`, so all
anatomical conduction delays are ignored in the whole-brain simulation.

Additionally, the `Iext` array prepared in cell 15 for V1-only stimulation
(`Iext[:, 0] = coupling`) is passed as `None`, leaving it as dead code.

---

## 5. MODERATE — Visual circuit delays zeroed with `* 0` factor

**File:** `notebooks/Figure2.ipynb` (cell 8)

```python
elif structure == "visual":
    A = SC_data["FLN"][np.ix_([0, 1, 2], [0, 1, 2])]
    D = SC_data["Distances"] * 0 * 1e-3 / 3.5   # ← * 0 zeroes all delays
    D /= dt
    D = D[np.ix_([0, 1, 2], [0, 1, 2])]
```

The `* 0` factor makes all delays zero. If intentional, this should be written as
`D = np.zeros_like(A)` for clarity. As written, it implies the distance matrix was
retrieved and then deliberately discarded, which is confusing.

---

## 6. MODERATE — `stim_mode="both"` is unimplemented (no-op)

**File:** `src/models.py:289–301`

```python
# Stim parameters
gain = 0
phi = 0
offset = 1

if stim_mode == "amp":
    gain = 1
    offset = 0
elif stim_mode == "phase":
    gain = 1
    phi = np.pi / 2
    offset = 0
# "both": no branch → gain=0, offset=1 remains
# → exp_phi = 0 * exp(...) + 1 = 1
# → Iext * exp_phi = Iext * 1 (no amplitude or phase weighting)
```

`stim_mode="both"` falls through both conditions. The external-input weighting term
`exp_phi` reduces to `1`, so the mode is identical to applying `Iext` with no modulation.
Since Figure 3 and 4 call `simulate(..., stim_mode="both")` with `Iext=None`, the
parameter has no effect in either case.

---

## 7. MINOR — Hardcoded absolute paths break portability

**Files:** `notebooks/Figure3.ipynb` (cell 1), `notebooks/Figure4.ipynb` (cell 1)

```python
# Figure3.ipynb
sys.path.insert(1, os.path.expanduser("~/projects/phase_amplitude_encoding/"))

# Figure4.ipynb
sys.path.insert(1, os.path.expanduser("~/projects/phase_amplitude_encoding"))
```

These paths are specific to the original developer's machine and will silently fail
on any other system. Should use a path relative to the notebook location, consistent
with how Figure 1 and 2 handle it:

```python
sys.path.insert(1, os.path.join("/", *os.getcwd().split("/")[:-1]))
```

---

## 8. MINOR — Unused import in Figure 4

**File:** `notebooks/Figure4.ipynb` (cell 2)

```python
from sklearn.metrics import mean_squared_error   # never used
```

---

## 9. MINOR — `ypos` array off-by-one in Figure 1 panel labels

**File:** `notebooks/Figure1.ipynb` (plotting cell)

```python
plot.add_panel_letters(
    fig,
    axes=[ax1, ax2, ax3, ax5, ax6, ax7],   # 6 axes
    fontsize=MEDIUM_SIZE,
    xpos=[-0.3] * 6,
    ypos=[1.1] * 5 + [0.95] * 2,           # 7 values — last one silently dropped
)
```

`zip(axes, xpos, ypos)` stops at 6 (shortest iterable), so the 7th `ypos` value is
never used. The likely intention was `[1.1] * 4 + [0.95] * 2` so that both right-column
panels (ax6, ax7) get `ypos=0.95`, but as written only ax7 does.

---

## 10. MINOR — Correlated noise at `t=0` in Stuart-Landau integrator

**File:** `src/models.py:128–129`

```python
+ eta * randn(size=(N,), seed=seed + t)
+ eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
```

At `t=0`: both calls resolve to `randn(seed=seed)`, producing identical values.
The complex noise at the first time step therefore has equal real and imaginary parts
(purely diagonal in the complex plane), instead of independent Gaussian components.
For `t > 0` the seeds differ (`seed+t` vs. `seed+3t`), so the issue is confined to
a single time step and has negligible effect on results.

---

## 11. MINOR — `_set_nodes()` docstring misidentifies model

**File:** `src/models_setup.py:8`

```python
def _set_nodes(A, f, fs, a):
    """
    Setup nodes for Kuramoto simulation without time delays.
    ...
    """
```

This function is used for both Kuramoto (`simulate_kuramoto`) and Stuart-Landau
(`simulate`, `simulate_delayed`) oscillators. The docstring should be updated.

---

## 12. MINOR — Unresolved author annotations left in PDF

**File:** `paper/spid_v2.pdf`

Two visible author notes remain in the manuscript body:

- **p. 8, lines 192–193** (red text): *"I don't understand this part so much, do we have any plot associated to this?"*
- **p. 10, lines 218–219** (plain text TODO): *"TO DO Figure 4. The labels V1-V2 and V1-24c should be inverted. It is not clear the definition of two-plets and three-plets. I would remove them. Maybe adjust the colors of the curves."*
- **p. 16, lines 311–313** (red text): *"COMMENT: I don't know if it's worth presenting this as a new approach called SPID…"*

These must be resolved before submission.
