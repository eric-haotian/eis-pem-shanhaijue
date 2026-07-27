# EIS-PEM ShanHaiJue + ShanHaiJian — CPU distribution

Routed multi-estimator identification (**ShanHaiJue**) and risk-controlled certification
(**ShanHaiJian**) for lithium-ion battery EIS parameter recovery, on a 48-parameter
DFN-type frequency-domain impedance model.

This distribution runs the identical algorithm on CPU (scipy trust-region-reflective).
It is the reference implementation: portable, dependency-light, and used for the
cross-engine reproduction checks reported in the paper. For the fastest path to the
strongest configuration use the GPU distribution.

| Configuration | Routed recovery | Certified @ ≥0.95 precision | Wall clock (18 cores) |
|---|---|---|---|
| 3 arms | 22.3 of 48 | 7.35 / 8.77 (validated, chain 1) | ~1.5 h per 60-cell panel |
| 6 arms | 26.2 of 48 | — | ~2.5 h |
| 8 arms (strongest) | 28.0 dev / **27.8 validated** | **10.38 @ 0.960** (validated) | ~4 h |

Cross-engine check: the six-arm CPU routing (26.2) and the six-arm GPU routing (26.4)
agree, so the two distributions are interchangeable up to run time.

---

## 1. Install

```bash
pip install -r requirements.txt
```

That is all. The physics model and priors are vendored under `shanhaijue/vendor/`; no
external data or configuration is needed.

**Install from `requirements.txt`, not by hand.** The scikit-learn bound in it
(`>=1.6,<1.9`) is not decoration: the shipped frozen system carries a pickled
gradient-boosted ranker, which loads and scores identically across that window and fails
outside it — 1.5.2 raises inside `predict_proba`, 1.9.0 cannot unpickle at all.
`artifacts/validated_system/ENVIRONMENT.json` records the measurement, and `load_frozen`
says so explicitly if you land outside the window anyway.

## 2. One-cell demo

```bash
python3 bin/certify.py --system artifacts/validated_system --budget 0.60 --demo
```

Simulates one fresh cell, fits the ensemble, routes every coordinate, and applies the
frozen certifier — printing the certified claims with their calibrated probabilities and,
because the demo cell has a known truth, marking which were correct.

This is the full 25-condition × 120-frequency grid and the whole eight-arm ensemble on one
core: **budget 15–20 minutes** on a laptop (measured: 16 min on an M-series MacBook Air).
If you only want to know the installation is sound, `python3 bin/selftest.py` exercises
every stage on a deliberately small grid in about 90 seconds.

## 3. Full workflow

```bash
# (a) development panel (start small: --truths 4 --arms 3 runs in minutes)
python3 bin/run_panel.py --truths 30 --noise 2 --arms 8 --procs 18 --out dev_rows.jsonl

# (b) out-of-fold routed recovery, truth-grouped 5-fold
python3 bin/train_router.py --rows dev_rows.jsonl

# (c) freeze a deployable system; RESERVE the validation seeds here
python3 bin/freeze_system.py --rows dev_rows.jsonl --out artifacts/my_system \
        --budgets 0.32,0.50,0.60 --val-draw-seed 990101001 --val-noise-seed 990102001

# (d) generate the validation panel with EXACTLY those reserved seeds
python3 bin/run_panel.py --truths 30 --noise 2 --arms 8 \
        --draw-seed 990101001 --noise-seed 990102001 --out val_rows.jsonl

# (e) single-pass validation — run this ONCE
python3 bin/validate_system.py --system artifacts/my_system --rows val_rows.jsonl
```

## 4. Desktop interface

Double-click **`启动界面_Start_GUI.command`** (macOS) or **`启动界面_Start_GUI.bat`**
(Windows). The launcher finds Python, installs anything missing from
`requirements.txt`, and opens the window. Interface language switches between Chinese and
English at any time from the top-right button.

Take a CSV, get an XLSX:

```
input CSV  ->  ensemble fit  ->  routing  ->  certification  ->  XLSX (4 sheets)
```

**Canonical input format.** Five columns, one row per frequency, each condition's spectrum
concatenated head to tail:

```
Temperature_K,SOC,Frequency_Hz,Re_Z,Im_Z
278.15,0.1,0.001,0.019984408733,-0.005062697
278.15,0.1,0.001167419,0.019721069778,-0.004671138
...
```

`examples/calibration_grid_25x120.csv` is exactly that: 25 conditions × 120 frequencies on
the grid the frozen system was calibrated on, so it is both the format reference and the
acquisition design that carries the ≥0.95 guarantee. `examples/format_sample.csv` is a
30-row version for reading at a glance.

**Other layouts are accepted too.** Only frequency and complex impedance are genuinely
required; everything else is inferred and reported:

| Quantity | Accepted headers | If absent |
|---|---|---|
| frequency | `frequency`, `freq`, `f`, `f_hz`, `hz`, `频率` | required |
| Z real | `zre`, `zreal`, `z_re`, `re`, `z'`, `实部` | required, or give `\|Z\|`+`phase` |
| Z imag | `zim`, `zimag`, `z_im`, `im`, `z''`, `虚部` | as above |
| −Z imag | `-zim`, `-z''`, `zim_neg`, `负虚部` | sign flipped automatically |
| temperature | `temperature`, `temp`, `t`, `t_k`, `温度` | panel default, °C or K auto-detected |
| SOC | `soc`, `state_of_charge`, `荷电状态` | panel default, 0–1 or 0–100 auto-detected |
| cell | `cell`, `cell_id`, `sample`, `电池` | all spectra treated as one cell |

Units in brackets are ignored (`Z' (ohm)` works), the delimiter is sniffed (comma,
semicolon, tab), and a headerless numeric file is read as `f, Z', Z''`. Spectra are split
at cell/condition changes or wherever the frequency sweep restarts.

**Grid.** The measured conditions go to the forward model verbatim, so a single spectrum,
a scattered set of temperature/SOC pairs, and the full 5 × 5 grid are all admissible;
conditions measured at different frequencies are resampled onto their common range.
Estimates for all 48 coordinates are returned in every case.

**Certification is grid-bound, identification is not.** The calibrated probabilities and
the false-claim threshold were fitted at one information content. Run on a different grid
and the window reports the comparison axis by axis — which of temperature, SOC, and
frequency moved, and whether the point count is unchanged — the workbook marks every
certification cell *out of calibration*, and those rows should be read as point estimates.
For a valid guarantee on a new grid, re-freeze and re-validate on it (§3c–e). With far
fewer points than the calibration grid, expect the certifier to abstain on nearly
everything: that is the intended behaviour, not a failure.

**Parameter panel.** Nineteen tunables in six groups, each carrying a one-line statement of
which way the result moves and a hover explanation with the mechanism, the compute cost, and
any caveat. Two distinctions are enforced rather than documented:

| Marking | Meaning |
|---|---|
| greyed out | the CSV supplies this quantity, so the fallback control is inactive (the fallback temperature and SOC behave this way) |
| *re-freeze only* | inert for this run; the value belongs to building a new system with the `bin/` scripts, and is shown so the full surface is visible |
| red label | changing it moves the feature distribution the frozen router and certifier were calibrated against, so estimates still come out but the precision guarantee no longer transfers — the log says so at run time |

**Output.** Four sheets: certified claims, all 48 coordinates with probabilities and
abstentions, a run summary recording every setting used, and the full log. The same table
appears live in the window as each cell finishes, filterable and sortable.

## 5. Deployment on your own spectra

```python
import numpy as np
from shanhaijue import panel, arms, features, pipeline

# declare the grid your spectra were actually measured on, then fit
panel.configure_grid(conditions=[(298.15, 0.3), (298.15, 0.7)],   # (T in K, SOC)
                     frequencies=np.logspace(-2, 4, 60))
z = np.load("my_spectrum.npz")["z"]        # complex, conditions concatenated in order

fitted, aux = arms.fit_ensemble(z, seed=0)
rows = features.extract(fitted, aux, z, cell_id=("cell-A", 0))

frozen = pipeline.load_frozen("artifacts/validated_system/frozen_system.joblib")
certified, routed = pipeline.apply_frozen(frozen, rows, budget=0.60)

for c in certified:
    if c["certified"]:
        print(c["coord"], c["p_certified"], c["arm"])
```

`configure_grid` accepts the measured conditions verbatim — one spectrum, a scattered set
of temperature/SOC pairs, or a full cross product — so identification runs on any
acquisition design. Certification does not transfer that freely: compare your grid against
`pipeline.frozen_grid(frozen)`, and where it differs, treat the claims as point estimates
until you have re-frozen and re-validated on that grid (§3c–e).

## 6. What the pieces do

| Module | Role |
|---|---|
| `shanhaijue/panel.py` | 48-parameter forward model, physics priors, acquisition grid, physical claim basis, synthetic panel generation |
| `shanhaijue/arms.py` | **ShanHaiJue**: the estimator ensemble along the prior-strength axis λ ∈ {3, 1, 0.6, 0.3, 0.1, 0.03} + wide multistart + data-only refinement |
| `shanhaijue/features.py` | ten ground-truth-free per-coordinate diagnostics (curvature, dispersion, held-out shift, cross-arm disagreement, …) |
| `shanhaijue/router.py` | gradient-boosted ranker + per-arm calibration + arg-max routing, with truth-grouped cross-fitting |
| `shanhaijue/certifier.py` | **ShanHaiJian**: Platt calibration on cross-fitted routed scores + false-claim budget enforced at a truth-cluster bootstrap upper confidence bound |
| `shanhaijue/pipeline.py` | panel → ensemble → features → router → certifier orchestration, freezing, and single-pass validation |
| `shanhaijue/vendor/eispem/` | vendored physics layer: the 48-parameter impedance model, its parameter/prior machinery, and the numerical utilities they need (35 modules, one flat namespace, empty package initialiser so importing one module drags in nothing else) |
| `gui/` | desktop interface: `app.py` window, `params.py` the tunable panel declared as data, `i18n.py` every user-visible string in both languages, `csv_io.py` permissive reader and XLSX writer, `results.py` the live results table, `widgets.py` tooltip/row/scroll helpers, `fitjob.py` the per-cell parallel job |
| `examples/` | `calibration_grid_25x120.csv` (canonical format, on the calibration grid) and a 30-row `format_sample.csv` |
| `artifacts/validated_system/` | the frozen eight-arm system behind the validated numbers (transferable to this backend) |

## 7. Why the ensemble works

Arms differ **only** in the prior scale λ and initialization. At λ = 1 the prior
stabilises the flat directions of the sloppy cost valley at the price of bias on the
data-determined directions; as λ → 0 that bias vanishes but the flat directions
destabilise. The regimes therefore fail on *different* coordinates (correct-set Jaccard
index 0.515), so a per-coordinate oracle over the ensemble reaches 32.4 of 48 at six
arms against 17.1–18.5 for any single member. The router captures ~60 % of that headroom
without ever seeing a truth label at inference time.

## 8. Scope and honest limits

- All quantities are **synthetic-panel** estimates within the calibration generator
  class. Transfer to measured cells is not validated; the out-of-distribution guard in
  the accompanying paper withheld calibrated labels from all 87 real commercial-cell
  spectra tested.
- "Certified" means certified under a **stated empirical risk budget**, not a
  finite-sample guarantee. The cluster bootstrap is anti-conservative at ~30 clusters
  (measured one-sided coverage 0.88–0.93 against nominal 0.95); read budgets with a
  2–7 percentage-point haircut.
- **Freeze before you validate.** Reserve the validation seeds at freeze time, generate
  the validation panel afterwards, analyse it once, and never pool panels.

## 9. Self-test status

Both distributions were verified end-to-end on the reduced self-test grid
(4 conditions x 24 frequencies, three-arm ladder, small iteration budget):

| distribution | backend | result | wall clock |
|---|---|---|---|
| CPU | scipy trust-region-reflective | **PASSED** (6/6 stages) | 479 s |
| GPU | JAX/CUDA batched LM | **PASSED** (6/6 stages) | 146 s |

The self-test exercises every stage — forward model, estimator ensemble, ground-truth-free
features, panel routing, freezing with reserved validation seeds, and single-pass
validation on a fresh panel. Its recovery and precision numbers are meaningless as
performance figures: the grid is deliberately tiny so the check runs in minutes. The
validated performance is in `artifacts/validated_system/VALIDATED_PERFORMANCE.json`.

## 10. Reference

The accompanying manuscript describes the method, the five freeze–validate chains, and
the acquisition-design rules. The manuscript sources (`paper_eis_sci_cn/`) and the full
evidence map (`paper_experiment_runs/PIPELINE_MAP.md`) live in the research repository
this distribution is cut from; they are not part of the distribution itself.

## 11. License

Apache License 2.0 — see `LICENSE`. The vendored physics layer under
`shanhaijue/vendor/eispem/` is covered by the same license.
