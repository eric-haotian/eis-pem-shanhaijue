# EIS-PEM ShanHaiJue 山海决

Routed multi-estimator identification and risk-controlled
certification for lithium-ion battery EIS parameter recovery on a
48-parameter DFN-like SEIS forward model.

This is the next version of
[eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification), codename
**ShanHaiJue**. The forward model, the parameter machinery, and the identifiability
diagnostics carry over unchanged; what is new is the layer above them.

learning AI at **www.haotianblog.com**

## What Changed

The previous version screened identifiability once, estimated the free subset, and
reported the rest as fixed reference assumptions. It stated the boundary honestly — a good
impedance fit does not imply reliable physical parameters — but it answered that boundary
with a single estimator and a single selection.

ShanHaiJue answers it with a population and a budget:

- **An ensemble along the prior-strength axis.** Arms differ only in the prior scale λ and
  in initialization. At λ = 1 the prior stabilises the flat directions of the sloppy cost
  valley at the price of bias on the data-determined directions; as λ → 0 that bias
  vanishes but the flat directions destabilise. The regimes therefore fail on *different*
  coordinates (correct-set Jaccard index 0.515), so a per-coordinate oracle over eight arms
  reaches 35.7 of 48 against 17.1–18.5 for any single member.
- **Per-coordinate routing on ground-truth-free evidence.** A gradient-boosted ranker maps
  ten diagnostics — none of which requires knowing the answer — to a claim probability per
  (arm, coordinate), and each coordinate is taken from its highest-scoring arm. Truth labels
  train the ranker offline; deployment consumes none.
- **Certification under a stated false-claim budget.** The claim threshold is the smallest
  one whose false claims per cell respect a preregistered budget at a 95 % upper confidence
  bound, bootstrapped over generating truths rather than over rows. Abstention is the
  default state: a coordinate below the threshold is simply not claimed.

| Quantity | ShanHaiJue | Previous version |
|---|---|---|
| Recovered coordinates per cell (of 48) | **27.8** | 17.1 single-estimator baseline |
| Certified at realized precision 0.960 | **10.38** | no certification layer |
| Certified at realized precision 0.968 | 8.48 | — |
| Evidence used at inference | ten ground-truth-free diagnostics | identifiability screen |
| Cost per cell | 23 bounded fits + 1 Jacobian | 1 fit |

Recovery is validated on three never-before-generated panels; every certification rung was
validated on its own panel, analysed exactly once, never pooled.

## Method Boundary

The previous version's boundary statement still holds, and the certification layer adds two
of its own. Read these before quoting any number above.

- All quantities are **synthetic-panel** estimates within the calibration generator class.
  Transfer to measured cells is not validated: the out-of-distribution guard withheld
  calibrated labels from all 87 real commercial-cell spectra tested.
- "Certified" means certified under a **stated empirical risk budget**, not a finite-sample
  guarantee. The cluster bootstrap is anti-conservative at ~30 clusters (measured one-sided
  coverage 0.88–0.93 against nominal 0.95), so read budgets with a 2–7 percentage-point
  haircut.
- **Identification runs on any grid; certification does not.** The measured conditions go to
  the forward model verbatim, so a single spectrum, a scattered set of temperature/SOC
  pairs, and a full 5 × 5 grid all produce estimates for all 48 coordinates. The calibrated
  probabilities and the claim threshold were fitted at one information content, so on a
  different grid the claims are point estimates until you re-freeze and re-validate.
- **Freeze before you validate.** Reserve the validation seeds at freeze time, generate the
  validation panel afterwards, analyse it once, and never pool panels. Re-tuning on a
  validation panel silently destroys the meaning of every number above.
- The best operating point (10.38 @ 0.960) is a maximum over nine rungs selected by
  sequential dose-finding and carries the corresponding selection optimism. Two
  preregistered criteria were narrowly missed and are reported as recorded.

## The Two Editions

Identical science, different solver. The pipeline package, the ten diagnostics, the router,
the certifier, and the frozen validated system are byte-identical between them; they differ
only in how the ensemble is fitted.

| | [`cpu/`](cpu/) | [`gpu/`](gpu/) |
|---|---|---|
| Estimator ensemble | scipy trust-region-reflective, multiprocessing across cells | batched JAX/CUDA Levenberg-Marquardt with SVD-damped steps |
| Requires | Python 3.9+ only | a CUDA device, ~8 GB VRAM at the default batch size |
| Cost per cell | ~48 s | ~2 s per fit, ~100× single-fit throughput |
| Desktop interface | yes | yes |
| macOS app | yes, see below | — |

Pick `cpu/` unless you are generating panels. Panel generation (60 cells) is the only step
where the GPU edition changes what is practical: ~45 minutes instead of hours.

## Install and Run

```bash
git clone https://github.com/eric-haotian/eis-pem-shanhaijue.git
cd eis-pem-shanhaijue/cpu
pip install -r requirements.txt
python3 bin/certify.py --system artifacts/validated_system --budget 0.60 --demo
```

The demo simulates one fresh cell, fits the eight-arm ensemble, routes every coordinate,
and applies the frozen certifier, printing the certified claims with their calibrated
probabilities and marking which were correct. It runs the full 25 × 120 grid on one core,
so budget **15–20 minutes** (measured: 16 min on an M-series MacBook Air). To check the
installation instead, `python3 bin/selftest.py` exercises every stage on a deliberately
small grid in about 90 seconds.

For the GPU edition, `cd gpu` instead and add the CUDA build of JAX that matches your
driver:

```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
python3 -c "import jax; print(jax.default_backend(), jax.devices())"   # expect: gpu
```

Each edition's own README documents the full freeze-and-validate workflow, the desktop
interface, and the deployment API.

## Desktop Interface

Double-click **`启动界面_Start_GUI.command`** (macOS) or **`启动界面_Start_GUI.bat`**
(Windows) in either edition. The launcher finds Python, installs anything missing from
`requirements.txt`, and opens the window. The interface switches between Chinese and
English at any time.

```text
input CSV -> ensemble fit -> routing -> certification -> XLSX (4 sheets)
```

The window reports which backend it is running on, marks every certification cell *out of
calibration* when the input grid differs from the frozen system's, and records in the log
every inference the CSV reader had to make.

## macOS App

The CPU edition packages into a double-clickable `.app`:

```bash
cd cpu
./script/package_macos_app.sh
open "dist/EIS-PEM ShanHaiJue.app"
```

The bundle carries the pipeline, the frozen validated system, the reference CSVs, and an
embedded Python runtime, so it launches without the user installing anything. It also
exposes each `bin/` script as a command-line entry inside `Contents/MacOS/`, and
`Contents/Resources/PipelineManifest.txt` records the exact git state, frozen system, and
package versions that went in.

Like the previous version's `package_macos_app.sh`, this is a **local-machine bundle**: the
embedded runtime is a virtual environment built from the packaging machine's Python. A
cross-machine distribution still needs a frozen runtime plus Developer ID signing and
notarization.

## CSV Contract

The canonical format is five columns, one row per frequency, each condition's spectrum
concatenated head to tail:

```text
Temperature_K,SOC,Frequency_Hz,Re_Z,Im_Z
278.15,0.1,0.001,0.019984408733,-0.005062697
```

`examples/calibration_grid_25x120.csv` is that format on the grid the frozen system was
calibrated on, so it is both the format reference and the acquisition design that carries
the certification guarantee. `examples/format_sample.csv` is a 30-row version.

Other layouts are accepted: only frequency and complex impedance are genuinely required.
Header aliases (including Chinese ones), bracketed units, sniffed delimiters, headerless
numeric files, `|Z|`+phase input, and °C-or-K and 0–1-or-0–100 auto-detection are all
handled, and every inference is reported in the log and the workbook. See each edition's
README for the full table.

## Repository Layout

```text
cpu/                       CPU edition — the reference implementation
gpu/                       GPU edition — same science, batched JAX/CUDA solver
  shanhaijue/              panel, arms, features, router, certifier, pipeline
    vendor/eispem/         the 48-parameter physics layer, 35 modules, one flat namespace
  gui/                     desktop interface
  bin/                     panel generation, routing, freezing, validation, certification
  artifacts/               the frozen validated system behind the numbers above
  examples/                reference CSVs
cpu/script/                macOS app packaging
```

The two trees differ in exactly six files — `README.md`, `requirements.txt`,
`shanhaijue/__init__.py`, `shanhaijue/arms.py`, `bin/run_panel.py`, `bin/selftest.py` —
plus `cpu/script/`, which has no GPU counterpart. Everything else, the vendored physics
layer and the frozen artifact included, is byte-identical.

## Reference

The accompanying manuscript describes the method, the five freeze–validate chains, and the
acquisition-design rules. The previous version, its benchmark comparisons, and the research
history are at
[eric-haotian/eis-pem-identification](https://github.com/eric-haotian/eis-pem-identification).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
