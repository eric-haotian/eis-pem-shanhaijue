#!/usr/bin/env python3
# learning AI at www.haotianblog.com
"""Desktop interface for ShanHaiJue identification and ShanHaiJian certification.

Double-click the launcher in the distribution root, or run:

    python3 gui/app.py

The window takes a CSV holding several spectra concatenated head to tail, fits the
estimator ensemble on each cell, routes every coordinate, applies a frozen certifier, and
writes an XLSX workbook. Fitting runs on a worker thread; the window stays responsive, the
results table fills in as cells finish, and the log records every inference the reader made.

Canonical input format:

    Temperature_K, SOC, Frequency_Hz, Re_Z, Im_Z

`examples/calibration_grid_25x120.csv` is that format on the frozen system's calibration
grid. Many other layouts are accepted too — see `csv_io` — and whatever is missing is
either inferred or filled from the panel's fallbacks, with each such decision logged.

Any spectrum is identifiable. The measured conditions are passed to the forward model
verbatim, so a single spectrum, a scattered set of temperature/SOC pairs, and the full
5 x 5 grid all produce estimates for all 48 coordinates. What does not follow the data is
the certification guarantee: a frozen certifier's probabilities and thresholds were fitted
at one information content, so when the grid differs the window reports which axis moved
and marks the certification columns out of calibration.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import csv_io  # noqa: E402
import params as PARAMS  # noqa: E402
from i18n import T, language, set_language  # noqa: E402
from results import ResultsPanel  # noqa: E402
from widgets import ParamRow, ScrollFrame, Tooltip  # noqa: E402

BACKEND = "GPU" if "jax" in (ROOT / "shanhaijue" / "arms.py").read_text() else "CPU"


# ----------------------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------------------
class Worker(threading.Thread):
    """Runs the pipeline off the UI thread, reporting through a queue."""

    def __init__(self, cfg, out_q):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.q = out_q
        self.stop_flag = threading.Event()

    def emit(self, kind, **payload):
        self.q.put(dict(kind=kind, **payload))

    def run(self):
        try:
            self._run()
        except csv_io.CsvFormatError as e:
            self.emit("error", msg=T("err_columns", cols=", ".join(e.header) or "-"))
        except Exception:                                     # noqa: BLE001
            self.emit("error", msg=traceback.format_exc(limit=4))

    # -- knob application ----------------------------------------------------------
    def _apply_solver_knobs(self, A):
        """Push the panel's solver settings into the backend module."""
        cfg = self.cfg
        if BACKEND == "CPU":
            A.MAX_NFEV = int(cfg["max_nfev"])
        else:
            A.LM_STEPS = int(cfg["max_nfev"])
            A.CHUNK = int(cfg["chunk"])
        A.N_STARTS = int(cfg["starts"])
        A.JITTER = float(cfg["jitter"])
        A.WIDE_JITTER = float(cfg["wide_jitter"])
        A.N_WIDE = int(cfg["n_wide"])

    def _run(self):
        cfg = self.cfg
        self.emit("status", text=T("reading"))
        cells, meta = csv_io.read_spectra(
            cfg["input"], split="auto", group_column=cfg["group_column"] or None,
            negate_imag=cfg["negate"], default_temperature_K=cfg["def_temp"] + 273.15,
            default_soc=cfg["def_soc"])
        if not cells:
            self.emit("error", msg=T("err_empty"))
            return
        self.emit("log", text=T("log_loaded", rows=meta["rows"], spectra=meta["spectra"],
                                cells=meta["cells"]))
        for key, fmt in meta["warnings"]:
            self.emit("log", text=T(key, **fmt))

        # panel first, so the grid is configured before any backend caches it
        from shanhaijue import panel as P
        from shanhaijue import pipeline

        frozen = pipeline.load_frozen(cfg["system"])
        ref_conditions, ref_freq = pipeline.frozen_grid(frozen)

        prior_base = getattr(P, "_GUI_PRIOR_BASE", None)
        if prior_base is None:
            prior_base = P.PRIOR_W.copy()
            P._GUI_PRIOR_BASE = prior_base
        P.PRIOR_W = prior_base * float(cfg["prior_weight"])

        ladder_name = {3: "LADDER_3", 6: "LADDER_6", 8: "LADDER_8"}[int(cfg["arms"])]
        results, t_start = [], time.time()
        n_cells, done = len(cells), 0
        any_out_of_cal = False
        last_grid = (0, 0)

        # several independent cells on the CPU backend are worth spreading over processes;
        # the GPU backend batches internally and would only contend with itself
        n_procs = int(cfg.get("procs", 1)) if BACKEND == "CPU" else 1
        if n_procs > 1 and n_cells > 1:
            self._run_parallel(cells, frozen, pipeline, ref_conditions, ref_freq,
                               cfg, n_procs, t_start)
            return

        for i, (cid, spectra) in enumerate(cells.items(), 1):
            if self.stop_flag.is_set():
                self.emit("log", text=T("log_stop_req"))
                break
            self.emit("progress", done=i - 1, total=n_cells, elapsed=time.time() - t_start)
            self.emit("status",
                      text=f"{T('fitting')} · {T('cell_progress')} {cid} ({i}/{n_cells})")

            conditions, freq = csv_io.grid_of(spectra)
            n_pts = len(conditions) * len(freq)
            self.emit("log", text=T("log_grid", cond=len(conditions), freq=len(freq),
                                    pts=n_pts))
            report = csv_io.compare_grids(conditions, freq, ref_conditions, ref_freq)
            in_cal = report["match"]
            if not in_cal:
                any_out_of_cal = True
            if i == 1 or not in_cal:
                self._report_grid(report, in_cal)
            if n_pts < 480:
                self.emit("log", text=T("warn_thin", pts=n_pts))
            self.emit("calibration", in_calibration=not any_out_of_cal)

            # the measured conditions are used verbatim, not as a temperature x SOC
            # cross product: a single spectrum and a scattered set are both admissible
            P.configure_grid(conditions=conditions, frequencies=freq)

            from shanhaijue import arms as A
            from shanhaijue import features as F
            ladder = getattr(A, ladder_name)
            self._apply_solver_knobs(A)
            self.emit("log", text=T("log_cell_start", cell=cid, arms=len(ladder) + 2))

            t0 = time.time()
            z = csv_io.stack(spectra)
            fitted, aux = A.fit_ensemble(z, seed=abs(hash(str(cid))) % (2 ** 31),
                                         ladder=ladder)
            rows = F.extract(fitted, aux, z, basis=cfg["basis"], cell_id=(str(cid), 0))

            self.emit("status", text=T("routing"))
            certified, _routed = pipeline.apply_frozen(frozen, rows, float(cfg["budget"]))
            batch = []
            for c in certified:
                batch.append(dict(
                    cell=str(cid), coord=c["coord"], family=c["family"],
                    estimate=float(10.0 ** c["est_log10"]),
                    p=round(float(c["p_certified"]), 4),
                    threshold=round(float(c["threshold"]), 4), arm=c["arm"],
                    certified=bool(c["certified"]), in_calibration=in_cal))
            results.extend(batch)
            self.emit("rows", records=batch, budget=cfg["budget"])
            done = i
            self.emit("log", text=T("log_cell_done", cell=cid, sec=time.time() - t0,
                                    n=sum(r["certified"] for r in batch)))
            last_grid = (len(conditions), len(freq))

        self.emit("progress", done=done, total=n_cells, elapsed=time.time() - t_start)
        self.emit("status", text=T("writing"))
        self.emit("write", results=results, in_calibration=not any_out_of_cal, meta=meta,
                  elapsed=time.time() - t_start,
                  conditions=last_grid[0], freqs=last_grid[1])

    def _run_parallel(self, cells, frozen, pipeline, ref_conditions, ref_freq,
                      cfg, n_procs, t_start):
        """Same pipeline, cells spread over a process pool. Routing stays in this process."""
        import concurrent.futures as cf

        import fitjob

        knobs = {k: cfg[k] for k in ("arms", "starts", "jitter", "wide_jitter", "n_wide",
                                     "max_nfev", "prior_weight", "basis")
                 if k in cfg}
        knobs.setdefault("chunk", cfg.get("chunk", 10))

        tasks, grids = [], {}
        for cid, spectra in cells.items():
            conditions, freq = csv_io.grid_of(spectra)
            grids[cid] = (conditions, freq)
            tasks.append((cid, conditions, freq, csv_io.stack(spectra), knobs))

        first = next(iter(grids.values()))
        report = csv_io.compare_grids(first[0], first[1], ref_conditions, ref_freq)
        self._report_grid(report, report["match"])
        self.emit("calibration", in_calibration=report["match"])
        self.emit("log", text=T("log_parallel", n=n_procs, cells=len(tasks)))

        results, done, any_out = [], 0, False
        with cf.ProcessPoolExecutor(max_workers=n_procs) as pool:
            futures = {pool.submit(fitjob.fit_cell, t): t[0] for t in tasks}
            for fut in cf.as_completed(futures):
                if self.stop_flag.is_set():
                    for f in futures:
                        f.cancel()
                    self.emit("log", text=T("log_stop_req"))
                    break
                cid, rows = fut.result()
                if isinstance(rows, str):
                    self.emit("log", text=T("log_error", msg=f"{cid}: {rows[9:]}"))
                    continue
                conditions, freq = grids[cid]
                in_cal = csv_io.grids_match(conditions, freq, ref_conditions, ref_freq)
                any_out = any_out or not in_cal
                certified, _ = pipeline.apply_frozen(frozen, rows, float(cfg["budget"]))
                batch = [dict(cell=str(cid), coord=c["coord"], family=c["family"],
                              estimate=float(10.0 ** c["est_log10"]),
                              p=round(float(c["p_certified"]), 4),
                              threshold=round(float(c["threshold"]), 4), arm=c["arm"],
                              certified=bool(c["certified"]), in_calibration=in_cal)
                         for c in certified]
                results.extend(batch)
                done += 1
                self.emit("rows", records=batch, budget=cfg["budget"])
                self.emit("progress", done=done, total=len(tasks),
                          elapsed=time.time() - t_start)
                self.emit("log", text=T("log_cell_done_par", cell=cid,
                                        n=sum(r["certified"] for r in batch),
                                        done=done, total=len(tasks)))

        self.emit("calibration", in_calibration=not any_out)
        self.emit("status", text=T("writing"))
        self.emit("write", results=results, in_calibration=not any_out, meta={},
                  elapsed=time.time() - t_start,
                  conditions=len(first[0]), freqs=len(first[1]))

    def _report_grid(self, r, in_cal):
        """Say which axis moved, rather than only that something did."""
        if in_cal:
            self.emit("log", text=T("log_grid_match"))
            return
        self.emit("log", text=T("grid_axis_head"))
        self.emit("log", text=T("grid_axis_temp_ok", v=csv_io.describe_axis(
            r["temperatures"], kelvin=True)) if r["temperatures_match"]
            else T("grid_axis_temp_no",
                   v=csv_io.describe_axis(r["temperatures"], kelvin=True),
                   r=csv_io.describe_axis(r["ref_temperatures"], kelvin=True)))
        self.emit("log", text=T("grid_axis_soc_ok", v=csv_io.describe_axis(r["socs"]))
                  if r["socs_match"]
                  else T("grid_axis_soc_no", v=csv_io.describe_axis(r["socs"]),
                         r=csv_io.describe_axis(r["ref_socs"])))
        self.emit("log", text=T("grid_axis_freq_ok", n=r["n_frequencies"])
                  if r["frequencies_match"]
                  else T("grid_axis_freq_no", n=r["n_frequencies"],
                         r=r["ref_n_frequencies"]))
        if r["same_size_different_design"]:
            self.emit("log", text=T("grid_same_size",
                                    pts=r["n_conditions"] * r["n_frequencies"]))
        self.emit("log", text=T("log_grid_mismatch"))


# ----------------------------------------------------------------------------------
# window
# ----------------------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q = queue.Queue()
        self.worker = None
        self.log_lines = []
        self.in_calibration = True
        self.labels = {}                # widget -> i18n key, for the language switch
        self.rows = {}                  # param name -> ParamRow
        self.vars = {}                  # param name -> tk variable
        self.specs = {p["name"]: p for p in PARAMS.build(BACKEND)}
        self.probe = None

        self.var_in = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_sys = tk.StringVar(value=str(ROOT / "artifacts" / "validated_system"))
        self.var_status = tk.StringVar(value=T("idle"))
        self.var_time = tk.StringVar(value="")
        self.var_detect = tk.StringVar(value="")

        self._build()
        self.root.after(120, self._pump)

    # -- construction --------------------------------------------------------------
    def _label(self, parent, key, **kw):
        w = ttk.Label(parent, text=T(key), **kw)
        self.labels[w] = key
        return w

    def _build(self):
        self.root.title(T("title"))
        self.root.minsize(1020, 760)
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        # ---- title bar
        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        self.title_label = ttk.Label(top, text=T("title"), font=("", 15, "bold"))
        self.title_label.grid(row=0, column=0, sticky="w")
        self.labels[self.title_label] = "title"
        ttk.Label(top, text=f"[{BACKEND}]", foreground="#556").grid(row=0, column=1, padx=8)
        self.lang_btn = ttk.Button(top, text=T("lang_button"), width=9,
                                   command=self._toggle_lang)
        self.lang_btn.grid(row=0, column=2, sticky="e")

        # ---- files
        self.f_io = ttk.LabelFrame(outer, text=T("sec_io"), padding=10)
        self.labels[self.f_io] = "sec_io"
        self.f_io.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.f_io.columnconfigure(1, weight=1)
        for r, (key, var, mode) in enumerate([("in_csv", self.var_in, "open"),
                                              ("out_xlsx", self.var_out, "save"),
                                              ("system", self.var_sys, "dir")]):
            self._label(self.f_io, key).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(self.f_io, textvariable=var).grid(row=r, column=1, sticky="ew", padx=8)
            b = ttk.Button(self.f_io, text=T("browse"), width=10,
                           command=lambda v=var, m=mode: self._browse(v, m))
            self.labels[b] = "browse"
            b.grid(row=r, column=2)
        self.detect = ttk.Label(self.f_io, textvariable=self.var_detect, foreground="#2c7a39")
        self.detect.grid(row=3, column=1, columnspan=2, sticky="w", padx=8, pady=(4, 0))
        Tooltip(self.detect, lambda: T("fmt_body"))

        # ---- tabs
        self.nb = ttk.Notebook(outer)
        self.nb.grid(row=2, column=0, sticky="nsew", pady=(10, 0))

        self.tab_params = ScrollFrame(self.nb)
        self.nb.add(self.tab_params, text=T("tab_params"))
        self._build_params(self.tab_params.body)

        self.results = ResultsPanel(self.nb, T, padding=8)
        self.nb.add(self.results, text=T("tab_results"))

        logf = ttk.Frame(self.nb, padding=6)
        self.nb.add(logf, text=T("tab_log"))
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log = tk.Text(logf, height=10, wrap="word", font=("Menlo", 11))
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

        # ---- status
        self.f_st = ttk.Frame(outer)
        self.f_st.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.f_st.columnconfigure(0, weight=1)
        self.bar = ttk.Progressbar(self.f_st, mode="determinate", maximum=100)
        self.bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(self.f_st, textvariable=self.var_time, width=36,
                  anchor="e").grid(row=0, column=1, padx=(10, 0))
        ttk.Label(self.f_st, textvariable=self.var_status,
                  foreground="#234").grid(row=1, column=0, columnspan=2, sticky="w",
                                          pady=(6, 0))
        self.cal_banner = ttk.Label(self.f_st, text="", foreground="#a33", wraplength=940)
        self.cal_banner.grid(row=2, column=0, columnspan=2, sticky="w")

        # ---- buttons
        btns = ttk.Frame(outer)
        btns.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        self.btn_start = ttk.Button(btns, text=T("start"), command=self._start)
        self.labels[self.btn_start] = "start"
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(btns, text=T("stop"), command=self._stop, state="disabled")
        self.labels[self.btn_stop] = "stop"
        self.btn_stop.pack(side="left", padx=8)
        self.btn_open = ttk.Button(btns, text=T("open_out"), command=self._open_out)
        self.labels[self.btn_open] = "open_out"
        self.btn_open.pack(side="left")
        self.btn_reset = ttk.Button(btns, text=T("reset"), command=self._reset)
        self.labels[self.btn_reset] = "reset"
        self.btn_reset.pack(side="right")
        self.btn_clear = ttk.Button(btns, text=T("clear_log"), command=self._clear_log)
        self.labels[self.btn_clear] = "clear_log"
        self.btn_clear.pack(side="right", padx=8)

    def _build_params(self, body):
        """One LabelFrame per section, one ParamRow per tunable."""
        body.columnconfigure(0, weight=1)
        self.section_frames = {}
        for si, (skey, _zh) in enumerate(PARAMS.SECTIONS):
            entries = [s for s in self.specs.values() if s["section"] == skey]
            if not entries:
                continue
            frame = ttk.LabelFrame(body, text=T(skey), padding=10)
            frame.grid(row=si, column=0, sticky="ew", pady=(0, 10), padx=2)
            frame.columnconfigure(2, weight=1)
            self.labels[frame] = skey
            self.section_frames[skey] = frame
            for ri, spec in enumerate(entries):
                var = self._make_var(spec)
                self.vars[spec["name"]] = var
                self.rows[spec["name"]] = ParamRow(frame, ri, spec, var, T)

    def _make_var(self, spec):
        kind = spec["vartype"]
        if kind == "bool":
            return tk.BooleanVar(value=spec["default"])
        if kind == "int":
            return tk.IntVar(value=spec["default"])
        if kind == "str":
            return tk.StringVar(value=spec["default"])
        return tk.DoubleVar(value=spec["default"])

    # -- helpers -------------------------------------------------------------------
    def _toggle_lang(self):
        set_language("en" if language() == "zh" else "zh")
        for w, key in self.labels.items():
            try:
                w.configure(text=T(key))
            except tk.TclError:
                pass
        for row in self.rows.values():
            row.retranslate()
        self.results.retranslate()
        self.root.title(T("title"))
        self.lang_btn.configure(text=T("lang_button"))
        for i, key in enumerate(("tab_params", "tab_results", "tab_log")):
            self.nb.tab(i, text=T(key))
        self._apply_probe()
        self._set_banner()
        if self.worker is None:
            self.var_status.set(T("idle"))

    def _set_banner(self):
        self.cal_banner.configure(text="" if self.in_calibration else T("banner_out_cal"))

    def _browse(self, var, mode):
        if mode == "open":
            p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if p:
                var.set(p)
                if not self.var_out.get():
                    self.var_out.set(str(Path(p).with_suffix("")) + "_certified.xlsx")
                self._probe_file(p)
            return
        if mode == "save":
            p = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel", "*.xlsx")])
        else:
            p = filedialog.askdirectory()
        if p:
            var.set(p)

    def _probe_file(self, path):
        """Read the file's head and show what it supplies, so the fallbacks can't mislead."""
        self.probe = csv_io.probe(path, group_column=self.vars["group_column"].get() or None)
        self._apply_probe()

    def _apply_probe(self):
        p = self.probe
        if p is None:
            self.var_detect.set("")
            for name in ("def_temp", "def_soc"):
                self.rows[name].set_enabled(True)
            return
        if p.get("error") or not p["found"].get("freq"):
            self.var_detect.set(T("probe_bad"))
            self.detect.configure(foreground="#a33")
            return
        names = {"freq": "f", "zre": "Z'", "zim": "Z''", "temp": "T", "soc": "SOC",
                 "cell": "cell"}
        have = [names[k] for k in ("freq", "zre", "zim", "temp", "soc", "cell")
                if p["found"].get(k)]
        self.var_detect.set(T("probe_ok", cols=" · ".join(have)))
        self.detect.configure(foreground="#2c7a39")
        # the fallbacks are inert whenever the file itself carries the quantity
        self.rows["def_temp"].set_enabled(
            not p["found"].get("temp"),
            "probe_from_csv" if p["found"].get("temp") else "probe_fallback")
        self.rows["def_soc"].set_enabled(
            not p["found"].get("soc"),
            "probe_from_csv" if p["found"].get("soc") else "probe_fallback")

    def _append_log(self, text):
        line = f"[{time.strftime('%H:%M:%S')}] {text}"
        self.log_lines.append(line)
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log_lines.clear()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _reset(self):
        for name, spec in self.specs.items():
            self.vars[name].set(spec["default"])

    def _open_out(self):
        p = Path(self.var_out.get() or ROOT)
        folder = p.parent if p.suffix else p
        if sys.platform == "darwin":
            subprocess.run(["open", str(folder)], check=False)
        elif os.name == "nt":
            os.startfile(str(folder))                                    # noqa: S606
        else:
            subprocess.run(["xdg-open", str(folder)], check=False)

    # -- run -----------------------------------------------------------------------
    def _collect(self):
        cfg = {}
        for name, spec in self.specs.items():
            if spec["scope"] == "refreeze":
                continue                       # inert for this run, by construction
            try:
                cfg[name] = self.vars[name].get()
            except tk.TclError:                # a half-typed spinbox value
                cfg[name] = spec["default"]
        return cfg

    def _start(self):
        if not self.var_in.get():
            messagebox.showwarning(T("title"), T("err_no_input"))
            return
        system = Path(self.var_sys.get())
        if system.is_dir():
            system = system / "frozen_system.joblib"
        if not system.exists():
            messagebox.showwarning(T("title"), T("err_no_system"))
            return
        if not self.var_out.get():
            self.var_out.set(str(Path(self.var_in.get()).with_suffix("")) + "_certified.xlsx")

        self.in_calibration = True
        self._set_banner()
        self.results.clear()
        self.bar.configure(value=0)
        self.run_start = time.time()
        self.prog_done, self.prog_total = 0, 0

        cfg = self._collect()
        cfg.update(input=self.var_in.get(), output=self.var_out.get(), system=str(system))
        changed = [n for n, s in self.specs.items()
                   if s["scope"] == "inference" and n in cfg
                   and str(cfg[n]) != str(s["default"])]
        risky = [n for n in changed if self.specs[n]["safety"] == "recalibrate"]
        if risky:
            self._append_log(T("log_knobs_changed", names=", ".join(
                T(self.specs[n]["label_key"]) for n in risky)))

        self.last_cfg = cfg
        self.worker = Worker(cfg, self.q)
        self.worker.start()
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.nb.select(1)                      # show the table the run is about to fill

    def _stop(self):
        if self.worker:
            self.worker.stop_flag.set()
            self.btn_stop.configure(state="disabled")

    def _finish(self, msg_key: str):
        if hasattr(self, "run_start"):
            self._update_time()               # freeze the clock at the true final elapsed
        self.worker = None
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.var_status.set(T(msg_key))

    # -- queue pump ----------------------------------------------------------------
    def _pump(self):
        try:
            while True:
                m = self.q.get_nowait()
                k = m["kind"]
                if k == "log":
                    self._append_log(m["text"])
                elif k == "status":
                    self.var_status.set(m["text"])
                elif k == "rows":
                    self.results.add(m["records"], budget=m["budget"])
                elif k == "calibration":
                    self.in_calibration = m["in_calibration"]
                    self._set_banner()
                elif k == "progress":
                    self.prog_done, self.prog_total = m["done"], m["total"]
                    self.bar.configure(value=100.0 * m["done"] / max(m["total"], 1))
                elif k == "error":
                    self._append_log(T("log_error", msg=m["msg"]))
                    self.nb.select(2)
                    messagebox.showerror(T("title"), m["msg"])
                    self._finish("failed")
                elif k == "write":
                    self._write(m)
        except queue.Empty:
            pass
        if self.worker is not None:
            self._update_time()               # tick the clock every frame, not only on progress
        self.root.after(120, self._pump)

    def _update_time(self):
        """Elapsed ticks off the wall clock; remaining is only shown once it can be estimated.

        A single cell emits progress just twice — at its start and its end — so nothing between
        can predict how long the fit will take. Rather than show a frozen 0:00, the clock keeps
        counting elapsed and leaves remaining as a dash until at least one cell has finished.
        """
        el = time.time() - self.run_start
        done, total = self.prog_done, self.prog_total
        count = f"{done}/{total}" if total else "…"
        if done > 0 and total > done:
            rem = el / done * (total - done)
            rem_str = f"{int(rem) // 60}:{int(rem) % 60:02d}"
        elif total and done >= total:
            rem_str = "0:00"
        else:
            rem_str = "—"
        self.var_time.set(
            f"{T('cell_progress')} {count} · "
            f"{T('elapsed')} {int(el) // 60}:{int(el) % 60:02d} · "
            f"{T('remaining')} {rem_str}")

    def _write(self, m):
        cfg = getattr(self, "last_cfg", None) or self._collect()
        summary = [(T("in_csv"), self.var_in.get()), (T("system"), self.var_sys.get()),
                   ("backend", BACKEND),
                   (T("sum_grid"), f"{m['conditions']} x {m['freqs']}"),
                   (T("col_status"),
                    T("val_in_cal") if m["in_calibration"] else T("val_out_cal")),
                   (T("elapsed"), f"{m['elapsed']:.0f} s")]
        summary += [(T(self.specs[n]["label_key"]), cfg[n]) for n in sorted(cfg)
                    if n in self.specs]
        try:
            path = csv_io.write_xlsx(self.var_out.get(), m["results"], summary,
                                     self.log_lines, T)
            self._append_log(T("log_written", path=path))
            self._finish("done")
        except Exception as e:                                           # noqa: BLE001
            self._append_log(T("log_error", msg=str(e)))
            messagebox.showerror(T("title"), str(e))
            self._finish("failed")


def main():
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.3)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
