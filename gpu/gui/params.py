# learning AI at www.haotianblog.com
"""The tunable panel, declared as data.

Each entry becomes one row in the window: label, control, and a one-line directional hint,
with the full statement — meaning, both trend directions, compute cost, caveat — in the
hover text. Adding a knob means adding an entry here and three strings to `i18n`.

Two fields carry the load and must be filled honestly:

`scope`     'inference' knobs act on the run the Start button performs. 'refreeze' knobs
            only matter while building a new system with the `bin/` scripts; they are shown
            for completeness, marked as inert for this run, and never silently applied.

`safety`    'safe' means changing it does not move the feature distribution that the frozen
            router and certifier were calibrated against. 'recalibrate' means the run still
            produces estimates but the certification guarantee no longer transfers, so the
            panel says so at the point of use. Claiming 'safe' wrongly is the one error that
            would let the interface mislead, so when in doubt it is not safe.
"""
from __future__ import annotations

# section key -> (order, i18n key for the section title)
SECTIONS = [
    ("sec_input", "输入解析"),
    ("sec_ensemble", "估计器集成"),
    ("sec_solver", "求解器"),
    ("sec_claim", "声明与认证"),
    ("sec_compute", "算力"),
    ("sec_refreeze", "重新冻结（不影响本次运行）"),
]


def P(name, section, widget, default, *, scope="inference", safety="safe",
      min=None, max=None, step=None, choices=None, vartype="double", danger=False,
      backend=None):
    """One panel entry. i18n keys are derived from the name to keep them in lockstep."""
    return dict(name=name, section=section, widget=widget, default=default,
                scope=scope, safety=safety, min=min, max=max, step=step,
                choices=choices, vartype=vartype, danger=danger, backend=backend,
                label_key=f"p_{name}", hint_key=f"p_{name}_hint",
                tooltip_key=f"p_{name}_tip")


# The list is filled in below by `build()`, which is what the window imports. Keeping it
# behind a function lets the backend (CPU vs GPU) drop the knobs that do not exist for it
# rather than showing controls that would do nothing.
def build(backend: str) -> list:
    p = [
        # ---- input parsing -------------------------------------------------------
        P("negate", "sec_input", "checkbox", False, vartype="bool"),
        P("def_temp", "sec_input", "spinbox", 25.0, min=-40, max=80, step=5),
        P("def_soc", "sec_input", "spinbox", 0.5, min=0.0, max=1.0, step=0.05),
        P("group_column", "sec_input", "entry", "", vartype="str"),

        # ---- estimator ensemble --------------------------------------------------
        P("arms", "sec_ensemble", "radio", 8, choices=(3, 6, 8), vartype="int"),
        P("starts", "sec_ensemble", "spinbox", 3, min=1, max=12, step=1, vartype="int",
          safety="recalibrate"),
        P("jitter", "sec_ensemble", "spinbox", 0.15, min=0.0, max=1.0, step=0.05,
          safety="recalibrate"),
        P("wide_jitter", "sec_ensemble", "spinbox", 0.35, min=0.0, max=2.0, step=0.05,
          safety="recalibrate"),
        P("n_wide", "sec_ensemble", "spinbox", 3, min=0, max=12, step=1, vartype="int",
          safety="recalibrate"),
        P("prior_weight", "sec_ensemble", "spinbox", 1.0, min=0.1, max=10.0, step=0.1,
          safety="recalibrate", danger=True),

        # ---- solver --------------------------------------------------------------
        P("max_nfev", "sec_solver", "spinbox", 400 if backend == "CPU" else 100,
          min=10, max=4000, step=10, vartype="int", safety="recalibrate"),
        P("noise", "sec_solver", "spinbox", 0.005, min=0.0001, max=0.2, step=0.001,
          scope="refreeze"),

        # ---- claim and certification ---------------------------------------------
        P("budget", "sec_claim", "combobox", "0.60",
          choices=("0.32", "0.50", "0.55", "0.60", "0.65"), vartype="str"),
        P("basis", "sec_claim", "combobox", "phys", choices=("phys", "raw"),
          vartype="str", safety="recalibrate", danger=True),
        P("tolerance", "sec_claim", "spinbox", 0.10, min=0.01, max=0.5, step=0.01,
          scope="refreeze", danger=True),

        # ---- compute -------------------------------------------------------------
    ]
    if backend == "CPU":
        p.append(P("procs", "sec_compute", "spinbox", 4, min=1, max=64, step=1,
                   vartype="int", backend="CPU"))
    else:
        p.append(P("chunk", "sec_compute", "spinbox", 10, min=1, max=256, step=1,
                   vartype="int", backend="GPU"))

    # ---- re-freeze only ----------------------------------------------------------
    p += [
        P("n_folds", "sec_refreeze", "spinbox", 5, min=2, max=20, step=1, vartype="int",
          scope="refreeze"),
        P("n_boot", "sec_refreeze", "spinbox", 2000, min=200, max=20000, step=200,
          vartype="int", scope="refreeze"),
        P("seed", "sec_refreeze", "spinbox", 0, min=0, max=10 ** 9, step=1, vartype="int",
          scope="refreeze"),
    ]
    return p
