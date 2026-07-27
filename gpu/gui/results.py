# learning AI at www.haotianblog.com
"""The in-window results table.

Every coordinate the run touches lands here as it is produced, not only at the end, so a
long run is readable while it is still going. Certified claims and abstentions are both
shown: an abstention carries information (the system declined to claim that coordinate at
this budget) and hiding it would misrepresent what the method does.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

COLUMNS = ("cell", "coord", "family", "estimate", "p", "threshold", "arm", "certified")
NUMERIC = {"estimate", "p", "threshold"}
WIDTHS = {"cell": 110, "coord": 150, "family": 130, "estimate": 120, "p": 100,
          "threshold": 90, "arm": 100, "certified": 110}
HEADER_KEY = {"cell": "col_cell", "coord": "col_param", "family": "col_family",
              "estimate": "col_value", "p": "col_prob", "threshold": "col_thr",
              "arm": "col_arm", "certified": "col_certified"}


def _fmt(value) -> str:
    """Physical estimates span many decades, so significant figures beat fixed decimals."""
    if isinstance(value, float):
        if value == 0:
            return "0"
        a = abs(value)
        return f"{value:.6g}" if 1e-4 <= a < 1e6 else f"{value:.4e}"
    return str(value)


class ResultsPanel(ttk.Frame):
    def __init__(self, parent, T, **kw):
        super().__init__(parent, **kw)
        self.T = T
        self.rows = []
        self.sort_by = None
        self.sort_desc = False
        self.filter = tk.StringVar(value="all")
        self.budget = ""

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.summary = ttk.Label(bar, text=T("res_none"))
        self.summary.pack(side="left")
        self.hint = ttk.Label(bar, text=T("res_sort_hint"), foreground="#7a828c")
        self.hint.pack(side="right", padx=(10, 0))
        self.radios = []
        for key, value in (("res_filter_all", "all"), ("res_filter_cert", "certified"),
                           ("res_filter_abst", "abstained")):
            rb = ttk.Radiobutton(bar, text=T(key), value=value, variable=self.filter,
                                 command=self.refresh)
            rb.pack(side="right", padx=(12, 0))
            self.radios.append((rb, key))

        self.tree = ttk.Treeview(self, columns=COLUMNS, show="headings", selectmode="extended")
        for c in COLUMNS:
            self.tree.heading(c, text=T(HEADER_KEY[c]), command=lambda col=c: self._sort(col))
            self.tree.column(c, width=WIDTHS[c],
                             anchor="e" if c in NUMERIC else "w", stretch=(c == "coord"))
        self.tree.grid(row=1, column=0, sticky="nsew")

        ysb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        ysb.grid(row=1, column=1, sticky="ns")
        xsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        xsb.grid(row=2, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.tree.tag_configure("certified", background="#e7f6e9")
        self.tree.tag_configure("abstained", foreground="#6b7280")
        self.tree.tag_configure("outcal", background="#fdf2e2")

        self.tree.bind("<Control-c>", self._copy)
        self.tree.bind("<Command-c>", self._copy)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    # -- data ----------------------------------------------------------------------
    def clear(self):
        self.rows.clear()
        self.tree.delete(*self.tree.get_children())
        self.summary.configure(text=self.T("res_none"))

    def add(self, records, budget=""):
        """Append one cell's coordinates and repaint."""
        self.budget = budget or self.budget
        self.rows.extend(records)
        self.refresh()

    def refresh(self):
        mode = self.filter.get()
        rows = [r for r in self.rows
                if mode == "all" or (mode == "certified") == bool(r["certified"])]
        if self.sort_by:
            key = self.sort_by
            rows.sort(key=lambda r: (r[key] if key in NUMERIC else str(r[key])),
                      reverse=self.sort_desc)

        self.tree.delete(*self.tree.get_children())
        for r in rows:
            tags = ["certified" if r["certified"] else "abstained"]
            if r["certified"] and not r.get("in_calibration", True):
                tags.append("outcal")
            self.tree.insert("", "end", tags=tags, values=(
                r["cell"], r["coord"], r["family"], _fmt(r["estimate"]),
                f"{r['p']:.4f}", f"{r['threshold']:.4f}", r["arm"],
                self.T("val_yes") if r["certified"] else self.T("val_no")))

        if self.rows:
            cells = len({r["cell"] for r in self.rows})
            claimed = sum(1 for r in self.rows if r["certified"])
            self.summary.configure(text=self.T(
                "res_summary", cells=cells, coords=len(self.rows) // max(cells, 1),
                claimed=claimed, budget=self.budget))

    # -- interaction ---------------------------------------------------------------
    def _sort(self, column):
        self.sort_desc = not self.sort_desc if self.sort_by == column else False
        self.sort_by = column
        self.refresh()

    def _copy(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return "break"
        lines = ["\t".join(self.T(HEADER_KEY[c]) for c in COLUMNS)]
        lines += ["\t".join(str(v) for v in self.tree.item(i, "values")) for i in sel]
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.summary.configure(text=self.T("res_copied", n=len(sel)))
        return "break"

    def retranslate(self):
        for c in COLUMNS:
            self.tree.heading(c, text=self.T(HEADER_KEY[c]))
        for rb, key in self.radios:
            rb.configure(text=self.T(key))
        self.hint.configure(text=self.T("res_sort_hint"))
        self.refresh()
        if not self.rows:
            self.summary.configure(text=self.T("res_none"))
