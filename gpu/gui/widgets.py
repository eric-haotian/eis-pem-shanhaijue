# learning AI at www.haotianblog.com
"""Small tkinter helpers shared by the interface.

`Tooltip` gives every control a hover explanation, and `ParamRow` lays out one tunable as
label + widget + a one-line directional hint, which is the layout the whole parameter panel
is built from. Both re-read their text through the translation callable when the language
changes, so nothing has to be rebuilt on a language switch.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Tooltip:
    """Hover text for a widget, positioned under the pointer.

    Text is supplied as a zero-argument callable rather than a string so a language switch
    is picked up on the next hover without re-registering anything.
    """

    DELAY_MS = 450
    WRAP_PX = 420

    def __init__(self, widget, text_fn):
        self.widget = widget
        self.text_fn = text_fn
        self.tip = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self.after_id = self.widget.after(self.DELAY_MS, self._show)

    def _cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def _show(self):
        text = self.text_fn()
        if not text or self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(self.tip, background="#2b2b2b", padx=1, pady=1)
        frame.pack()
        tk.Label(frame, text=text, justify="left", wraplength=self.WRAP_PX,
                 background="#fbfbe8", foreground="#1a1a1a", padx=9, pady=7,
                 font=("", 11)).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ParamRow:
    """One tunable: label, widget, and a short directional hint on the same line.

    The hint states which way the result moves, so the panel is readable without hovering;
    the tooltip carries the full statement including compute cost and any caveat.
    """

    def __init__(self, parent, row, spec, variable, T, on_change=None):
        self.spec = spec
        self.T = T
        self.variable = variable
        self.widgets = []

        self.label = ttk.Label(parent, text=T(spec["label_key"]))
        self.label.grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))

        kind = spec["widget"]
        if kind == "spinbox":
            w = ttk.Spinbox(parent, from_=spec["min"], to=spec["max"],
                            increment=spec.get("step", 1), width=10, textvariable=variable)
        elif kind == "combobox":
            w = ttk.Combobox(parent, textvariable=variable, width=12, state="readonly",
                             values=list(spec["choices"]))
        elif kind == "checkbox":
            w = ttk.Checkbutton(parent, variable=variable, text="")
        elif kind == "radio":
            w = ttk.Frame(parent)
            for value in spec["choices"]:
                ttk.Radiobutton(w, text=str(value), value=value,
                                variable=variable).pack(side="left", padx=(0, 10))
        else:
            w = ttk.Entry(parent, textvariable=variable, width=14)
        w.grid(row=row, column=1, sticky="w")
        self.widget = w
        self.widgets.append(w)

        self.hint = ttk.Label(parent, text=T(spec["hint_key"]), foreground="#5a6270")
        self.hint.grid(row=row, column=2, sticky="w", padx=(12, 0))

        for target in (self.label, w, self.hint):
            Tooltip(target, lambda s=spec: self.T(s["tooltip_key"]))

        if spec.get("danger"):
            self.label.configure(foreground="#a33")

        if on_change is not None:
            variable.trace_add("write", lambda *_a: on_change(spec["name"]))

    def retranslate(self):
        self.label.configure(text=self.T(self.spec["label_key"]))
        self.hint.configure(text=self.T(self.spec["hint_key"]))

    def set_enabled(self, enabled: bool, note_key=None):
        """Grey the control out — used when the CSV itself supplies the quantity."""
        state = "!disabled" if enabled else "disabled"
        for w in self.widgets:
            try:
                w.state([state])
            except tk.TclError:
                w.configure(state="normal" if enabled else "disabled")
        self.label.configure(foreground="" if enabled else "#9aa0a8")
        if note_key is not None:
            self.hint.configure(text=self.T(note_key),
                                foreground="#2c7a39" if not enabled else "#5a6270")
        else:
            self.hint.configure(text=self.T(self.spec["hint_key"]), foreground="#5a6270")


class ScrollFrame(ttk.Frame):
    """A vertically scrollable container, so a tall parameter panel stays usable."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.bar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.body.bind("<Configure>",
                       lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        for target in (self.canvas, self.body):
            target.bind("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta if abs(event.delta) < 20
                                       else event.delta // 120), "units")
