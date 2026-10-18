#!/usr/bin/env python3
"""
NanoSlicer
----------
One app for the whole chain

    continuous phase  ->  quantized height map (PNG / STL)  ->  sliced fields  ->  Nanoscribe job files

    Tab 1  Quantizer           quantizer.py              load, quantize, preview, export PNG / STL
    Tab 2  Job File Generator  slicing method + recipe + slicer preview -> job.gwl, data.gwl and field files

Two ways to use it
    Quantizer only                 load -> quantize -> preview -> export
    Quantizer -> Job File Generator   "Send to Job File Generator" hands the quantized result over (state.AppState)
    Job File Generator on its own  open an STL or a height map directly in tab 2

Launch
    python nanoslicer.py                              # GUI
    python nanoslicer.py quantize ...                 # the quantizer command line (same arguments as before)
    python nanoslicer.py stl INPUT PIXEL_UM [--height-max H] [--levels N] [--base B]
    python nanoslicer.py preview INPUT OUT.png [--method standard|adaptive] ...   # slicer preview as an image
    python nanoslicer.py job INPUT OUTDIR --pixel-size P --height-max H [--levels N] [--method standard|adaptive]
                            [--recipe NAME] [--set key=value ...]
    python nanoslicer.py recipes [list | show NAME | delete NAME]
"""
import argparse
import queue
import sys
import threading
from dataclasses import fields as dc_fields
from pathlib import Path

import gradient_segmentation as gs
import height_map_stl as hms
import job_generator as jg
import route_solvers as rs
import quantizer as pq
import slicer_preview as sp
from recipes import DEFAULT_NAME, RECIPE_CHOICES, Recipe, RecipeStore
from state import AppState, Design, design_from_file

try:                                                   # the GUI is optional (CLI / library work without tkinter)
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
    from PIL import ImageTk
except ImportError:
    tk = ttk = filedialog = messagebox = simpledialog = ImageTk = None

APP_NAME = "NanoSlicer"
PAD = 10
PREVIEW_QUICK, PREVIEW_FULL = 700, 1100               # max pixels per side: fast first look, then (nearly) exact

# (key, label, kind) in the order shown in the recipe panel
RECIPE_FIELDS = [
    ("objective", "Objective", str), ("mode", "Mode", str), ("resin", "Resin", str),
    ("scan_speed", "Scan speed (µm/s)", float), ("laser_power", "Laser power", float),
    ("slicing_distance", "Slicing distance (µm)", float), ("hatching_distance", "Hatching distance (µm)", float),
    ("shells", "Shells", int), ("hatch_angle", "Hatch angle (°, -1 = random)", float),
    ("field_size", "Field size (µm), standard", float), ("adaptive_radius", "Field radius (µm), adaptive", float),
    ("scan_field_diameter", "Scan field Ø (µm)", float), ("overlap", "Field overlap (µm)", float),
    ("interface_pos", "FindInterfaceAt (µm, -1 = off)", float), ("piezo_settling_time", "Piezo settling time (ms)", float),
    ("galvo_acceleration", "Galvo acceleration", float), ("stage_velocity", "Stage velocity (µm/s)", float),
    ("base_slicing_distance", "Base slicing (µm, 0 = same), height-based", float),
    ("base_power_multiplier", "Base power ×, height-based", float),
    ("block_shear_angle", "Block shear angle (°)", float), ("block_shear_direction", "Block shear direction (°, 0 = +x)", float),
]
BLOCK_SHEAR_TIP = ("Block shear angle\n"
                   "How far the block faces lean from the substrate normal, so the seams between blocks are inclined "
                   "(DeScribe: stitching with trapezoidal blocks avoids the shadowing of the objective's light cone). "
                   "0 = vertical seams (default, output unchanged); DeScribe recommends 15-30 deg for its block stitching. "
                   "The direction is the angle in the xy plane the blocks lean towards (0 = +x, 90 = +y). Fields, centres and "
                   "writing order do not change; the seam between two fields leans by height x tan(angle). Only matters for "
                   "tall structures: a 1 um design moves the seam by well under 1 um.")
SEG_LABELS = {"footprint_edges": "Footprint edges (default)", "height_threshold": "Height threshold at z",
              "height_gradient_guided": "Height gradient-guided", "phase_quality_guided": "Phase-quality-guided"}
SEG_NOTES = {
    "footprint_edges": "Adaptive edge voting on the footprint of the structure.",
    "height_threshold": "Edge voting on the cross-section at z (base included): the seams avoid the features above the base. "
                        "Below z the recipe's base slicing and base power apply. Only the outline of that one level is seen.",
    "height_gradient_guided": "Reads the full height map: the block borders are attracted to every height step (all quantized "
                              "levels), and avoid cutting through smooth areas. Blocks stay inside the writing field.",
    "phase_quality_guided": "As gradient-guided, then every border between two blocks is re-optimised as a minimum-cost seam "
                            "(graph cut): cheap on height steps, dear in smooth phase areas; connected blocks, short writing order.",
}
SLICING_FIELDS = ("field_size", "adaptive_radius", "overlap")     # recipe fields that change the layout
METHOD_TEXT = {
    "standard": "Rectangular fields of the recipe's field size, written row by row (DeScribe-style).",
    "adaptive": "TipSlicer adaptive stitching: field centres chosen by edge voting so the seams avoid the features.",
}
NO_INPUT_TEXT = ("No structure yet.\n\nQuantize a file in the Quantizer tab and press\n“Send to Job File Generator”,\n"
                 "or choose an STL / height map under Source → From file.")


# ==========================================================================
# GUI
# ==========================================================================
class ImageView(ttk.Frame if ttk else object):
    """A canvas that shows a PIL image scaled to fit (aspect kept), or a message."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0, width=420, height=420)
        self.canvas.pack(fill="both", expand=True)
        self.caption = ttk.Label(self, text=" ", foreground="#888", wraplength=520, justify="left")
        self.caption.pack(anchor="w", pady=(4, 0))
        self.img, self.message, self._photo = None, NO_INPUT_TEXT, None
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def show(self, img, caption=" "):
        self.img, self.message = img, None
        self.caption.config(text=caption)
        self.redraw()

    def say(self, text):
        self.img, self.message = None, text
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if self.img is not None:
            scale = min(w / self.img.width, h / self.img.height)
            size = (max(1, int(self.img.width * scale)), max(1, int(self.img.height * scale)))
            self._photo = ImageTk.PhotoImage(self.img.resize(size))
            c.create_image(w // 2, h // 2, image=self._photo)
        if self.message:
            c.create_text(w // 2, 18 if self.img is not None else h // 2, text=self.message, fill="#ddd", justify="center")


class JobFrame(ttk.Frame if ttk else object):
    """Tab 2: source + recipe + slicing method -> slicer preview -> Nanoscribe job files (and height map -> STL)."""

    def __init__(self, master, state=None, store=None):
        super().__init__(master, padding=PAD)
        self.state = state or AppState()
        self.store = store or RecipeStore()
        self.v = {}                                     # tk variables by name
        self.choice_boxes = {}                          # recipe field -> drop-down (objective, mode)
        self.events = queue.Queue()                     # worker threads -> UI
        self.cancel = threading.Event()
        self.busy = False
        self._loaded = None                             # recipe as loaded, to show "modified"
        self._quiet = False
        self._preview_after = None
        self._preview_serial = 0
        self._on_done = None
        self._last_design = None                        # object identity of the design last published by the Quantizer
        self._build()
        self.select_recipe(DEFAULT_NAME)
        self.state.subscribe(self._state_changed)
        self._seg_update()
        self._source_changed()
        self._poll()

    # ------------------------------------------------------------ helpers
    def _var(self, name, value, cls=tk.StringVar, preview=False):
        self.v[name] = cls(value=value)
        if preview:
            self.v[name].trace_add("write", lambda *_: self.schedule_preview())
        return self.v[name]

    def _section(self, parent, title):
        ttk.Label(parent, text=title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(PAD, 2))
        f = ttk.Frame(parent)
        f.pack(fill="x")
        f.columnconfigure(1, weight=1)
        return f

    def _row(self, parent, label, name, values=None, default="", button=None, readonly=False, width=14, preview=True):
        r = parent.grid_size()[1]
        lab = ttk.Label(parent, text=label)
        lab.grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        box = ttk.Frame(parent)
        box.grid(row=r, column=1, sticky="ew", pady=2)
        var = self._var(name, default, preview=preview)
        if values is not None:
            w = ttk.Combobox(box, textvariable=var, values=values, width=width, state="readonly" if readonly else "normal")
        else:
            w = ttk.Entry(box, textvariable=var, width=width)
        w.pack(side="left", fill="x", expand=True)
        if button:
            ttk.Button(box, text=button[0], width=9, command=button[1]).pack(side="left", padx=(4, 0))
        return lab, box, w

    # ------------------------------------------------------------ layout
    def _build(self):
        cols = ttk.Frame(self)
        cols.pack(fill="both", expand=True)
        self.left_scroll = pq.ScrollFrame(cols)           # the long settings column scrolls when the window is short
        self.left_scroll.pack(side="left", fill="y", padx=(0, PAD))
        left = self.left_scroll.body
        ttk.Separator(cols, orient="vertical").pack(side="left", fill="y")
        right = ttk.Frame(cols, padding=(PAD, 0, 0, 0))
        right.pack(side="right", fill="y")
        ttk.Separator(cols, orient="vertical").pack(side="right", fill="y")
        centre = ttk.Frame(cols, padding=(PAD, PAD, PAD, 0))
        centre.pack(side="left", fill="both", expand=True)

        # ---- source
        f = self._section(left, "Source")
        self._var("source", "file")
        seg = ttk.Frame(f)
        seg.grid(row=0, column=0, columnspan=2, sticky="w")
        self.rb_quant = ttk.Radiobutton(seg, text="From Quantizer", value="quantizer", variable=self.v["source"],
                                        command=self._source_changed, state="disabled")
        self.rb_quant.pack(side="left", padx=(0, 14))
        ttk.Radiobutton(seg, text="From file", value="file", variable=self.v["source"], command=self._source_changed).pack(side="left")
        self.quant_info = ttk.Label(f, text="", foreground="#555", wraplength=360, justify="left")
        self.quant_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.file_rows = []
        self.file_rows.append(self._row(f, "File", "input", button=("Browse…", self.browse_input)))
        self.key_row = self._row(f, "MAT variable", "key", values=[""])
        self.input_hint = ttk.Label(f, text="", foreground="#888", wraplength=360, justify="left")
        self.input_hint.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")
        self.v["input"].trace_add("write", lambda *_: self._input_changed())

        # ---- design
        f = self._section(left, "Design")
        self.design_rows = [
            self._row(f, "Pixel size (µm)", "pitch"),
            self._row(f, "Height max (µm)", "height_max", default="1"),
            self._row(f, "Levels", "levels", values=["", "2", "3", "4", "6", "8", "16"], default="6"),
        ]
        self._row(f, "Base thickness (µm)", "base", default="0.2")
        self.design_note = ttk.Label(f, text="", foreground="#888", wraplength=360, justify="left")
        self.design_note.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")

        # ---- slicing method
        f = self._section(left, "Slicing method")
        self._var("method", "standard")
        seg = ttk.Frame(f)
        seg.grid(row=0, column=0, columnspan=2, sticky="w")
        for text, val in (("Standard", "standard"), ("Adaptive", "adaptive")):
            ttk.Radiobutton(seg, text=text, value=val, variable=self.v["method"],
                            command=self._method_changed).pack(side="left", padx=(0, 14))
        self.method_text = ttk.Label(f, text=METHOD_TEXT["standard"], foreground="#888", wraplength=360, justify="left")
        self.method_text.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.seg_row0 = self._row(f, "Segmentation", "seg_mode", values=list(SEG_LABELS.values()), default=SEG_LABELS["footprint_edges"],
                                  readonly=True, width=24)
        self.v["seg_mode"].trace_add("write", lambda *_: self._seg_changed())
        self.seg_row = self._row(f, "Segment at z (µm)", "seg_h", width=8)
        self.seg_note = ttk.Label(f, foreground="#888", wraplength=360, justify="left", text=SEG_NOTES["footprint_edges"])
        self.seg_note.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")

        # ---- output
        f = self._section(left, "Output")
        self._row(f, "Folder", "output_dir", button=("Choose…", self.browse_outdir), preview=False)
        self._row(f, "Name", "name", preview=False)
        ttk.Label(f, text="Empty folder = <name>_job next to the input. Empty name = the input file name.",
                  foreground="#888", wraplength=360).grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")
        self._var("keep", False, tk.BooleanVar)
        ttk.Checkbutton(f, text="Keep intermediate files (H5 stacks, TSP) in <folder>/work", variable=self.v["keep"]).grid(
            row=f.grid_size()[1], column=0, columnspan=2, sticky="w", pady=2)
        self._var("one_interface", False, tk.BooleanVar)
        cb = ttk.Checkbutton(f, text="Global interfaceFound", variable=self.v["one_interface"])
        cb.grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w", pady=2)
        _Tip(cb, GLOBAL_INTERFACE_TIP)

        # ---- advanced
        self.adv_open = tk.BooleanVar(value=False)
        self.adv_btn = ttk.Button(left, command=self._toggle_adv)
        self.adv_btn.pack(fill="x", pady=(PAD, 0))
        self.adv = ttk.Frame(left)
        self.adv.columnconfigure(1, weight=1)
        self._row(self.adv, "Value unit (.mat/.npy)", "value_unit", values=list(hms.VALUE_UNITS), default="um", readonly=True)
        self._var("top_plus_y", True, tk.BooleanVar, preview=True)
        ttk.Checkbutton(self.adv, text="Image row 0 is the +y edge", variable=self.v["top_plus_y"]).grid(
            row=self.adv.grid_size()[1], column=0, columnspan=2, sticky="w")
        self._var("stl_merge", True, tk.BooleanVar)
        ttk.Checkbutton(self.adv, text="STL: merge equal-height pixels (fewer triangles)", variable=self.v["stl_merge"]).grid(
            row=self.adv.grid_size()[1], column=0, columnspan=2, sticky="w")
        self._toggle_adv(initial=True)

        # ---- recipe (right column)
        ttk.Label(right, text="Recipe", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(PAD, 2))
        top = ttk.Frame(right)
        top.pack(fill="x")
        self.recipe_box = ttk.Combobox(top, state="readonly", values=self.store.names(), width=20)
        self.recipe_box.pack(side="left", fill="x", expand=True)
        self.recipe_box.bind("<<ComboboxSelected>>", lambda e: self.select_recipe(self.recipe_box.get()))
        ttk.Button(top, text="Save as…", width=9, command=self.save_recipe).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Delete", width=7, command=self.delete_recipe).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Reset", width=6, command=lambda: self.select_recipe(self.recipe_box.get())).pack(side="left", padx=(4, 0))
        self.recipe_state = ttk.Label(right, text="", foreground="#888")
        self.recipe_state.pack(anchor="w", pady=(2, 0))
        f = ttk.Frame(right)
        f.pack(fill="x", pady=(2, 0))
        f.columnconfigure(1, weight=1)
        for key, label, _kind in RECIPE_FIELDS:
            choices = RECIPE_CHOICES.get(key)                  # objective / mode: pick from a drop-down, no typing
            row = self._row(f, label, "r_" + key, values=choices, readonly=choices is not None, preview=False, width=12)
            if choices:
                self.choice_boxes[key] = row[2]
            if key.startswith("block_shear"):
                _Tip(row[0], BLOCK_SHEAR_TIP)
                _Tip(row[2], BLOCK_SHEAR_TIP)
            self.v["r_" + key].trace_add("write", lambda *_, k=key: self._recipe_edited(k))

        # ---- centre: preview, run controls, log
        head = ttk.Frame(centre)
        head.pack(fill="x")
        ttk.Label(head, text="Slicer preview", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        self._var("show_preview", True, tk.BooleanVar)
        ttk.Checkbutton(head, text="Show", variable=self.v["show_preview"], command=self._toggle_preview).pack(side="left", padx=(10, 0))
        self.preview_state = ttk.Label(head, text="", foreground="#888")
        self.preview_state.pack(side="right")
        self.view = ImageView(centre)
        self.view.pack(fill="both", expand=True, pady=(2, 0))
        legend = ttk.Label(centre, foreground="#888", wraplength=520, justify="left",
                           text="red dashes = field borders · orange = overlap · yellow = field centres · cyan = writing order "
                                "(green = first, magenta = last)")
        legend.pack(anchor="w")
        bar = ttk.Frame(centre)
        bar.pack(fill="x", pady=(PAD, 0))
        self.gen_btn = ttk.Button(bar, text="Generate job files", default="active", command=self.generate)
        self.gen_btn.pack(side="left", ipady=4)
        self.stl_btn = ttk.Button(bar, text="Height map → STL…", command=self.convert_stl)
        self.stl_btn.pack(side="left", padx=(PAD, 0), ipady=4)
        self.cancel_btn = ttk.Button(bar, text="Cancel", command=self.cancel.set, state="disabled")
        self.cancel_btn.pack(side="left", padx=(PAD, 0), ipady=4)
        ttk.Label(bar, text="⌘G / Ctrl+G", foreground="#888").pack(side="right")
        self.progress = ttk.Progressbar(centre, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(PAD, 2))
        self.status = tk.StringVar(value="")
        ttk.Label(centre, textvariable=self.status, foreground="#888").pack(anchor="w")
        logf = ttk.Frame(centre)
        logf.pack(fill="x", pady=(4, 0))
        self.log = tk.Text(logf, height=4, wrap="word", state="disabled", font=("TkFixedFont", 10))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")

    def _toggle_adv(self, initial=False):
        if not initial:
            self.adv_open.set(not self.adv_open.get())
        if self.adv_open.get():
            self.adv.pack(fill="x", pady=(4, 0), after=self.adv_btn)
            if not initial:
                self.after_idle(self.left_scroll.scroll_to_end)      # show the options that just opened
        else:
            self.adv.pack_forget()
        self.adv_btn.config(text=("▾ " if self.adv_open.get() else "▸ ") + "Advanced (units, orientation, STL)")

    # ------------------------------------------------------------ source: quantizer / file
    def _state_changed(self, state):
        """The Quantizer published a design (or files for it)."""
        d = state.design
        new = d is not self._last_design
        self._last_design = d
        self.rb_quant.config(state="normal" if d is not None else "disabled")
        if new and d is not None:
            self.v["source"].set("quantizer")
        self._source_changed()

    def _from_quantizer(self):
        return self.v["source"].get() == "quantizer" and self.state.design is not None

    def _source_changed(self):
        quant = self._from_quantizer()
        d = self.state.design
        # file widgets
        for lab, box, _ in self.file_rows:
            for x in (lab, box):
                x.grid_remove() if quant else x.grid()
        if quant:
            for x in self.key_row[:2]:
                x.grid_remove()
            self.input_hint.grid_remove()
            png = d.png_path or "not exported yet (the design in memory is used)"
            stl = d.stl_path or "not exported yet"
            self.quant_info.config(text=f"{Path(d.source_path).name}\n{d.summary()}\nPNG: {png}\nSTL: {stl}")
            self.quant_info.grid()
            # the design fields mirror the Quantizer's result (read-only): the state is the single source of truth
            self._quiet = True
            self.v["pitch"].set(f"{d.pixel_size_um:g}" if d.pixel_size_um else "")
            self.v["height_max"].set(f"{d.height_max_um:.6g}" if d.height_max_um else "")
            self.v["levels"].set(str(d.levels) if d.levels else "")
            self._quiet = False
            self.design_note.config(text="Taken from the Quantizer; change them there and send again.")
        else:
            self.quant_info.grid_remove()
            self.input_hint.grid()
            self.design_note.config(text="Height max: peak height of the design (images need it; .mat / .npy: leave empty to use "
                                         "their values in the chosen unit). Levels: snap to this many equidistant levels.")
            self._input_changed()
        self._set_design_state()
        self.schedule_preview()

    def _set_design_state(self):
        """Pixel size / height / levels are editable only for a height map read from a file."""
        quant = self._from_quantizer()
        stl = not quant and Path(self.v["input"].get().strip()).suffix.lower() == ".stl"
        for _, box, _ in self.design_rows:
            for child in box.winfo_children():
                child.configure(state="disabled" if quant or stl else "normal")

    def browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("Height maps and STL", "*.png *.bmp *.mat *.npy *.stl"), ("All", "*.*")])
        if p:
            self.v["source"].set("file")
            self.v["input"].set(p)
            self._source_changed()

    def browse_outdir(self):
        p = filedialog.askdirectory()
        if p:
            self.v["output_dir"].set(p)

    def _input_changed(self):
        if self._from_quantizer():
            return
        path = Path(self.v["input"].get().strip())
        ext = path.suffix.lower()
        lab, box, w = self.key_row
        for x in (lab, box):
            x.grid() if ext == ".mat" else x.grid_remove()
        self.input_hint.config(text="STL: the solid defines pixel size, height and levels." if ext == ".stl"
                               else "Height map: .png .bmp .mat .npy, or a solid: .stl")
        if ext == ".mat" and path.is_file():
            try:
                names = pq.mat_variables(path)
                w.config(values=names)
                cur = self.v["key"].get()
                self.v["key"].set(names[0] if len(names) == 1 else cur if cur in names else "")
            except Exception as e:                         # bad file -> show, don't crash
                self.status.set(f"Cannot read the .mat: {e}")
        self._set_design_state()
        self.schedule_preview()

    def _method_changed(self):
        self.method_text.config(text=METHOD_TEXT[self.v["method"].get()])
        self._seg_update()
        self.schedule_preview()

    def _seg_changed(self):
        mode = self._segmentation_mode()
        if mode == "height_threshold" and not self.v["seg_h"].get().strip():
            self.v["seg_h"].set(self.v["base"].get().strip() or "0.2")      # just above the base: the structure's features
        self.seg_note.config(text=SEG_NOTES[mode])
        self._seg_update()
        self.schedule_preview()

    def _seg_update(self):
        """The segmentation modes belong to the adaptive method; only the threshold mode needs a height."""
        adaptive = self.v["method"].get() == "adaptive"
        for child in self.seg_row0[1].winfo_children():
            child.configure(state="readonly" if adaptive else "disabled")
        for child in self.seg_row[1].winfo_children():
            child.configure(state="normal" if adaptive and self._segmentation_mode() == "height_threshold" else "disabled")

    def _segmentation_mode(self):
        """"footprint_edges" (default), "height_threshold" or "height_gradient_guided" (the method must be adaptive)."""
        label = self.v["seg_mode"].get()
        mode = next((k for k, v in SEG_LABELS.items() if v == label), "footprint_edges")
        return mode if self.v["method"].get() == "adaptive" else "footprint_edges"

    def _segment_height(self):
        """z of the height-threshold segmentation, or None in the other modes."""
        if self._segmentation_mode() != "height_threshold":
            return None
        return self._num("seg_h", "Segmentation height")

    # ------------------------------------------------------------ the design being sliced
    def _num(self, name, label, required=True):
        t = self.v[name].get().strip()
        if not t:
            if required:
                raise ValueError(f"{label} is required")
            return None
        try:
            return float(t)
        except ValueError:
            raise ValueError(f"{label} must be a number, got {t!r}") from None

    def current_design(self) -> Design:
        """The structure to slice: the Quantizer's published design, or one built from the file. Raises ValueError."""
        if self._from_quantizer():
            return self.state.design
        inp = self.v["input"].get().strip()
        if not inp:
            raise ValueError("choose an input file")
        if not Path(inp).is_file():
            raise ValueError(f"file not found: {inp}")
        levels = self._num("levels", "Levels", False)
        if levels is not None and levels != int(levels):
            raise ValueError("Levels must be a whole number")
        d = design_from_file(inp, pixel_size_um=self._num("pitch", "Pixel size", False), height_max_um=self._num("height_max", "Height max", False),
                             levels=int(levels) if levels else None, key=self.v["key"].get().strip() or None,
                             value_unit=self.v["value_unit"].get())
        return d

    def _base(self):
        return self._num("base", "Base thickness", False) or 0.0

    # ------------------------------------------------------------ slicer preview
    def schedule_preview(self):
        if self._quiet or not hasattr(self, "view"):
            return
        if self._preview_after:
            self.after_cancel(self._preview_after)
        self._preview_after = self.after(400, self.update_preview)

    def _toggle_preview(self):
        if self.v["show_preview"].get():
            self.schedule_preview()
        else:
            self.update_preview()                               # shows the "off" message and stops pending work

    def update_preview(self):
        """Compute the layout in a worker thread: a fast reduced one first, then the full-resolution one."""
        self._preview_after = None
        self._preview_serial += 1                               # a newer request (or "off") invalidates running ones
        serial = self._preview_serial
        if not self.v["show_preview"].get():
            self.view.say("Slicer preview is off. Tick “Show” to see how the slicing method divides the structure.")
            self.preview_state.config(text="")
            return
        try:
            d = self.current_design()
            rec = self._fields_to_recipe()
            method, base, seg, segmode = self.v["method"].get(), self._base(), self._segment_height(), self._segmentation_mode()
            top_plus_y = True if self._from_quantizer() else self.v["top_plus_y"].get()
            if d.heights_um is not None and not d.pixel_size_um:
                raise ValueError("enter the pixel size to see the preview")
        except ValueError as e:
            self.view.say(f"Preview: {e}")
            self.preview_state.config(text="")
            return
        post = self.events.put
        self.preview_state.config(text="computing…")
        if self.view.img is None:
            self.view.say("Computing the slicer preview…")

        def make(side):
            if d.heights_um is None:
                return sp.stl_footprint(d.stl_path or d.source_path, max_side=side)
            return sp.design_footprint(d.heights_um, d.pixel_size_um, base, top_plus_y, max_side=side, segment_height_um=seg)

        def work():
            try:
                quick = None
                sides = (PREVIEW_QUICK, PREVIEW_FULL) if method == "adaptive" else (PREVIEW_FULL,)
                for side in sides:
                    if serial != self._preview_serial:
                        return                                  # a newer request replaced this one
                    fp = make(side)
                    if quick is not None and fp.reduced == quick:
                        break                                   # the quick pass was already at full resolution
                    lay = sp.compute_layout(fp, method, rec, segment_height_um=seg, segmentation_mode=segmode,
                                            route_solvers=rs.ROUTE_AUTO, route_time_limit=PREVIEW_ROUTE_SECONDS)   # same solver as the job, short limit
                    final = side == sides[-1] or fp.reduced == 1
                    post(("preview", (serial, sp.render(lay, rec.overlap), sp.caption(lay, rec.overlap), final)))
                    quick = fp.reduced
                    if final:
                        break
            except Exception as e:
                post(("preview_error", (serial, str(e))))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ recipes
    def _recipe_edited(self, key):
        if self._quiet:
            return
        self._update_state()
        if key in SLICING_FIELDS:
            self.schedule_preview()

    def _update_state(self):
        try:
            changed = self._fields_to_recipe().to_dict() != self._loaded.to_dict()
        except ValueError:
            changed = True
        name = self.recipe_box.get()
        base = "built-in default" if self.store.is_builtin(name) else "saved recipe"
        self.recipe_state.config(text=f"{base}{' · modified (not saved)' if changed else ''}")

    def select_recipe(self, name):
        try:
            rec = self.store.get(name)
        except KeyError:
            rec = self.store.get(DEFAULT_NAME)
            name = DEFAULT_NAME
        self.recipe_box.config(values=self.store.names())
        self.recipe_box.set(name)
        self._quiet = True
        for key, _, kind in RECIPE_FIELDS:
            val = getattr(rec, key)
            if key in self.choice_boxes:                       # a saved label that is not in the list stays selectable
                self.choice_boxes[key].config(values=list(RECIPE_CHOICES[key]) + ([val] if val not in RECIPE_CHOICES[key] else []))
            self.v["r_" + key].set(str(val) if kind is str else f"{val:g}")
        self._quiet = False
        self._loaded = rec
        self._update_state()
        self.schedule_preview()

    def _fields_to_recipe(self, name=None) -> Recipe:
        kw = {"name": name or self.recipe_box.get() or DEFAULT_NAME}
        for key, label, kind in RECIPE_FIELDS:
            t = self.v["r_" + key].get().strip()
            if kind is str:
                kw[key] = t
                continue
            try:
                kw[key] = int(t) if kind is int else float(t)
            except ValueError:
                raise ValueError(f"{label.split(' (')[0]} must be a number, got {t!r}") from None
        return Recipe(**kw).validate()

    def save_recipe(self):
        try:
            rec = self._fields_to_recipe()
        except ValueError as e:
            messagebox.showerror("Recipe", str(e))
            return
        current = self.recipe_box.get()
        suggestion = "" if self.store.is_builtin(current) else current
        name = simpledialog.askstring("Save recipe as", "Recipe name:", initialvalue=suggestion, parent=self)
        if name is None:
            return
        name = name.strip()
        rec.name = name
        try:
            overwrite = False
            if name in self.store.names() and not self.store.is_builtin(name):
                if not messagebox.askyesno("Recipe", f"Overwrite the recipe {name!r}?"):
                    return
                overwrite = True
            self.store.save(rec, overwrite=overwrite)
        except (ValueError, OSError) as e:
            messagebox.showerror("Recipe", str(e))
            return
        self.select_recipe(name)
        self.status.set(f"Saved recipe {name!r}")

    def delete_recipe(self):
        name = self.recipe_box.get()
        if self.store.is_builtin(name):
            messagebox.showinfo("Recipe", f"{DEFAULT_NAME!r} is the built-in default and cannot be deleted.")
            return
        if not messagebox.askyesno("Recipe", f"Delete the recipe {name!r}?"):
            return
        try:
            self.store.delete(name)
        except (KeyError, ValueError, OSError) as e:
            messagebox.showerror("Recipe", str(e))
            return
        self.select_recipe(DEFAULT_NAME)
        self.status.set(f"Deleted recipe {name!r}")

    # ------------------------------------------------------------ settings -> job
    def _job_settings(self) -> jg.JobSettings:
        d = self.current_design()
        name = self.v["name"].get().strip()
        out = self.v["output_dir"].get().strip() or str(Path(d.source_path).parent / f"{name or Path(d.source_path).stem}_job")
        return jg.settings_from_design(
            d, out, name=name, method=self.v["method"].get(), base_um=self._base(), key=self.v["key"].get().strip() or None,
            value_unit=self.v["value_unit"].get(), image_top_is_plus_y=True if self._from_quantizer() else self.v["top_plus_y"].get(),
            keep_intermediate=self.v["keep"].get(), segment_height_um=self._segment_height(),
            segmentation_mode=self._segmentation_mode(), global_interface_found=self.v["one_interface"].get())

    # ------------------------------------------------------------ running
    def _set_busy(self, busy):
        self.busy = busy
        for b in (self.gen_btn, self.stl_btn):
            b.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")

    def _log(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _start(self, work, start_msg, on_done=None):
        """Run work() in a thread; its events (progress / log / result) come back through self.events. on_done(result)
        runs on the UI thread when it finishes."""
        self._on_done = on_done
        self.cancel.clear()
        self._set_busy(True)
        self.progress.config(value=0)
        self.status.set(start_msg)
        self._log(start_msg)

        def target():                                       # the worker never touches Tk
            try:
                self.events.put(("done", work()))
            except jg.JobCancelled:
                self.events.put(("cancelled", None))
            except Exception as e:
                self.events.put(("error", e))

        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    stage, done, total = payload
                    self.progress.config(value=100.0 * done / max(total, 1))
                    self.status.set(f"{stage}: {done} / {total}")
                elif kind == "preview":
                    serial, img, cap, final = payload
                    if serial == self._preview_serial:
                        self.view.show(img, cap)
                        self.preview_state.config(text="" if final else "refining…")
                elif kind == "preview_error":
                    serial, msg = payload
                    if serial == self._preview_serial:
                        self.view.say(f"Preview: {msg}")
                        self.preview_state.config(text="")
                elif kind == "done":
                    self._set_busy(False)
                    self._finished(payload)
                    cb, self._on_done = self._on_done, None
                    if cb:
                        cb(payload)
                elif kind == "cancelled":
                    self._set_busy(False)
                    self.status.set("Cancelled (files already written were left in place)")
                    self._log("cancelled")
                elif kind == "error":
                    self._set_busy(False)
                    self.status.set("Failed")
                    self._log(f"ERROR: {payload}")
                    messagebox.showerror(APP_NAME, str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finished(self, rep):
        self.progress.config(value=100)
        if "fields" in rep:                                   # job files
            hours = f", about {rep['print_hours']:.2f} h of writing" if rep["print_hours"] else ""
            msg = f"Wrote {rep['fields']} fields to {rep['output_dir']}{hours}"
            if rep.get("route"):
                msg += f"; Route: {rep['route']['solver']}"
        else:                                                 # STL
            msg = f"Wrote {rep['output']}: {rep['triangles']:,} triangles, {rep['volume_mm3']:.5f} mm³"
        self.status.set(msg)
        self._log(msg)

    def generate(self):
        if self.busy:
            return
        try:
            js, rec = self._job_settings(), self._fields_to_recipe()
        except ValueError as e:
            messagebox.showerror("Job files", str(e))
            return
        put = self.events.put
        self._start(lambda: jg.generate_job_files(
            js, rec, progress=lambda s, d, t: put(("progress", (s, d, t))), log=lambda m: put(("log", m)),
            cancel=self.cancel.is_set), f"Generating job files ({js.method} slicing, recipe {rec.name!r})…")

    def convert_stl(self):
        """Height map -> STL. From the Quantizer the STL is built from the quantized heights in memory."""
        if self.busy:
            return
        try:
            d = self.current_design()
            if d.heights_um is None:
                raise ValueError("the input is already an STL")
            if not d.pixel_size_um:
                raise ValueError("enter the pixel size first")
            base = self._base()
        except ValueError as e:
            messagebox.showerror("Height map → STL", str(e))
            return
        out = filedialog.asksaveasfilename(defaultextension=".stl", initialfile=Path(d.source_path).with_suffix(".stl").name,
                                           initialdir=str(Path(d.source_path).parent), filetypes=[("STL", "*.stl")])
        if not out:
            return
        top_plus_y = True if self._from_quantizer() else self.v["top_plus_y"].get()
        merge = self.v["stl_merge"].get()
        from_quantizer = self._from_quantizer()                  # read here: the worker thread must not touch Tk

        def work():
            return hms.heights_to_stl(d.heights_um, out, pixel_size_um=d.pixel_size_um, base_um=base,
                                      image_top_is_plus_y=top_plus_y, merge=merge)

        def attach(_rep):                                        # UI thread: the Quantizer's design now has an STL
            if from_quantizer:
                self.state.files_written(d.design_id, stl_path=out)
                self._log(f"STL attached to the design from the Quantizer: {out}")

        self._start(work, "Converting the height map to STL…", on_done=attach)


PREVIEW_ROUTE_SECONDS = 10.0        # the preview tries pyconcorde for this long, then shows the 2-opt route

GLOBAL_INTERFACE_TIP = ("Global interfaceFound\n"
                        "Off (default): interfaceFound (FindInterfaceAt) runs before every field.\n"
                        "On: it runs once, at the first field, and that interface position is reused for all later fields.")


class _Tip:
    """Minimal hover tooltip."""

    def __init__(self, widget, text):
        self.widget, self.text, self.win = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _e=None):
        if self.win is not None:
            return
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{self.widget.winfo_rootx() + 20}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}")
        tk.Label(self.win, text=self.text, justify="left", background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=320).pack()

    def hide(self, _e=None):
        if self.win is not None:
            self.win.destroy()
            self.win = None


class NanoSlicerApp(tk.Tk if tk else object):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1320x880")
        self.minsize(1100, 720)
        self.state = AppState()
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.quantizer = pq.QuantizerFrame(self.tabs, on_send=self._on_send, on_export=self._on_export)
        self.jobs = JobFrame(self.tabs, state=self.state)
        self.tabs.add(self.quantizer, text="  Quantizer  ")
        self.tabs.add(self.jobs, text="  Job File Generator  ")
        self.bind("<Command-e>", lambda e: self._on_tab(self.quantizer.export))
        self.bind("<Control-e>", lambda e: self._on_tab(self.quantizer.export))
        self.bind("<Command-g>", lambda e: self._on_tab(self.jobs.generate, job=True))
        self.bind("<Control-g>", lambda e: self._on_tab(self.jobs.generate, job=True))

    def _on_send(self, payload):
        """Quantizer -> Job File Generator: publish the design and show the tab."""
        self.state.publish(Design(origin="quantizer", **payload))
        self.tabs.select(self.jobs)

    def _on_export(self, report, serial):
        files = report.get("files", [])
        png = next((f for f in files if f.endswith(".png") and not f.endswith("_continuous.png") and "_mask" not in Path(f).name), None)
        stl = next((f for f in files if f.endswith(".stl")), None)
        self.state.files_written(serial, png_path=png, stl_path=stl)

    def _on_tab(self, fn, job=False):
        """Shortcuts act on the tab that is showing."""
        if (self.tabs.select() == str(self.jobs)) == job:
            fn()


# ==========================================================================
# COMMAND LINE
# ==========================================================================
def _console_hooks():
    def progress(stage, done, total):
        print(f"\r{stage}: {done}/{total}", end="\n" if done == total else "", flush=True)
    return progress, lambda m: print(m, flush=True)


def _apply_overrides(rec: Recipe, sets):
    types = {f.name: f.type for f in dc_fields(Recipe)}
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in types or k == "name":
            raise ValueError(f"unknown recipe field {k!r}; fields: {[x for x in types if x != 'name']}")
        setattr(rec, k, v if types[k] is str else int(v) if types[k] is int else float(v))
    return rec.validate()


def _solvers(a):
    return rs.ROUTE_FAST if a.route_solver == "2-opt" else rs.ROUTE_AUTO


def cli(argv):
    ap = argparse.ArgumentParser(prog="nanoslicer.py", description=APP_NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("quantize", help="continuous phase -> PNG / STL (same arguments as quantizer.py; "
                                    "run it as: nanoslicer.py quantize INPUT PITCH WAVELENGTH ...)")

    def design_args(s):
        s.add_argument("--pixel-size", type=float)
        s.add_argument("--height-max", type=float)
        s.add_argument("--levels", type=int)
        s.add_argument("--base", type=float, default=0.2)
        s.add_argument("--mat-key")
        s.add_argument("--value-unit", default="um", choices=list(hms.VALUE_UNITS))
        s.add_argument("--row0-minus-y", action="store_true", help="image row 0 is the -y edge")
        s.add_argument("--segment-height", type=float, help="adaptive only: z (um, base included) at which the fields are "
                       "segmented (height-based slicing; also set base_slicing_distance / base_power_multiplier with --set)")
        s.add_argument("--segmentation", choices=jg.SEGMENTATIONS, help="adaptive only: how the fields are found; default "
                       "footprint_edges (height_threshold when --segment-height is given)")
        s.add_argument("--route-solver", choices=("auto", "2-opt"), default="auto",
                       help="writing order of phase_quality_guided: auto = exact via pyconcorde if installed, else 2-opt")
        s.add_argument("--route-time-limit", type=float, default=rs.DEFAULT_TIME_LIMIT, help="seconds pyconcorde may take")
    s = sub.add_parser("interface", help="existing <name>_data.gwl -> Global interfaceFound: keep only the first FindInterfaceAt call")
    s.add_argument("data_file")
    s.add_argument("-o", "--output", help="default: change the file in place")
    s.add_argument("--command", default="FindInterfaceAt", help="name of the interface command (default FindInterfaceAt)")
    s.add_argument("--remove", action="store_true", help="delete the later calls instead of commenting them out")

    s = sub.add_parser("stl", help="height map (.png .bmp .mat .npy) -> STL")
    s.add_argument("input")
    s.add_argument("pixel_um", type=float)
    s.add_argument("-o", "--output")
    s.add_argument("--height-max", type=float, help="peak height in um (required for images)")
    s.add_argument("--levels", type=int, help="snap to this many equidistant levels")
    s.add_argument("--base", type=float, default=0.2, help="base thickness in um")
    s.add_argument("--mat-key")
    s.add_argument("--value-unit", default="um", choices=list(hms.VALUE_UNITS))
    s.add_argument("--no-merge", action="store_true", help="one column per pixel (strictly watertight, larger)")
    s.add_argument("--row0-minus-y", action="store_true", help="image row 0 is the -y edge")

    s = sub.add_parser("preview", help="draw how a slicing method divides the structure into fields")
    s.add_argument("input", help="height map or STL")
    s.add_argument("output", help="image to write (.png)")
    s.add_argument("--method", default="standard", choices=jg.METHODS)
    s.add_argument("--recipe", default=DEFAULT_NAME)
    s.add_argument("--set", action="append", metavar="KEY=VALUE")
    s.add_argument("--max-side", type=int, default=PREVIEW_FULL, help="pixels per side of the layout (smaller = faster)")
    design_args(s)

    s = sub.add_parser("compare", help="the slicing modes side by side (standard, footprint-edge, threshold, height-gradient, "
                                       "phase-quality) with seam metrics")
    s.add_argument("input", help="height map (.png .bmp .mat .npy)")
    s.add_argument("output", help="image to write (.png)")
    s.add_argument("--recipe", default=DEFAULT_NAME)
    s.add_argument("--set", action="append", metavar="KEY=VALUE")
    s.add_argument("--max-side", type=int, default=700, help="pixels per side of each panel")
    s.add_argument("--labels", help="also write the block-id image of the height-gradient-guided segmentation (16-bit PNG)")
    s.add_argument("--three", action="store_true", help="only standard, threshold and height-gradient-guided (no phase-quality, no metrics)")
    s.add_argument("--metrics", help="also write the seam metrics of every mode as JSON")
    s.add_argument("--path-image", help="also write the writing order of the phase-quality-guided mode as an image")
    design_args(s)

    s = sub.add_parser("job", help="height map / STL -> Nanoscribe job files")
    s.add_argument("input")
    s.add_argument("output_dir")
    s.add_argument("--method", default="standard", choices=jg.METHODS)
    s.add_argument("--recipe", default=DEFAULT_NAME)
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a recipe field, e.g. --set laser_power=60")
    s.add_argument("--name", default="")
    s.add_argument("--keep-intermediate", action="store_true")
    s.add_argument("--global-interface-found", "--single-interface", dest="global_interface_found", action="store_true",
                   help="Global interfaceFound: run FindInterfaceAt once, at the first field, and reuse that interface "
                        "position for all later fields (default: before every field)")
    s.add_argument("--block-shear-angle", type=float, metavar="DEG", help="block faces lean by this angle from the substrate "
                   "normal (DeScribe 'Block shear angle'; 0 = vertical seams, the default; same as --set block_shear_angle=DEG)")
    s.add_argument("--block-shear-direction", type=float, metavar="DEG", help="direction in xy the blocks lean towards "
                   "(0 = +x, 90 = +y)")
    design_args(s)

    s = sub.add_parser("recipes", help="list / show / delete recipes")
    s.add_argument("action", nargs="?", default="list", choices=["list", "show", "delete"])
    s.add_argument("name", nargs="?")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "stl":
            rep = hms.height_map_to_stl(a.input, a.output, pixel_size_um=a.pixel_um, height_max_um=a.height_max,
                                        levels=a.levels, base_um=a.base, image_top_is_plus_y=not a.row0_minus_y,
                                        merge=not a.no_merge, key=a.mat_key, value_unit=a.value_unit)
            print(f"wrote {rep['output']}: {rep['triangles']:,} triangles, volume {rep['volume_mm3']:.5f} mm^3")
        elif a.cmd == "preview":
            rec = _apply_overrides(RecipeStore().get(a.recipe), a.set)
            d = design_from_file(a.input, pixel_size_um=a.pixel_size, height_max_um=a.height_max, levels=a.levels,
                                 key=a.mat_key, value_unit=a.value_unit)
            fp = (sp.stl_footprint(a.input, max_side=a.max_side) if d.heights_um is None else
                  sp.design_footprint(d.heights_um, d.pixel_size_um, a.base, not a.row0_minus_y, max_side=a.max_side,
                                      segment_height_um=a.segment_height))
            lay = sp.compute_layout(fp, a.method, rec, jg._Ctx(log=print), segment_height_um=a.segment_height,
                                    segmentation_mode=a.segmentation, route_solvers=_solvers(a), route_time_limit=a.route_time_limit)
            sp.render(lay, rec.overlap).save(a.output)
            print(f"wrote {a.output}: {sp.caption(lay, rec.overlap)}")
        elif a.cmd == "compare":
            rec = _apply_overrides(RecipeStore().get(a.recipe), a.set)
            d = design_from_file(a.input, pixel_size_um=a.pixel_size, height_max_um=a.height_max, levels=a.levels,
                                 key=a.mat_key, value_unit=a.value_unit)
            if d.heights_um is None:
                raise ValueError("the comparison needs a height map, not an STL")
            modes = None if a.three else ("standard", "footprint", "threshold", "gradient", "phase")
            img, lays = sp.comparison_image(d.heights_um, d.pixel_size_um, a.base, not a.row0_minus_y, rec, a.segment_height,
                                            a.max_side, jg._Ctx(log=print), modes=modes, route_solvers=_solvers(a),
                                            route_time_limit=a.route_time_limit)
            img.save(a.output)
            print(f"wrote {a.output}")
            if not a.three:
                met = sp.layout_metrics(lays)
                print(f"{'mode':10s} {'blocks':>6s} {'seam mm':>8s} {'cost':>6s} {'on step':>8s} {'islands':>8s} {'travel mm':>10s}  route")
                for k, m in met.items():
                    route = (m["route_solver"] + (" (optimal)" if m["route_optimal"] else "")) if m["route_solver"] else "-"
                    print(f"{k:10s} {m['blocks']:6d} {m['seam_length_um'] / 1000:8.1f} {m['seam_cost_mean']:6.3f} "
                          f"{m['seam_on_step'] * 100:7.1f}% {m['extra_pieces']:8d} {m['travel_um'] / 1000:10.1f}  {route}")
                if a.metrics:
                    import json
                    Path(a.metrics).write_text(json.dumps(met, indent=2))
                    print(f"wrote {a.metrics}")
                if a.path_image:
                    sp.render_path(lays["phase"]).save(a.path_image)
                    print(f"wrote {a.path_image}")
            if a.labels:
                gs.save_label_image(lays["gradient"].labels, a.labels)
                print(f"wrote {a.labels}")
        elif a.cmd == "job":
            rec = _apply_overrides(RecipeStore().get(a.recipe), (a.set or []) + [f"{k}={v}" for k, v in (
                ("block_shear_angle", a.block_shear_angle), ("block_shear_direction", a.block_shear_direction)) if v is not None])
            js = jg.JobSettings(input_file=a.input, output_dir=a.output_dir, name=a.name, method=a.method,
                                pixel_size_um=a.pixel_size, height_max_um=a.height_max, levels=a.levels, base_um=a.base,
                                key=a.mat_key, value_unit=a.value_unit, image_top_is_plus_y=not a.row0_minus_y,
                                keep_intermediate=a.keep_intermediate, segment_height_um=a.segment_height,
                                segmentation_mode=a.segmentation, route_solvers=_solvers(a), route_time_limit=a.route_time_limit,
                                global_interface_found=a.global_interface_found)
            progress, log = _console_hooks()
            rep = jg.generate_job_files(js, rec, progress=progress, log=log)
            print(f"wrote {rep['fields']} fields to {rep['output_dir']}" +
                  (f" (about {rep['print_hours']:.2f} h of writing)" if rep["print_hours"] else ""))
            if rep.get("route"):
                r_ = rep["route"]
                print(f"writing order: {r_['solver']}" + (" (proven optimal open path)" if r_["optimal"] else "") +
                      f", {r_['length_um'] / 1000:.1f} mm of stage travel" + (f" [{r_['detail']}]" if r_["solver"] == "2-opt" and r_["detail"] else ""))
        elif a.cmd == "interface":
            seen = jg.global_interface_file(a.data_file, a.output, a.command, comment=not a.remove)
            print(f"{a.command}: {seen} call(s) found, " + ("nothing to change" if seen < 2 else f"kept the first, {'removed' if a.remove else 'commented out'} {seen - 1}")
                  + f" -> {a.output or a.data_file}")
        elif a.cmd == "recipes":
            store = RecipeStore()
            if a.action == "list":
                print("\n".join(store.names()))
            elif not a.name:
                raise ValueError(f"'{a.action}' needs a recipe name")
            elif a.action == "show":
                for k, v in store.get(a.name).to_dict().items():
                    print(f"{k:22s} {v}")
            else:
                store.delete(a.name)
                print(f"deleted {a.name!r}")
    except (ValueError, KeyError, RuntimeError, jg.JobCancelled) as e:
        sys.exit(f"[error] {e.args[0] if isinstance(e, KeyError) and e.args else e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quantize":
        pq.cli(sys.argv[2:])                           # the quantizer's own command line, unchanged
    elif len(sys.argv) > 1:
        cli(sys.argv[1:])
    elif tk is None:
        sys.exit("[error] tkinter is not available; use the command line (see --help)")
    else:
        NanoSlicerApp().mainloop()
