#!/usr/bin/env python3
"""
state.py
--------
The data NanoSlicer's tabs share, in one place.

    Quantizer tab ──publishes──▶  AppState.design  ◀──reads── Job File Generator tab

`Design` is a height map that is ready for slicing: heights in micrometres plus everything that describes it
(pixel size, levels, height per level, physical size, where it came from, the PNG / STL written for it). The
Quantizer builds one from its quantized result and publishes it; the Job File Generator can also build one from a
file of its own. Listeners are told whenever the published design (or the files written for it) changes, so the
tabs never copy values into each other's fields.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class Design:
    source_path: str                              # input file the design came from (name for the output files)
    origin: str                                   # "quantizer" or "file"
    heights_um: Optional[np.ndarray] = None       # (rows, cols), row 0 = +y edge; None for a pure STL input
    pixel_size_um: Optional[float] = None
    levels: Optional[int] = None                  # number of equidistant height levels (None = continuous)
    step_um: Optional[float] = None               # height per level ("height scale")
    height_max_um: Optional[float] = None
    png_path: Optional[str] = None                # 8-bit level map written for this design, if any
    stl_path: Optional[str] = None                # STL written for this design, if any
    design_id: Optional[int] = None               # identifies the quantizer result this came from

    @property
    def shape(self):
        return None if self.heights_um is None else tuple(self.heights_um.shape)

    @property
    def footprint_um(self):
        """(width, height) of the structure in micrometres, or None when the pixel size or the map is missing."""
        if self.heights_um is None or not self.pixel_size_um:
            return None
        rows, cols = self.heights_um.shape
        return (cols * self.pixel_size_um, rows * self.pixel_size_um)

    def summary(self) -> str:
        parts = [f"{self.shape[1]} × {self.shape[0]} px" if self.shape else "STL"]
        if self.pixel_size_um:
            parts.append(f"{self.pixel_size_um:g} µm/px")
        if self.footprint_um:
            parts.append(f"{self.footprint_um[0]:g} × {self.footprint_um[1]:g} µm")
        if self.levels:
            parts.append(f"{self.levels} levels" + (f" of {self.step_um * 1000:.1f} nm" if self.step_um else ""))
        if self.height_max_um:
            parts.append(f"height {self.height_max_um:g} µm")
        return "  ·  ".join(parts)


def design_from_file(path, *, pixel_size_um=None, height_max_um=None, levels=None, key=None, value_unit="um") -> Design:
    """A Design from a file of its own: a height map (.png .bmp .mat .npy) or an STL. Raises ValueError on bad input."""
    import height_map_stl as hms
    p = str(path)
    if p.lower().endswith(".stl"):
        return Design(source_path=p, origin="file", stl_path=p)
    a = hms.load_height_array(p, key)
    if p.lower().endswith(tuple(hms.pq.IMAGE_EXTS)) and not height_max_um:
        raise ValueError("height max (µm) is required for images: grey values are relative")
    h = hms.heights_from_array(a, height_max_um or None, levels or None, value_unit)
    step = float(h.max()) / (levels - 1) if levels else None
    return Design(source_path=p, origin="file", heights_um=h, pixel_size_um=pixel_size_um, levels=levels or None,
                  step_um=step, height_max_um=float(h.max()), png_path=p if p.lower().endswith(".png") else None)


class AppState:
    """What the tabs share. `design` is the one published by the Quantizer."""

    def __init__(self):
        self.design: Optional[Design] = None
        self._listeners: list[Callable[["AppState"], None]] = []

    def subscribe(self, fn):
        self._listeners.append(fn)

    def _notify(self):
        for fn in list(self._listeners):
            fn(self)

    def publish(self, design: Design):
        """The Quantizer hands over a new design."""
        self.design = design
        self._notify()

    def files_written(self, design_id, png_path=None, stl_path=None):
        """The Quantizer exported files; attach them to the published design if it is still the same result."""
        d = self.design
        if d is None or d.origin != "quantizer" or d.design_id != design_id:
            return
        d.png_path = png_path or d.png_path
        d.stl_path = stl_path or d.stl_path
        self._notify()
