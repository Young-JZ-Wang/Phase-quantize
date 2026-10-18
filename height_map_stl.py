#!/usr/bin/env python3
"""
height_map_stl.py
-----------------
Height map (.png / .bmp / .mat / .npy)  ->  closed STL solid (binary, micrometres).

Ported from TipSlicer's png_to_stl.py, same geometry and behaviour, extended to .mat and .npy inputs.

Every pixel becomes a flat-topped column, so the steps of a quantised phase mask / DOE stay sharp (no sloped flanks).
Neighbouring pixels of equal height are merged into one rectangle, which keeps the triangle count small, and walls are
only created where heights differ. The solid has a flat bottom at z = 0 and a solid base of `base_um` under the lowest
level.

Heights
    height_max_um given : the lowest value maps to 0 and the highest to height_max_um (always the case for images,
                          whose grey values are relative)
    height_max_um None  : (.mat / .npy) the values are heights, `value_unit` ("um", "nm", "mm", "m"), minimum shifted to 0
    levels              : optionally snap to this many equidistant levels first
"""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

import quantizer as pq

HEIGHT_MAP_EXTS = (".png", ".bmp", ".mat", ".npy")
VALUE_UNITS = {"um": 1.0, "nm": 1e-3, "mm": 1e3, "m": 1e6}      # -> micrometres


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_height_array(path, key=None) -> np.ndarray:
    """2-D float64 array of the raw values of a .png / .bmp / .mat / .npy file. Raises ValueError on bad input."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in pq.IMAGE_EXTS:
        with Image.open(path) as im:
            if im.mode in ("I;16", "I;16L", "I;16B", "I"):
                a = np.asarray(im, dtype=np.float64)
            else:
                a = np.asarray(im.convert("L"), dtype=np.float64)
        return a
    if ext == ".npy":
        a = np.load(path)
    elif ext == ".mat":
        d = pq._read_mat(path)
        names = [k for k, v in d.items() if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2]
        if key is None:
            if len(names) != 1:
                raise ValueError(f".mat has 2-D variables {names}; set the variable name")
            key = names[0]
        if key not in d:
            raise ValueError(f"variable {key!r} not in .mat (found {list(d)})")
        a = d[key]
    else:
        raise ValueError(f"height map must be .png, .bmp, .mat or .npy, got {ext!r}")
    a = np.asarray(a)
    if np.iscomplexobj(a):
        raise ValueError("complex data is not a height map; quantize its phase first (Quantizer tab)")
    a = np.squeeze(a).astype(np.float64)
    if a.ndim != 2:
        raise ValueError(f"height map must be 2-D, got shape {a.shape}")
    return np.nan_to_num(a)


def heights_from_array(a, height_max_um=None, levels=None, value_unit="um") -> np.ndarray:
    """Raw values -> heights in micrometres, minimum shifted to 0 (see the module docstring)."""
    if value_unit not in VALUE_UNITS:
        raise ValueError(f"value unit must be one of {list(VALUE_UNITS)}, got {value_unit!r}")
    if levels is not None and levels < 2:
        raise ValueError("levels must be >= 2")
    a = np.asarray(a, dtype=np.float64)
    g = a - a.min()
    if g.max() == 0:
        raise ValueError("the height map is flat, nothing to build")
    peak_um = float(g.max()) * VALUE_UNITS[value_unit]                    # the values' own peak height
    if levels is not None:
        g = np.round(g / g.max() * (levels - 1)) / (levels - 1)            # 0 ... 1 in `levels` steps
        return g * (height_max_um if height_max_um is not None else peak_um)
    if height_max_um is not None:
        return g / g.max() * height_max_um
    return g * VALUE_UNITS[value_unit]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def runs(a, merge=True):
    """Start indices and lengths of the runs of equal consecutive values in the 1D array a (single pixels if not merge)."""
    if not merge:
        return np.arange(len(a)), np.ones(len(a), dtype=int)
    change = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(a)]))
    return starts, ends - starts


def rects_to_triangles(corners, outward):
    """corners: (n, 4, 3) quads in order around the rectangle; outward: (3,) or (n, 3) desired normal.
    Returns (2n, 3, 3) triangles whose normals point along 'outward'."""
    tri = np.concatenate((corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]))
    out = np.asarray(outward, dtype=np.float64)
    out = np.tile(out, (2, 1)) if out.ndim == 2 else out
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = np.einsum("ij,ij->i", n, np.broadcast_to(out, n.shape)) < 0
    tri[flip] = tri[flip][:, ::-1]
    return tri


def build_triangles(top, pixel_size, image_top_is_plus_y=True, merge=True):
    """top: (rows, cols) height of the upper surface of every pixel column (um). Returns (T, 3, 3) triangles."""
    rows, cols = top.shape
    p = float(pixel_size)
    xe = (np.arange(cols + 1) - cols / 2.0) * p                      # pixel edges in x
    ye = ((rows / 2.0 - np.arange(rows + 1)) if image_top_is_plus_y else (np.arange(rows + 1) - rows / 2.0)) * p
    ylo = np.minimum(ye[:-1], ye[1:])                                 # y of the lower / upper edge of image row r
    yhi = np.maximum(ye[:-1], ye[1:])
    T = []

    def quad_x(x0, x1, yl, yh, z):        # horizontal rectangle (top face), arrays over runs
        return np.stack((np.stack((x0, yl, z), -1), np.stack((x1, yl, z), -1),
                         np.stack((x1, yh, z), -1), np.stack((x0, yh, z), -1)), axis=1)

    # top faces: runs of equal height along x, one rectangle per run and row
    for r in range(rows):
        s, ln = runs(top[r], merge)
        z = top[r, s]
        T.append(rects_to_triangles(quad_x(xe[s], xe[s + ln], np.full(len(s), ylo[r]), np.full(len(s), yhi[r]), z), (0, 0, 1)))

        # vertical walls inside the row (between neighbouring runs) and at the left / right end
        zl = np.concatenate(([0.0], z))          # height on the left of boundary k
        zr = np.concatenate((z, [0.0]))          # height on the right of boundary k
        xb = np.concatenate((xe[s], [xe[-1]]))   # boundary x positions
        keep = zl != zr
        xs, lo, hi = xb[keep], np.minimum(zl, zr)[keep], np.maximum(zl, zr)[keep]
        direction = np.where(zl > zr, 1, -1)[keep]
        if len(xs):
            c = np.stack((np.stack((xs, np.full(len(xs), ylo[r]), lo), -1), np.stack((xs, np.full(len(xs), yhi[r]), lo), -1),
                          np.stack((xs, np.full(len(xs), yhi[r]), hi), -1), np.stack((xs, np.full(len(xs), ylo[r]), hi), -1)), axis=1)
            out = np.zeros((len(xs), 3))
            out[:, 0] = direction                                   # wall faces toward the lower side
            T.append(rects_to_triangles(c, out))

    # horizontal walls between image rows r and r+1 (and the two outer long sides), runs of equal (upper, lower) pair
    for r in range(rows + 1):
        a = top[r - 1] if r > 0 else np.zeros(cols)        # row above the boundary r (in image order)
        b = top[r] if r < rows else np.zeros(cols)         # row below
        s, ln = runs(a * 1e6 + b, merge)
        keep = a[s] != b[s]
        if not keep.any():
            continue
        s, ln = s[keep], ln[keep]
        y = np.full(len(s), ye[r])
        lo = np.minimum(a[s], b[s])
        hi = np.maximum(a[s], b[s])
        c = np.stack((np.stack((xe[s], y, lo), -1), np.stack((xe[s + ln], y, lo), -1),
                      np.stack((xe[s + ln], y, hi), -1), np.stack((xe[s], y, hi), -1)), axis=1)
        step = -1.0 if image_top_is_plus_y else 1.0         # y direction of increasing row index
        out = np.zeros((len(s), 3))
        out[:, 1] = np.where(a[s] > b[s], step, -step)
        T.append(rects_to_triangles(c, out))

    # flat bottom
    x0, x1, y0, y1 = xe[0], xe[-1], min(ye[0], ye[-1]), max(ye[0], ye[-1])
    bottom = np.array([[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]], dtype=np.float64)
    T.append(rects_to_triangles(bottom, (0, 0, -1)))
    return np.concatenate(T)


def write_binary_stl(path, tri, header="height_map_stl"):
    """Binary STL: 80 byte header, uint32 count, then per triangle normal(3f) + 3 vertices(9f) + uint16."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(header.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())


def volume(tri) -> float:
    """Signed volume (um^3) from the triangles; positive for a correctly oriented closed solid."""
    return float(np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0)


# --------------------------------------------------------------------------
# Reading an STL (for the slicer preview; no extra dependency)
# --------------------------------------------------------------------------
def read_stl_triangles(path) -> np.ndarray:
    """(T, 3, 3) float64 triangle vertices of a binary or ASCII STL."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = struct.unpack("<I", raw[80:84])[0]
        if len(raw) == 84 + 50 * n:                                        # binary
            rec = np.frombuffer(raw, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n, offset=84)
            return rec["v"].astype(np.float64)
    try:                                                                     # ASCII
        pts = [[float(t) for t in line.split()[1:4]] for line in raw.decode("ascii", "ignore").splitlines()
               if line.strip().startswith("vertex")]
    except ValueError as e:
        raise ValueError(f"cannot read {path} as an STL ({e})") from None
    if not pts or len(pts) % 3:
        raise ValueError(f"{path} is not a valid STL")
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3, 3)


# --------------------------------------------------------------------------
# One call: file -> STL
# --------------------------------------------------------------------------
def default_stl_path(input_file) -> Path:
    return Path(input_file).with_suffix(".stl")


def heights_to_stl(heights_um, output_file, *, pixel_size_um, base_um=0.2, image_top_is_plus_y=True, merge=True) -> dict:
    """Build the solid for heights already in micrometres (e.g. a quantized design), write the binary STL and return a
    report dict. Raises ValueError on bad settings."""
    if not pixel_size_um or pixel_size_um <= 0:
        raise ValueError("pixel size must be > 0")
    if base_um is None or base_um < 0:
        raise ValueError("base thickness must be >= 0")
    top = np.asarray(heights_um, dtype=np.float64) + base_um
    if top.ndim != 2:
        raise ValueError("heights must be a 2-D array")
    if top.min() <= 0:
        raise ValueError("the lowest column has zero height: set a base thickness > 0 so the solid has no "
                         "zero-thickness regions")
    tri = build_triangles(top, pixel_size_um, image_top_is_plus_y, merge)
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_binary_stl(out, tri)
    rows, cols = top.shape
    return {"output": str(out), "triangles": int(len(tri)), "shape": [rows, cols], "pixel_size_um": pixel_size_um,
            "footprint_um": [cols * pixel_size_um, rows * pixel_size_um], "height_um": float(top.max()),
            "base_um": base_um, "volume_mm3": volume(tri) * 1e-9, "merge": bool(merge)}


def height_map_to_stl(input_file, output_file=None, *, pixel_size_um, height_max_um=None, levels=None,
                      base_um=0.2, image_top_is_plus_y=True, merge=True, key=None, value_unit="um") -> dict:
    """Load a .png / .bmp / .mat / .npy height map, build the solid, write the binary STL and return a report dict.
    Raises ValueError on bad input or settings."""
    input_file = Path(input_file)
    if not pixel_size_um or pixel_size_um <= 0:
        raise ValueError("pixel size must be > 0")
    a = load_height_array(input_file, key)
    if input_file.suffix.lower() in pq.IMAGE_EXTS and height_max_um is None:
        raise ValueError("height max (um) is required for images: grey values are relative")
    if height_max_um is not None and height_max_um <= 0:
        raise ValueError("height max must be > 0")
    heights = heights_from_array(a, height_max_um, levels, value_unit)
    return heights_to_stl(heights, output_file or default_stl_path(input_file), pixel_size_um=pixel_size_um,
                          base_um=base_um, image_top_is_plus_y=image_top_is_plus_y, merge=merge)
