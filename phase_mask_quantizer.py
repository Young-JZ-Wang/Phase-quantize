#!/usr/bin/env python3
"""
phase_mask_quantizer.py
-----------------
Continuous phase profile (.mat / .npy / .png / .bmp)  ->  discrete height steps  ->  8-bit PNG and/or STL.

Launch
    python phase_mask_quantizer.py                                    # GUI (or double-click the launcher)
    python phase_mask_quantizer.py INPUT PITCH_UM WAVELENGTH_NM [png|stl|both] [--step-nm DH | --levels N]

Pipeline
    1. Load the phase, optionally multiply it (PhlatCam MATLAB doubles it), wrap to [0, 2*pi)
    2. Height for a full 2*pi:  D = lambda / (n_material - n_medium)
    3. Quantize, either
         step mode   (PhlatCam):  hq = dh * floor(h / dh),  h = phase / (2*pi) * D
         levels mode:             N equal steps, dh = D / N, nearest level
    4. Export an 8-bit PNG (quantized height / its max, optional litho masks) and/or
       a watertight stepped STL (flat terraces, vertical walls, base slab), plus report.json

Output name
    phHeight_<input>_<maxH>umMaxH_<pitch>um_<size>um_q<step>nm_lam<nm>
    e.g. phHeight_perlin12_20_example_1.00umMaxH_1.00um_1886um_q200nm_lam532
    maxH = max quantized height,  size = max(Nx, Ny) * pitch
"""
import argparse
import json
import queue
import struct
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

try:                                                   # the GUI is optional (CLI / library work without tkinter)
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import ImageTk
except ImportError:
    tk = ttk = filedialog = messagebox = ImageTk = None


TWO_PI = 2.0 * np.pi
_UNIT = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}


# --------------------------------------------------------------------------
# Material refractive index n(lambda)
# --------------------------------------------------------------------------
def n_fused_silica(lam_m: float) -> float:
    """Malitson (1965) Sellmeier, valid ~0.21-3.7 um."""
    l2 = (lam_m * 1e6) ** 2
    B = (0.6961663, 0.4079426, 0.8974794)
    C = (0.0684043 ** 2, 0.1162414 ** 2, 9.896161 ** 2)
    return float(np.sqrt(1 + sum(b * l2 / (l2 - c) for b, c in zip(B, C))))


def n_su8(lam_m: float) -> float:
    """SU-8 Cauchy fit (MicroChem datasheet): A + B/l^2 + C/l^4, l in um."""
    l = lam_m * 1e6
    return 1.566 + 0.00796 / l ** 2 + 0.00014 / l ** 4


MATERIALS = {
    "air": lambda lam: 1.0,
    "fused_silica": n_fused_silica,
    "su8": n_su8,
    "pdms": lambda lam: 1.41,        # approx., weakly dispersive in the visible
    "ip_dip": lambda lam: 1.52,      # Nanoscribe IP-Dip, approx. (polymerized)
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_mat(path: Path) -> dict:
    """All variables of a .mat file (incl. MATLAB v7.3/HDF5) as numpy arrays."""
    try:
        from scipy.io import loadmat
        return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:                        # v7.3 .mat files are HDF5
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f if isinstance(f[k], h5py.Dataset)}


def mat_variables(path: Path) -> list[str]:
    """Names of the 2-D variables in a .mat file."""
    return [k for k, v in _read_mat(path).items()
            if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2]


def apply_unit(phi: np.ndarray, unit: str) -> np.ndarray:
    """Convert a phase map from `unit` ("rad", "waves" or "deg") to radians."""
    if unit == "waves":
        return phi * TWO_PI
    if unit == "deg":
        return np.deg2rad(phi)
    return phi


IMAGE_EXTS = (".png", ".bmp")


def _load_image(path: Path) -> np.ndarray:
    """Grayscale image -> phase in radians: black = 0, white = 2*pi (8- or 16-bit)."""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16L", "I;16B", "I"):
            arr, top = np.asarray(im, dtype=np.float64), 65535.0
        else:
            arr, top = np.asarray(im.convert("L"), dtype=np.float64), 255.0
    return arr / top * TWO_PI


def load_phase(path: Path, mat_key: str | None = None, unit: str = "rad") -> np.ndarray:
    """Load a continuous phase map from .npy, .mat or an image (.png / .bmp). Raises ValueError on bad input."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        phi = np.load(path)
    elif ext in IMAGE_EXTS:
        return _load_image(path)
    elif ext == ".mat":
        d = _read_mat(path)
        arrays = [k for k, v in d.items() if isinstance(v, np.ndarray) and np.squeeze(v).ndim == 2]
        if mat_key is None:
            if len(arrays) != 1:
                raise ValueError(f".mat has 2-D variables {arrays}; set the variable name")
            mat_key = arrays[0]
        if mat_key not in d:
            raise ValueError(f"variable {mat_key!r} not in .mat (found {list(d)})")
        phi = d[mat_key]
    else:
        raise ValueError(f"input must be .mat, .npy, .png or .bmp, got {ext!r}")

    phi = np.asarray(phi)
    if np.iscomplexobj(phi):                           # complex field -> its phase
        phi = np.angle(phi)
    phi = np.squeeze(phi).astype(np.float64)
    if phi.ndim != 2:
        raise ValueError(f"phase must be 2-D, got shape {phi.shape}")
    return apply_unit(phi, unit)


# --------------------------------------------------------------------------
# Core: quantization
# --------------------------------------------------------------------------
def quantize(phi: np.ndarray, levels: int) -> np.ndarray:
    """Wrap to [0, 2pi) and map to integer level index 0..levels-1."""
    wrapped = np.mod(phi, TWO_PI)
    return (np.rint(wrapped / (TWO_PI / levels)).astype(np.int64) % levels).astype(np.uint16)


# --------------------------------------------------------------------------
# PNG export
# --------------------------------------------------------------------------
def save_pngs(q: np.ndarray, levels: int, heights_m: np.ndarray, out: Path,
              litho_masks: bool, depth_step_m: float) -> list[Path]:
    """
    Main image (as in PhlatCam's imwrite(phMmHq / max(phMmHq))): the quantized height map divided by
    its own maximum, saved as 8-bit grayscale. Only the discrete levels appear in it.
    """
    if levels > 256:
        raise ValueError(f"8-bit PNG holds at most 256 levels (got {levels}); use a larger step height")
    written = []

    hmax = heights_m.max()
    norm = heights_m / hmax if hmax > 0 else np.zeros_like(heights_m)
    Image.fromarray(np.round(norm * 255).astype(np.uint8), mode="L").save(p := out.with_name(out.name + ".png"))
    written.append(p)

    # Binary masks for multi-step lithography: mask b etches depth 2^b * step
    if litho_masks:
        nbits = int(np.log2(levels))
        if 2 ** nbits != levels:
            print(f"[warn] litho masks need levels = 2^m (got {levels}); skipped")
        else:
            for b in range(nbits):
                mask = ((q >> b) & 1).astype(np.uint8) * 255
                d_nm = (2 ** b) * depth_step_m * 1e9
                p = out.with_name(f"{out.name}_mask{b + 1}_depth{d_nm:.1f}nm.png")
                Image.fromarray(mask, mode="L").save(p)
                written.append(p)
    return written


# --------------------------------------------------------------------------
# STL export: watertight stepped height field
# --------------------------------------------------------------------------
def _quads_to_tris(v0, v1, v2, v3):
    """Quads (counter-clockwise seen from outside) -> two triangles each."""
    return np.concatenate([np.stack([v0, v1, v2], 1), np.stack([v0, v2, v3], 1)], 0)


def _expand_steps(lo, hi):
    """For integer ranges [lo, hi), return (index, k) so each unit step k..k+1 is its own segment."""
    n = (hi - lo).astype(np.int64)
    idx = np.repeat(np.arange(len(n)), n)
    first = np.repeat(np.cumsum(n) - n, n)
    k = lo[idx] + (np.arange(n.sum()) - first)
    return idx, k


def heightfield_to_triangles(q: np.ndarray, step: float, pitch: float, base: float) -> np.ndarray:
    """
    q     : (ny, nx) integer level map (terrace height = q * step)
    step  : height per level
    pitch : pixel size
    base  : base slab thickness (bottom at z = -base)
    Returns (T, 3, 3) float32 triangle vertices.

    Every pixel is a flat terrace; vertical walls are added wherever neighbouring
    pixels differ and around the perimeter. Walls are split at every level
    boundary so all edges are shared exactly by matching triangles
    (watertight, no T-junctions) -- slicers / 2PP software accept it directly.
    """
    q = q.astype(np.int64)
    ny, nx = q.shape
    zb = -base
    Z = lambda k: np.asarray(k, dtype=np.float64) * step
    tris = []

    def P(x, y, zz):
        return np.stack(np.broadcast_arrays(x, y, zz), -1).reshape(-1, 3).astype(np.float64)

    def X(j):  # column edge coordinate
        return np.asarray(j) * pitch

    def Y(i):  # row edge coordinate; image row 0 at the top (+y), matching the PNG
        return (ny - np.asarray(i)) * pitch

    ii, jj = np.mgrid[0:ny, 0:nx]
    z = Z(q)
    # top terraces (+z) and bottom (-z)
    tris.append(_quads_to_tris(P(X(jj), Y(ii + 1), z), P(X(jj + 1), Y(ii + 1), z),
                               P(X(jj + 1), Y(ii), z), P(X(jj), Y(ii), z)))
    tris.append(_quads_to_tris(P(X(jj), Y(ii + 1), zb), P(X(jj), Y(ii), zb),
                               P(X(jj + 1), Y(ii), zb), P(X(jj + 1), Y(ii + 1), zb)))

    def xwall(x, ya, yb, za, zb_, face_pos):
        """wall in plane x=const, spanning y in [ya,yb] (ya<yb), z in [za,zb_]."""
        A, B, C, D = P(x, ya, za), P(x, yb, za), P(x, yb, zb_), P(x, ya, zb_)
        fp = np.broadcast_to(face_pos, (len(A),))
        return [_quads_to_tris(A[fp], B[fp], C[fp], D[fp]),
                _quads_to_tris(A[~fp], D[~fp], C[~fp], B[~fp])]

    def ywall(y, xa, xb, za, zb_, face_pos):
        """wall in plane y=const, spanning x in [xa,xb], faces +y if face_pos."""
        A, B, C, D = P(xa, y, za), P(xb, y, za), P(xb, y, zb_), P(xa, y, zb_)
        fp = np.broadcast_to(face_pos, (len(A),))
        return [_quads_to_tris(A[fp], D[fp], C[fp], B[fp]),
                _quads_to_tris(A[~fp], B[~fp], C[~fp], D[~fp])]

    # internal walls between column j and j+1
    ql, qr = q[:, :-1].ravel(), q[:, 1:].ravel()
    ri, rj = ii[:, :-1].ravel(), jj[:, :-1].ravel()
    m = ql != qr
    idx, k = _expand_steps(np.minimum(ql, qr)[m], np.maximum(ql, qr)[m])
    sel = np.flatnonzero(m)[idx]
    tris += xwall(X(rj[sel] + 1), Y(ri[sel] + 1), Y(ri[sel]), Z(k), Z(k + 1), ql[sel] > qr[sel])

    # internal walls between row i and i+1 (row i+1 is lower in y)
    qu, qd = q[:-1, :].ravel(), q[1:, :].ravel()
    ri, rj = ii[:-1, :].ravel(), jj[:-1, :].ravel()
    m = qu != qd
    idx, k = _expand_steps(np.minimum(qu, qd)[m], np.maximum(qu, qd)[m])
    sel = np.flatnonzero(m)[idx]
    tris += ywall(Y(ri[sel] + 1), X(rj[sel]), X(rj[sel] + 1), Z(k), Z(k + 1), ~(qu[sel] > qd[sel]))

    # perimeter walls: base slab segment [zb, 0] + one segment per level up to the terrace
    def perim(levels_edge):
        n = len(levels_edge)
        idx, k = _expand_steps(np.zeros(n, np.int64), levels_edge)
        seg_i = np.concatenate([np.arange(n), idx])
        z0 = np.concatenate([np.full(n, zb), Z(k)])
        z1 = np.concatenate([np.zeros(n), Z(k + 1)])
        return seg_i, z0, z1

    j = np.arange(nx); i = np.arange(ny)
    s, z0, z1 = perim(q[-1, :]); tris += ywall(Y(ny), X(j[s]), X(j[s] + 1), z0, z1, False)   # front, -y
    s, z0, z1 = perim(q[0, :]);  tris += ywall(Y(0), X(j[s]), X(j[s] + 1), z0, z1, True)     # back, +y
    s, z0, z1 = perim(q[:, 0]);  tris += xwall(X(0), Y(i[s] + 1), Y(i[s]), z0, z1, False)    # left, -x
    s, z0, z1 = perim(q[:, -1]); tris += xwall(X(nx), Y(i[s] + 1), Y(i[s]), z0, z1, True)    # right, +x

    return np.concatenate([t for t in tris if len(t)], 0).astype(np.float32)


def write_binary_stl(path: Path, tris: np.ndarray, header: str = "phase_mask_quantizer") -> None:
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, tris
    with open(path, "wb") as f:
        f.write(header.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        rec.tofile(f)


# --------------------------------------------------------------------------
# Output naming
# --------------------------------------------------------------------------
def design_name(stem, max_h, pitch, shape, step, lam) -> str:
    """
    phHeight_<input>_<maxH>umMaxH_<pitch>um_<size>um_q<step>nm_lam<lambda>
    e.g. phHeight_perlin12_20_example_1.00umMaxH_1.00um_1886um_q200nm_lam532
    The pitch and size fields are left out when no pitch is given.
    """
    parts = ["phHeight", stem, f"{max_h * 1e6:.2f}umMaxH"]
    if pitch is not None:
        parts.append(f"{pitch * 1e6:.2f}um")
    if pitch is not None:
        parts.append(f"{max(shape) * pitch * 1e6:.0f}um")          # physical mask size
    parts.append(f"q{step * 1e9:.0f}nm")
    parts.append(f"lam{lam * 1e9:.0f}")
    return "_".join(parts)


# ==========================================================================
# DESIGN + EXPORT
# ==========================================================================
RESAMPLE_MODES = ("original", "up", "down")
MAX_RESAMPLE_FACTOR = 64


def resampled_shape(shape, mode, factor):
    """Pixel grid after resampling: 'up' multiplies each axis by `factor`, 'down' divides it."""
    if mode not in RESAMPLE_MODES:
        raise ValueError(f"resample mode must be one of {RESAMPLE_MODES}, got {mode!r}")
    if mode == "original":
        return tuple(shape)
    if not 1.0 <= factor <= MAX_RESAMPLE_FACTOR:
        raise ValueError(f"resample factor must be between 1 and {MAX_RESAMPLE_FACTOR}, got {factor}")
    new = tuple(max(1, int(round(n * factor if mode == "up" else n / factor))) for n in shape)
    if min(new) < 2:
        raise ValueError(f"down-sampling {tuple(shape)} by {factor}x leaves fewer than 2 pixels")
    return new


def resample_phase(phi: np.ndarray, mode: str = "original", factor: float = 1.0) -> np.ndarray:
    """
    Resample a continuous phase map onto a finer ('up') or coarser ('down') pixel grid, keeping the
    physical footprint. The unit phasor exp(i*phi) is interpolated (bilinear, anti-aliased when
    shrinking) and its angle taken, so phase wraps are never averaged across. 'original' returns
    `phi` untouched.
    """
    new = resampled_shape(phi.shape, mode, factor)
    if new == phi.shape:
        return phi
    size = (new[1], new[0])
    re = np.asarray(Image.fromarray(np.cos(phi).astype(np.float32)).resize(size, Image.BILINEAR), np.float64)
    im = np.asarray(Image.fromarray(np.sin(phi).astype(np.float32)).resize(size, Image.BILINEAR), np.float64)
    return np.arctan2(im, re)


def to_si(wavelength_nm, pixel_pitch_um):
    """Convert user units to SI; pitch becomes None when not given."""
    lam = wavelength_nm * 1e-9
    pitch = pixel_pitch_um * 1e-6 if pixel_pitch_um else None
    return lam, pitch


def build_design(phi, *, wavelength_nm, levels=6, material="fused_silica",
                 n_index=None, n_medium=1.0, invert=False, phase_scale=1.0,
                 substrate="fused_silica", n_substrate=None, step_nm=None, rounding="floor",
                 resample="original", factor=1.0) -> dict:
    """Quantize a phase map (radians) and derive heights. Raises ValueError on bad settings.

    resample="up"/"down" with `factor` first moves the phase onto a finer / coarser pixel grid
    (see resample_phase); "original" keeps the input grid. The input `phi` is kept as d["phi_raw"].

    Two ways to set the staircase:
      step_nm=None : `levels` equal steps over one 2*pi depth (step = depth / levels), nearest level.
      step_nm=dh   : fixed height step dh (PhlatCam style): h = wrapped_phase * lambda / (2*pi*dn),
                     hq = dh * floor(h / dh)  (rounding="floor") or dh * round(h / dh)  ("round").
                     The number of levels follows from the 2*pi depth.
    """
    if step_nm is None and levels < 2:
        raise ValueError("levels must be >= 2")
    if rounding not in ("floor", "round"):
        raise ValueError("rounding must be 'floor' or 'round'")
    lam = wavelength_nm * 1e-9
    n = n_index if n_index is not None else MATERIALS[material](lam)              # structure
    n_sub = n_substrate if n_substrate is not None else MATERIALS[substrate](lam)  # substrate (reported only)
    dn = n - n_medium
    if dn <= 0:
        raise ValueError(f"n - n_medium must be > 0 (got {dn:.4f})")

    phi_raw = phi
    phi = phi * phase_scale                            # PhlatCam MATLAB uses 2 (mod(2*phMm, 2*pi))
    if invert:
        phi = -phi
    source_shape = phi.shape
    phi = resample_phase(phi, resample, factor)
    depth_2pi = lam / dn                               # height for a full 2*pi
    if step_nm is None:
        q = quantize(phi, levels)
        step = depth_2pi / levels                      # height per level
    else:
        step = step_nm * 1e-9
        if step <= 0 or step >= depth_2pi:
            raise ValueError(f"step height must be in (0, {depth_2pi * 1e9:.0f}) nm (the 2*pi depth)")
        levels = int(np.ceil(depth_2pi / step - 1e-9))
        h = np.mod(phi, TWO_PI) / TWO_PI * depth_2pi   # continuous height in [0, depth)
        k = np.floor(h / step) if rounding == "floor" else np.rint(h / step)
        q = np.clip(k, 0, levels - 1).astype(np.uint16)
    err = np.angle(np.exp(1j * (np.mod(phi, TWO_PI) - q * TWO_PI * step / depth_2pi)))
    return {"q": q, "heights": q.astype(np.float64) * step, "err": err, "n": n, "n_substrate": n_sub,
            "depth_2pi": depth_2pi, "step": step, "levels": levels, "lam": lam,
            "efficiency": float(np.sinc(step / depth_2pi) ** 2),   # = sinc^2(1/N) for N equal levels
            "rms_err": float(np.sqrt(np.mean(err ** 2))),
            "phi_raw": phi_raw,                        # continuous input, before any sampling / quantization
            "resample": {"mode": resample, "factor": float(factor) if resample != "original" else 1.0,
                         "source_shape": list(source_shape)},
            "pitch_scale": max(source_shape) / max(phi.shape)}     # new pixel size = pitch * pitch_scale


def output_path(d, in_path, pixel_pitch_um, output_dir=None, name_override=None) -> Path:
    """Base path (no extension) of all files written for design `d`."""
    _, pitch = to_si(d["lam"] * 1e9, pixel_pitch_um)
    if pitch is not None:
        pitch *= d["pitch_scale"]                      # resampled grid -> new pixel size, same footprint
    out_dir = Path(output_dir) if output_dir else Path(in_path).parent
    name = name_override or design_name(Path(in_path).stem, d["step"] * (d["levels"] - 1),
                                        pitch, d["q"].shape, d["step"], d["lam"])
    rs = d["resample"]
    if not name_override and rs["mode"] != "original":
        name += f"_{rs['mode']}{rs['factor']:g}x"
    return out_dir / name


def export_design(d, out: Path, in_path, *, pixel_pitch_um, output="both",
                  material="fused_silica", n_index=None, n_medium=1.0,
                  substrate="fused_silica", n_substrate=None, litho_masks=False,
                  save_npy=False, stl_base_um=10.0, stl_units="um", stl_z_scale=1.0,
                  export_continuous=True) -> dict:
    """
    Write the files for design `d` and return the report dict. Two stages, in order:
      1. original continuous phase (before any sampling / discretization): `<name>_continuous.npy`
         (float64 radians, exact) and `<name>_continuous.png` (16-bit, black = 0, white = 2*pi)
      2. discrete design on the (possibly resampled) pixel grid: 8-bit PNG, STL, optional extras,
         and `<name>_report.json`
    """
    if output not in ("png", "stl", "both"):
        raise ValueError("output must be 'png', 'stl' or 'both'")
    _, pitch = to_si(d["lam"] * 1e9, pixel_pitch_um)
    if pitch is not None:
        pitch *= d["pitch_scale"]                      # pixel size of the (resampled) grid
    if output in ("stl", "both") and pitch is None:
        raise ValueError("pixel pitch is required for STL output")

    q, levels, step = d["q"], d["levels"], d["step"]
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(q.ravel(), minlength=levels)
    report = {
        "input": str(in_path), "shape": list(q.shape),
        "wavelength_nm": d["lam"] * 1e9, "levels": levels,
        "material": "custom" if n_index is not None else material,
        "n": d["n"], "n_medium": n_medium,
        "substrate": "custom" if n_substrate is not None else substrate, "n_substrate": d["n_substrate"],
        "depth_2pi_nm": d["depth_2pi"] * 1e9, "step_height_nm": step * 1e9,
        "max_height_nm": step * (levels - 1) * 1e9,
        "pixel_pitch_um": pixel_pitch_um * d["pitch_scale"] if pixel_pitch_um else None,
        "footprint_um": [d["resample"]["source_shape"][1] * pixel_pitch_um,
                         d["resample"]["source_shape"][0] * pixel_pitch_um] if pixel_pitch_um else None,
        "resample": d["resample"], "source_pixel_pitch_um": pixel_pitch_um,
        "rms_quantization_error_rad": d["rms_err"],
        "ideal_efficiency_sinc2": d["efficiency"],
        "level_fill_fraction": (counts / counts.sum()).round(4).tolist(),
        "files": [],
    }

    if export_continuous:                                                    # stage 1: continuous
        raw = d["phi_raw"]
        np.save(p := out.with_name(out.name + "_continuous.npy"), raw); report["files"].append(str(p))
        wrapped16 = np.round(np.mod(raw, TWO_PI) / TWO_PI * 65535).astype(np.uint16)
        Image.fromarray(wrapped16).save(p := out.with_name(out.name + "_continuous.png"))
        report["files"].append(str(p))
        report["continuous_shape"] = list(raw.shape)

    if output in ("png", "both"):                                            # stage 2: discrete
        report["files"] += [str(p) for p in save_pngs(q, levels, d["heights"], out, litho_masks, step)]

    if output in ("stl", "both"):
        s = _UNIT[stl_units]
        ntri = 4 * q.size + 4 * (q.shape[0] + q.shape[1])
        if ntri > 20_000_000:
            print(f"[warn] >{ntri / 1e6:.0f}M triangles (>{ntri * 50 / 1e9:.1f} GB); consider cropping/binning")
        tris = heightfield_to_triangles(q, step * stl_z_scale / s, pitch / s, stl_base_um * 1e-6 / s)
        p = out.with_name(out.name + ".stl")
        write_binary_stl(p, tris, f"phase_mask_quantizer L={levels} lam={d['lam'] * 1e9:.1f}nm units={stl_units}")
        report["files"].append(str(p))
        report["stl_triangles"] = int(len(tris))
        report["stl_units"] = stl_units

    if save_npy:
        np.save(p := out.with_name(out.name + "_levels.npy"), q); report["files"].append(str(p))
        np.save(p := out.with_name(out.name + "_height_m.npy"), d["heights"]); report["files"].append(str(p))

    with open(p := out.with_name(out.name + "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    report["files"].append(str(p))
    return report


def run(input_file, pixel_pitch_um, wavelength_nm, output="both", *, mat_key=None,
        phase_unit="rad", invert=False, phase_scale=1.0, levels=6, material="fused_silica", n_index=None,
        substrate="fused_silica", n_substrate=None, step_nm=None, rounding="floor", resample="original", factor=1.0, n_medium=1.0,
        output_dir=None, name_override=None, **export_opts) -> dict:
    """Load -> (resample) -> quantize -> export.
    export_opts: export_continuous, litho_masks, save_npy, stl_base_um, stl_units, stl_z_scale."""
    phi = load_phase(input_file, mat_key, phase_unit)
    d = build_design(phi, wavelength_nm=wavelength_nm, levels=levels, material=material,
                     n_index=n_index, n_medium=n_medium, invert=invert, phase_scale=phase_scale,
                     substrate=substrate, n_substrate=n_substrate, step_nm=step_nm, rounding=rounding,
                     resample=resample, factor=factor)
    out = output_path(d, input_file, pixel_pitch_um, output_dir, name_override)
    return export_design(d, out, input_file, pixel_pitch_um=pixel_pitch_um,
                         output=output, material=material, n_index=n_index, n_medium=n_medium,
                         substrate=substrate, n_substrate=n_substrate, **export_opts)


# ==========================================================================
# GUI
# ==========================================================================
WAVELENGTHS = ["405", "450", "488", "532", "633", "780", "850", "1064", "1550"]
LEVELS = ["2", "3", "4", "6", "8", "16", "32"]
PITCHES = ["0.25", "0.5", "1", "2", "5", "10"]
MEDIA = ["1.0", "1.33", "1.45"]
BASES = ["0", "5", "10", "20", "50"]
ZSCALES = ["1", "2", "5", "10"]
FACTORS = ["1.5", "2", "4", "8"]
MATERIAL_CHOICES = list(MATERIALS)           # names; a typed number is used as n directly
PAD = 10


def fmt_length(um: float) -> str:
    """Micrometres -> '1886 µm (1.886 mm)'."""
    return f"{um:.0f} µm ({um / 1000:.3f} mm)" if um >= 1000 else f"{um:.1f} µm"


class Panel(ttk.Frame if ttk else object):
    """A titled canvas that shows a grayscale array scaled to fit, keeping aspect."""

    def __init__(self, master, title):
        super().__init__(master)
        ttk.Label(self, text=title, font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 4))
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0, width=260, height=260)
        self.canvas.pack(fill="both", expand=True)
        self.caption = ttk.Label(self, text=" ", foreground="#777")
        self.caption.pack(anchor="w", pady=(4, 0))
        self.arr = None                                  # uint8 (H, W)
        self._photo = None
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def show(self, arr, caption=" "):
        self.arr = arr
        self.caption.config(text=caption)
        self.redraw()

    def redraw(self):
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if self.arr is None:
            self.canvas.create_text(cw // 2, ch // 2, text="no data", fill="#666")
            return
        h, w = self.arr.shape
        scale = min(cw / w, ch / h)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = Image.fromarray(self.arr, mode="L").resize(size, Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)


class App(tk.Tk if tk else object):
    def __init__(self):
        super().__init__()
        self.title("Phase Mask Quantizer")
        self.geometry("1180x720")
        self.minsize(960, 600)

        self.raw = None                # loaded phase (radians for images / as stored otherwise)
        self.design = None             # last build_design() result
        self.after_id = None
        self._quiet = False
        self.v = {}                    # tk variables by name
        self.rows = {}                 # name -> (label, widget), so rows can be hidden
        self._build_ui()
        self.bind("<Command-e>", lambda e: self.export())
        self.bind("<Control-e>", lambda e: self.export())
        self.status.set("Open a .mat, .npy, .png or .bmp phase file to begin.")

    # ---------------------------------------------------------------- widgets
    def _var(self, name, value, cls=tk.StringVar):
        self.v[name] = cls(value=value)
        self.v[name].trace_add("write", lambda *_: self._changed(name))
        return self.v[name]

    def _section(self, parent, title):
        ttk.Label(parent, text=title, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(PAD, 2))
        f = ttk.Frame(parent)
        f.pack(fill="x")
        f.columnconfigure(1, weight=1)
        return f

    def _field(self, parent, label, name, values, default, readonly=False):
        """Dropdown you can also type into (readonly=True: pick only)."""
        r = parent.grid_size()[1]
        lab = ttk.Label(parent, text=label)
        lab.grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        w = ttk.Combobox(parent, textvariable=self._var(name, default), values=values,
                         state="readonly" if readonly else "normal", width=14)
        w.grid(row=r, column=1, sticky="ew", pady=2)
        self.rows[name] = (lab, w)
        return w

    def _entry_row(self, parent, label, name, button=None):
        r = parent.grid_size()[1]
        lab = ttk.Label(parent, text=label)
        lab.grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        box = ttk.Frame(parent)
        box.grid(row=r, column=1, sticky="ew", pady=2)
        e = ttk.Entry(box, textvariable=self._var(name, ""), width=14)
        e.pack(side="left", fill="x", expand=True)
        if button:
            ttk.Button(box, text=button[0], width=8, command=button[1]).pack(side="left", padx=(4, 0))
        self.rows[name] = (lab, box)
        return e

    def _check(self, parent, label, name, default=False):
        c = ttk.Checkbutton(parent, text=label, variable=self._var(name, default, tk.BooleanVar))
        c.grid(row=parent.grid_size()[1], column=0, columnspan=2, sticky="w", pady=2)
        self.rows[name] = (c, None)

    def _show(self, name, visible):
        for w in self.rows[name]:
            if w is not None:
                w.grid() if visible else w.grid_remove()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        side = ttk.Frame(self, padding=(PAD + 2, 4, PAD, PAD))
        side.pack(side="left", fill="y")
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y")
        main = ttk.Frame(self, padding=PAD + 2)
        main.pack(side="left", fill="both", expand=True)

        # Export pinned to the bottom of the sidebar
        bottom = ttk.Frame(side)
        bottom.pack(side="bottom", fill="x", pady=(PAD, 0))
        ttk.Button(bottom, text="Export", default="active", command=self.export).pack(fill="x", ipady=6)
        ttk.Label(bottom, text="⌘E / Ctrl+E", foreground="#888").pack(anchor="e")

        # --- input
        f = self._section(side, "Input")
        self._entry_row(f, "File", "input", ("Browse…", self.browse_input))
        self.mat_key = self._field(f, "MAT variable", "mat_key", [""], "")
        self._field(f, "Phase unit", "phase_unit", ["rad", "waves", "deg"], "rad", readonly=True)
        self._check(f, "Invert (conjugate mask)", "invert")

        # --- design
        f = self._section(side, "Design")
        self._field(f, "Wavelength (nm)", "wavelength", WAVELENGTHS, "532")
        self._field(f, "Levels", "levels", LEVELS, "6")
        self._field(f, "Step height (nm)", "step_nm", ["", "100", "150", "200", "250", "300"], "200")
        ttk.Label(f, text="Step height set = fixed-step quantization (overrides Levels).",
                  foreground="#888", wraplength=270).grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")
        self._field(f, "Pixel size (µm)", "pitch", PITCHES, "1")
        self._field(f, "Material", "material", MATERIAL_CHOICES, "fused_silica")
        self._field(f, "Substrate", "substrate", MATERIAL_CHOICES, "fused_silica")
        ttk.Label(f, text="Material / substrate: pick a preset or type an n value.",
                  foreground="#888", wraplength=270).grid(row=f.grid_size()[1], column=0, columnspan=2, sticky="w")

        # --- resampling: one control = mode toggle + factor (preset or typed)
        f = self._section(side, "Resampling")
        r = f.grid_size()[1]
        ttk.Label(f, text="Mode").grid(row=r, column=0, sticky="w", pady=2)
        seg = ttk.Frame(f)
        seg.grid(row=r, column=1, sticky="w", pady=2)
        self._var("resample", "original")
        for t in RESAMPLE_MODES:
            ttk.Radiobutton(seg, text=t.capitalize(), value=t, variable=self.v["resample"]).pack(side="left", padx=(0, 8))
        self.factor_box = self._field(f, "Factor (×)", "factor", FACTORS, "2")
        self.factor_box.config(state="disabled")

        # --- output
        f = self._section(side, "Output")
        r = f.grid_size()[1]
        ttk.Label(f, text="Format").grid(row=r, column=0, sticky="w", pady=2)
        seg = ttk.Frame(f)
        seg.grid(row=r, column=1, sticky="w", pady=2)
        self._var("output", "both")
        for t in ("png", "stl", "both"):
            ttk.Radiobutton(seg, text=t.upper(), value=t, variable=self.v["output"]).pack(side="left", padx=(0, 8))
        self._check(f, "Also export original continuous phase", "continuous", True)
        self._entry_row(f, "Folder", "output_dir", ("Choose…", self.browse_outdir))
        ttk.Label(f, text="Empty folder = next to the input file.", foreground="#888").grid(
            row=f.grid_size()[1], column=1, sticky="w")

        # --- advanced (collapsible)
        self.adv_open = tk.BooleanVar(value=False)
        self.adv_btn = ttk.Button(side, command=self._toggle_adv)
        self.adv_btn.pack(fill="x", pady=(PAD + 2, 0))
        self.adv = ttk.Frame(side)
        self.adv.columnconfigure(1, weight=1)
        self._field(self.adv, "Phase multiplier", "phase_scale", ["1", "2"], "1")
        self._field(self.adv, "n medium", "n_medium", MEDIA, "1.0")
        self._field(self.adv, "Step rounding", "rounding", ["floor", "round"], "floor", readonly=True)
        self._field(self.adv, "STL units", "stl_units", list(_UNIT), "um", readonly=True)
        self._field(self.adv, "STL base (µm)", "stl_base", BASES, "10")
        self._field(self.adv, "STL z-scale", "stl_z", ZSCALES, "1")
        self._entry_row(self.adv, "Name", "name")
        self._check(self.adv, "Litho binary masks (PNG)", "litho")
        self._check(self.adv, "Also save .npy arrays", "npy")
        self._toggle_adv(initial=True)

        # --- right side: previews + stats
        panels = ttk.Frame(main)
        panels.pack(fill="both", expand=True)
        self.panels = []
        for i, t in enumerate(["Input phase (wrapped 0–2π)", "Quantized levels"]):
            p = Panel(panels, t)
            p.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else PAD, 0))
            panels.columnconfigure(i, weight=1, uniform="p")
            self.panels.append(p)
        panels.rowconfigure(0, weight=1)

        ttk.Separator(main).pack(fill="x", pady=(PAD, 6))
        stats = ttk.Frame(main)
        stats.pack(fill="x")
        self.stat = {}
        keys = [("Footprint", "footprint"), ("Pixels", "pixels"), ("Step height", "step"),
                ("Max height", "maxh"), ("n material", "n"), ("n substrate", "nsub"),
                ("RMS error", "rms"), ("Efficiency", "eff"), ("Pixel size", "pitch")]
        for i, (label, key) in enumerate(keys):
            col, row = divmod(i, 3)                      # 3 columns of 3
            ttk.Label(stats, text=label, foreground="#888").grid(row=row, column=col * 2, sticky="w", padx=(0, 8), pady=1)
            self.stat[key] = ttk.Label(stats, text="—")
            self.stat[key].grid(row=row, column=col * 2 + 1, sticky="w", padx=(0, 20), pady=1)
        self.fill = ttk.Label(main, text="", foreground="#888")
        self.fill.pack(anchor="w", pady=(6, 0))
        self.status = tk.StringVar()
        ttk.Label(main, textvariable=self.status, foreground="#888").pack(anchor="w", pady=(6, 0))

    def _toggle_adv(self, initial=False):
        if not initial:
            self.adv_open.set(not self.adv_open.get())
        if self.adv_open.get():
            self.adv.pack(fill="x", pady=(4, 0), after=self.adv_btn)
        else:
            self.adv.pack_forget()
        self.adv_btn.config(text=("▾ " if self.adv_open.get() else "▸ ") + "Advanced (STL, medium, name)")

    # ---------------------------------------------------------------- actions
    def browse_input(self):
        p = filedialog.askopenfilename(filetypes=[("Phase files", "*.mat *.npy *.png *.bmp"), ("All", "*.*")])
        if p:
            self.v["input"].set(p)

    def browse_outdir(self):
        p = filedialog.askdirectory()
        if p:
            self.v["output_dir"].set(p)

    def _changed(self, name):
        if self._quiet:
            return
        if name == "resample":
            self.factor_box.config(state="disabled" if self.v["resample"].get() == "original" else "normal")
        if name == "input":
            self._load_input()
        elif name == "mat_key":
            self._load_input(keep_key=True)
        else:
            self._schedule()

    def _schedule(self):
        if self.after_id:
            self.after_cancel(self.after_id)
        self.after_id = self.after(150, self.update_preview)

    def _is_image(self):
        return Path(self.v["input"].get().strip()).suffix.lower() in IMAGE_EXTS

    def _load_input(self, keep_key=False):
        self.raw = self.design = None
        path = Path(self.v["input"].get().strip())
        if not path.is_file():
            return
        ext = path.suffix.lower()
        self._show("mat_key", ext == ".mat")
        self._show("phase_unit", ext not in IMAGE_EXTS)
        try:
            key = self.v["mat_key"].get().strip() or None
            if ext == ".mat" and not keep_key:
                names = mat_variables(path)
                self.mat_key.config(values=names)
                key = names[0] if len(names) == 1 else key if key in names else None
                self._quiet = True                       # don't re-trigger a load from our own set()
                self.v["mat_key"].set(key or "")
                self._quiet = False
            self.raw = load_phase(path, key, "rad")
        except Exception as e:                           # bad file -> show, don't crash
            self.status.set(f"Load failed: {e}")
            for p in self.panels:
                p.show(None)
            return
        self.update_preview()

    def _float(self, name, label, required=True):
        t = self.v[name].get().strip()
        if not t:
            if required:
                raise ValueError(f"{label} is required")
            return None
        try:
            return float(t)
        except ValueError:
            raise ValueError(f"{label} must be a number, got {t!r}") from None

    def _material(self, name, label):
        """Preset name -> (name, None); a typed number -> (name, n)."""
        t = self.v[name].get().strip()
        if t in MATERIALS:
            return t, None
        try:
            return "fused_silica", float(t)
        except ValueError:
            raise ValueError(f"{label} must be one of {list(MATERIALS)} or a number, got {t!r}") from None

    def _settings(self) -> dict:
        """Current settings as keyword arguments for build_design."""
        step_nm = self._float("step_nm", "Step height", False)
        levels = self._float("levels", "Levels", step_nm is None)
        if levels is not None and levels != int(levels):
            raise ValueError("Levels must be an integer")
        material, n_index = self._material("material", "Material")
        substrate, n_sub = self._material("substrate", "Substrate")
        return dict(wavelength_nm=self._float("wavelength", "Wavelength"), levels=int(levels or 2),
                    step_nm=step_nm, rounding=self.v["rounding"].get(),
                    material=material, n_index=n_index, substrate=substrate, n_substrate=n_sub,
                    n_medium=self._float("n_medium", "n medium"), invert=self.v["invert"].get(),
                    phase_scale=self._float("phase_scale", "Phase multiplier"),
                    resample=self.v["resample"].get(),
                    factor=1.0 if self.v["resample"].get() == "original" else self._float("factor", "Resample factor"))

    def update_preview(self):
        self.after_id = None
        if self.raw is None:
            return
        try:
            phi = self.raw if self._is_image() else apply_unit(self.raw, self.v["phase_unit"].get())
            d = self.design = build_design(phi, **self._settings())
            pitch = self._float("pitch", "Pixel size", False)
        except ValueError as e:
            self.status.set(f"Settings: {e}")
            return

        levels = d["levels"]
        ny, nx = phi.shape
        wrapped = np.mod(phi * self._float("phase_scale", "Phase multiplier") * (-1 if self.v["invert"].get() else 1), TWO_PI)
        self.panels[0].show((wrapped / TWO_PI * 255).astype(np.uint8),
                            f"{nx} × {ny} px    black = 0, white = 2π")
        self.panels[1].show((d["q"] * (255.0 / max(int(d["q"].max()), 1))).round().astype(np.uint8),
                            f"{d['q'].shape[1]} × {d['q'].shape[0]} px    {levels} levels    "
                            f"{d['step'] * 1e9:.1f} nm per level")

        counts = np.bincount(d["q"].ravel(), minlength=levels) / d["q"].size
        s = self.stat
        s["footprint"].config(text=f"{fmt_length(nx * pitch)} × {fmt_length(ny * pitch)}" if pitch else "— (set pixel size)")
        qy, qx = d["q"].shape
        s["pixels"].config(text=f"{qx} × {qy}" + ("" if (qx, qy) == (nx, ny) else f"  (from {nx} × {ny})"))
        s["pitch"].config(text=f"{pitch * d['pitch_scale']:.4g} µm" if pitch else "—")
        s["step"].config(text=f"{d['step'] * 1e9:.2f} nm")
        s["maxh"].config(text=f"{d['step'] * (levels - 1) * 1e9:.1f} nm")
        s["n"].config(text=f"{d['n']:.4f}")
        s["nsub"].config(text=f"{d['n_substrate']:.4f}")
        s["rms"].config(text=f"{d['rms_err']:.4f} rad")
        s["eff"].config(text=f"{d['efficiency']:.3f}")
        self.fill.config(text="Level fill:  " + "   ".join(f"{i}: {c * 100:.1f}%" for i, c in enumerate(counts)))
        self.status.set(f"Output name: {self._out_path().name}")

    def _out_path(self):
        return output_path(self.design, self.v["input"].get(), self._float("pitch", "Pixel size", False),
                           self.v["output_dir"].get().strip() or None, self.v["name"].get().strip() or None)

    def export(self):
        if self.design is None:
            messagebox.showwarning("Export", "Load a valid input file first.")
            return
        try:
            out = self._out_path()
            material, n_index = self._material("material", "Material")
            substrate, n_sub = self._material("substrate", "Substrate")
            kw = dict(pixel_pitch_um=self._float("pitch", "Pixel size", False),
                      output=self.v["output"].get(), material=material, n_index=n_index,
                      substrate=substrate, n_substrate=n_sub,
                      n_medium=self._float("n_medium", "n medium"),
                      litho_masks=self.v["litho"].get(), save_npy=self.v["npy"].get(),
                      stl_base_um=self._float("stl_base", "STL base"), stl_units=self.v["stl_units"].get(),
                      stl_z_scale=self._float("stl_z", "STL z-scale"),
                      export_continuous=self.v["continuous"].get())
        except ValueError as e:
            messagebox.showerror("Export", str(e))
            return

        design, src = self.design, self.v["input"].get()
        self.status.set("Exporting…")
        done = queue.Queue()

        def work():                                      # worker never touches Tk
            try:
                done.put(export_design(design, out, src, **kw))
            except Exception as e:
                done.put(e)

        def poll():
            try:
                res = done.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return
            if isinstance(res, Exception):
                self.status.set("Export failed")
                messagebox.showerror("Export", str(res))
            else:
                self._export_done(res)

        threading.Thread(target=work, daemon=True).start()
        poll()

    def _export_done(self, rep):
        self.status.set(f"Wrote {len(rep['files'])} files to {Path(rep['files'][0]).parent}")
        messagebox.showinfo("Export", "Wrote:\n" + "\n".join(Path(f).name for f in rep["files"]))


# ==========================================================================
# ENTRY POINT
# ==========================================================================
def cli(argv):
    ap = argparse.ArgumentParser(description="Continuous phase -> N-level PNG / STL.")
    ap.add_argument("input", help="continuous phase: .mat, .npy, or grayscale .png / .bmp (black = 0, white = 2pi)")
    ap.add_argument("pixel_pitch_um", type=float, help="mask pixel / feature size")
    ap.add_argument("wavelength_nm", type=float, help="design wavelength")
    ap.add_argument("output", nargs="?", default="both", choices=["png", "stl", "both"])
    ap.add_argument("--levels", type=int, default=6, help="number of discrete phase levels (>= 2)")
    ap.add_argument("--step-nm", type=float, help="fixed height step in nm (overrides --levels; PhlatCam style)")
    ap.add_argument("--rounding", default="floor", choices=["floor", "round"], help="used with --step-nm")
    ap.add_argument("--mat-key", help="variable name inside the .mat; default = the only 2-D array")
    ap.add_argument("--phase-unit", default="rad", choices=["rad", "waves", "deg"])
    ap.add_argument("--phase-scale", type=float, default=1.0, help="multiply the phase first (PhlatCam MATLAB uses 2)")
    ap.add_argument("--resample", default="original", choices=list(RESAMPLE_MODES),
                    help="re-grid the phase before quantizing: up / down by --factor (footprint is kept)")
    ap.add_argument("--factor", type=float, default=2.0, help="resample factor per axis (e.g. 1.5, 2, 4, 8)")
    ap.add_argument("--no-continuous", dest="export_continuous", action="store_false",
                    help="skip the original continuous export (<name>_continuous.npy / .png)")
    ap.add_argument("--invert", action="store_true", help="use -phase (conjugate mask)")
    ap.add_argument("--material", default="fused_silica", choices=list(MATERIALS))
    ap.add_argument("--n-index", type=float, help="override the material's refractive index")
    ap.add_argument("--n-medium", type=float, default=1.0, help="surrounding medium (1.0 = air)")
    ap.add_argument("--output-dir", help="default = same folder as the input")
    ap.add_argument("--name", dest="name_override", help="replace the auto-generated name")
    ap.add_argument("--litho-masks", action="store_true", help="PNG: also write binary masks (levels = 2^m)")
    ap.add_argument("--save-npy", action="store_true", help="also save level and height arrays as .npy")
    ap.add_argument("--stl-base-um", type=float, default=10.0, help="base slab thickness")
    ap.add_argument("--stl-units", default="um", choices=list(_UNIT))
    ap.add_argument("--stl-z-scale", type=float, default=1.0, help=">1 exaggerates heights (visualisation only)")
    a = vars(ap.parse_args(argv))
    try:
        report = run(a.pop("input"), a.pop("pixel_pitch_um"), a.pop("wavelength_nm"), a.pop("output"), **a)
    except ValueError as e:
        sys.exit(f"[error] {e}")

    print(f"n({report['wavelength_nm']:.1f} nm) = {report['n']:.5f}   "
          f"2pi depth = {report['depth_2pi_nm']:.1f} nm   "
          f"step = {report['step_height_nm']:.2f} nm   max height = {report['max_height_nm']:.1f} nm")
    print(f"RMS quantization error = {report['rms_quantization_error_rad']:.4f} rad   "
          f"ideal efficiency ({report['levels']} levels) = {report['ideal_efficiency_sinc2']:.3f}")
    for f in report["files"]:
        print("  wrote", f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli(sys.argv[1:])
    elif tk is None:
        sys.exit("[error] tkinter is not available; use the command line (see --help)")
    else:
        App().mainloop()
