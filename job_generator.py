#!/usr/bin/env python3
"""
job_generator.py
----------------
Height map (.png / .bmp / .mat / .npy) or STL  ->  sliced into fields  ->  Nanoscribe job files.

Two slicing methods
    standard : rectangular fields of `field_size` um, written row by row (DeScribe-style splitting).
               For an STL input the fields are TipSlicer's regular square tiling.
    adaptive : TipSlicer's adaptive stitching (edge voting): field centres are placed where a write disc of
               `adaptive_radius` covers most edge of the pattern, so the seams fall in the solid interior
               and not through the features. Fields are Voronoi cells, written in a travelling-salesman order.

Writing parameters (speed, power, slicing / hatching distance, ...) come from a recipe (recipes.py).

Output, named like a DeScribe DOE package (<name> = base name of the output, default the input file name):
    <name>.png  <name>.recipe  <name>_job.recipe    model image and DeScribe-style recipes (quantized height maps only)
    <name>_job.gwl                                  writing configuration, laser power / speed, includes the data file
    <name>_data.gwl                                 stage moves, '% FIELD', FindInterfaceAt, one include per field
    <name>_files/<name>_<i>_<j>.gwl                 one file per field (galvo coordinates relative to the field centre)
    work/                                           H5 stacks, TSP files (only with keep_intermediate)

TipSlicer (third_party/tipslicer, GPL-3) does the slicing, H5 stacks and hatching. It needs opencv, pytables and
matplotlib; an STL input also needs numpy-stl. They are imported when a job runs, so the rest of the app works without.
"""
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import gradient_segmentation as gs
import phase_quality as pqg
import route_solvers as rs
import height_map_stl as hms
import quantizer as pq
from recipes import Recipe

METHODS = ("standard", "adaptive")
# how the adaptive method finds its fields
SEGMENTATIONS = ("footprint_edges",          # default: edge voting on the footprint
                 "height_threshold",         # edge voting on the cross-section at one height (segment_height_um)
                 "height_gradient_guided",  # block borders attracted to every height step (gradient_segmentation.py)
                 "phase_quality_guided")    # minimum-cost seams (graph cut) on the height-gradient map (phase_quality.py)
LABEL_MODES = ("height_gradient_guided", "phase_quality_guided")   # segmentations that produce a label image, not Voronoi cells
THIRD_PARTY = Path(__file__).resolve().parent / "third_party" / "tipslicer"
INPUT_EXTS = hms.HEIGHT_MAP_EXTS + (".stl",)
SQRT2 = float(np.sqrt(2.0))


class JobCancelled(Exception):
    """Raised inside a running job when the caller asked to cancel."""


@dataclass
class JobSettings:
    """What is specific to one job; the writing parameters come from the recipe."""
    input_file: str = ""
    output_dir: str = ""
    name: str = ""                       # base name of all files; empty = input file name
    method: str = "standard"             # "standard" or "adaptive"
    pixel_size_um: float = None          # height maps only
    height_max_um: float = None          # peak height of the design; images need it, .mat / .npy may leave it out
    levels: int = None                   # snap to this many equidistant levels (DOE); None = continuous heights
    base_um: float = 0.2                 # solid base under the lowest level (DeScribe: base slices x slicing distance)
    key: str = None                      # variable name inside a .mat
    value_unit: str = "um"               # unit of .mat / .npy values when height_max_um is empty
    image_top_is_plus_y: bool = True     # image row 0 is the +y edge (as found in DeScribe's own files)
    keep_intermediate: bool = False      # keep the H5 stacks and TSP files in <output>/work
    stl_scale: float = 1.0               # STL input: multiply coordinates by this to get um
    heights_um: np.ndarray = None        # heights in um handed over by the Quantizer (row 0 = +y); replaces reading the file
    model_png: str = None                # PNG to copy as <name>.png when heights_um is given
    export_sliced_png: bool = True       # adaptive jobs also write <name>_sliced_standard_square.png / _sliced_adaptive.png
    segment_height_um: float = None      # height-based adaptive slicing: z (um, base included) at which the fields are
                                         # segmented; None = off (the fields follow the whole footprint)
    segmentation_mode: str = None        # one of SEGMENTATIONS; None = "height_threshold" if segment_height_um is set,
                                         # else "footprint_edges"
    ridge_cost: float = None             # height_gradient_guided: cost of crossing a full-gradient pixel (None = default)
    route_solvers: tuple = rs.ROUTE_AUTO # writing order of phase_quality_guided: exact (pyconcorde, if installed) else 2-opt
    route_time_limit: float = rs.DEFAULT_TIME_LIMIT   # seconds the exact solver may take
    global_interface_found: bool = False  # Global interfaceFound: FindInterfaceAt once, at the first field (default: before every field)


# --------------------------------------------------------------------------
# TipSlicer (imported lazily)
# --------------------------------------------------------------------------
_TIPSLICER = None


def tipslicer():
    """The bundled TipSlicer module. Raises RuntimeError with the install hint if a dependency is missing."""
    global _TIPSLICER
    if _TIPSLICER is None:
        os.environ.setdefault("MPLBACKEND", "Agg")           # TipSlicer imports pyplot; never open a GUI backend
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        try:
            import TipSlicer
        except ImportError as e:
            raise RuntimeError(f"Job generation needs opencv, pytables and matplotlib ({e}).\n"
                               "Install with: pip install opencv-python-headless tables matplotlib") from e
        _TIPSLICER = TipSlicer
    return _TIPSLICER


class _Ctx:
    """Progress / log / cancel hooks, all optional."""

    def __init__(self, progress=None, log=None, cancel=None):
        self._progress, self._log, self._cancel = progress, log, cancel

    def log(self, msg):
        if self._log:
            self._log(msg)

    def progress(self, stage, done, total):
        if self._progress:
            self._progress(stage, done, total)

    def check(self):
        if self._cancel and self._cancel():
            raise JobCancelled("cancelled")


# --------------------------------------------------------------------------
# Stitching helpers
# --------------------------------------------------------------------------
class Cells:
    """The field area of every centre: its Voronoi cell. TipSlicer's polygon route needs at least 4 non-degenerate
    centres; for fewer (a design smaller than a field) the same cell is rasterised from the nearest centre."""

    def __init__(self, ts, centres):
        self.ts, self.centres, self.vor, self.tree = ts, np.asarray(centres), None, None
        if len(self.centres) >= 4:
            try:
                self.vor = ts.makeVoronoiPolygons(self.centres)
            except Exception:                                  # qhull: collinear / degenerate centres
                self.vor = None
        if self.vor is None:
            from scipy.spatial import cKDTree
            self.tree = cKDTree(self.centres)

    def mask(self, idx, Xloc, Yloc):
        if self.vor is not None:
            regions, vertices = self.vor
            return self.ts.makeVoronoiMask(vertices[regions[idx]], Xloc, Yloc)
        _, nearest = self.tree.query(np.column_stack((Xloc.ravel(), Yloc.ravel())))
        return (nearest.reshape(Xloc.shape) == idx).astype(np.uint8)


def tsp_order(ts, points, work_dir, ctx):
    """Writing order of the field centres: Concorde ($CONCORDE or PATH) if available, else nearest neighbour."""
    if len(points) < 3:                                        # nothing to optimise
        return np.arange(len(points))
    tsp, sol = str(Path(work_dir) / "blocks.tsp"), str(Path(work_dir) / "tsp.sol")
    ts.writeTSPfile(tsp, points)
    exe = os.environ.get("CONCORDE")
    exe = exe if exe and os.path.isfile(exe) else shutil.which("concorde")
    if exe is not None:
        subprocess.run([exe, "-o", sol, tsp], check=True)
        return ts.readTSPfile(sol)
    ctx.log("no Concorde executable on PATH: the starting order is nearest neighbour (the route solver is reported separately)")
    order, _ = ts.TSPnearest(points)
    return order


def shear_vector(rec: Recipe):
    """Block shear angle (DeScribe 'Block shear angle'): the block faces lean by `block_shear_angle` from the substrate normal
    towards `block_shear_direction`, so the seams between blocks are inclined. Returns (kx, ky), the lateral shift in um per
    um of height above the block base; None for 0 degrees (vertical seams, the default)."""
    t = np.tan(np.radians(rec.block_shear_angle))
    if t <= 0:
        return None
    d = np.radians(rec.block_shear_direction)
    return float(t * np.cos(d)), float(t * np.sin(d))


def shear_pad(r, height, ctx):
    """How much the stack frame of a field has to grow so the leaning block still lies inside it (0 without shear). Warns
    when the grown frame no longer fits the scan field of the objective (DeScribe warns about the Block width likewise)."""
    k = r.get("shear")
    if k is None:
        return 0.0
    pad = float(np.ceil(np.hypot(*k) * height / r["hashing"] - 1e-9) * r["hashing"])
    side = 2 * (r["frame_half"] + pad)
    if side * SQRT2 > r["scan_field"] + 1e-6:
        ctx.log(f"WARNING: with the block shear the stack frame diagonal ({side * SQRT2:.0f} um) is larger than the scan field "
                f"({r['scan_field']:.0f} um); reduce the shear angle or the field radius")
    return pad


def sheared_mask_fn(mask_at_base, k, z_base):
    """z -> mask of a leaning block: the outline of the vertical block (mask_at_base(dx, dy) samples it shifted) moved by
    (z - z_base) * k. Every block moves the same way, so the blocks still tile the structure with no gap and no overlap."""
    def mask(z):
        h = z - z_base
        return mask_at_base(k[0] * h, k[1] * h)
    return mask


def block_grid(xc, yc, half, hashing):
    """Square frame of 2*half around a field centre, sampled every `hashing` um."""
    am = int(np.round(2 * half / hashing)) + 1
    return np.meshgrid(np.linspace(xc - half, xc + half, am), np.linspace(yc - half, yc + half, am))


def _dilate(mask, iterations):
    import scipy.ndimage as ndi
    return ndi.binary_dilation(mask, iterations=iterations) if iterations >= 1 else mask.astype(bool)


def in_field(xc, yc, Xa, Ya, half):
    """Half-open square of side 2*half around (xc, yc): neighbouring fields share no pixel."""
    return (Xa >= xc - half) & (Xa < xc + half) & (Ya >= yc - half) & (Ya < yc + half)


def z_levels(z0, z1, slicing, seg=None, coarse=0.0):
    """z of the slice planes in [z0, z1): `slicing` apart; with a segmentation height `seg` and a coarse distance, the
    planes below `seg` are `coarse` apart and the ones above `slicing` apart (the notebook's zzbase / zzlenses)."""
    if seg is None or not coarse or coarse <= 0 or seg <= z0:
        return np.arange(z0, z1, slicing)
    below = np.arange(z0, min(seg, z1), coarse)
    above = np.arange(seg, z1, slicing) if seg < z1 else np.array([])
    return np.concatenate((below, above))


def height_based_centres(ts, T_full, T_seg, X, Y, pitch, radius):
    """Adaptive field centres from the cross-section of the structure at the segmentation height (T_seg): the seams
    avoid the features that stand above the base. The part of the footprint the discs do not reach (the plain base) is
    then covered by the usual edge voting, so every pixel of T_full still belongs to a field."""
    if not T_seg.any():
        raise ValueError("the structure has no material at the segmentation height")
    import cv2
    feat = ts.getOptimalStitchCentres(T_seg, radius, radius, X, Y)
    wr = int(round(radius / pitch))
    disc = np.zeros(T_full.shape, np.uint8)
    for x, y in feat:
        cv2.circle(disc, (int(round((x - X[0, 0]) / pitch)), int(round((y - Y[0, 0]) / pitch))), wr, 1, -1)
    rest = ((T_full.astype(int) - disc) > 0).astype(np.uint8)
    if not rest.any():
        return feat
    return np.vstack((feat, ts.getOptimalStitchCentres(rest, radius, radius, X, Y)))


def route_order(ts, centres, work_dir, ctx, route_solvers=rs.ROUTE_AUTO, route_time_limit=rs.DEFAULT_TIME_LIMIT, route_info=None):
    """Writing order of the adaptive field centres: the TipSlicer TSP order as the start, then route_solvers (exact through
    pyconcorde if installed, else 2-opt). If `route_info` is a dict it is filled with the solver, optimal flag, length, detail."""
    route = rs.solve_route(centres, lambda: tsp_order(ts, centres, work_dir, ctx), route_solvers, route_time_limit, log=ctx.log)
    if route_info is not None:
        route_info.update(route.info())
    return route.order


def gradient_layout(ts, H, T, X, Y, pitch, radius, work_dir, ctx, ridge_cost=None, route_solvers=rs.ROUTE_AUTO,
                    route_time_limit=rs.DEFAULT_TIME_LIMIT, route_info=None):
    """Height-gradient-guided fields for the height map H (um, rows grow with y) on the aperture T: field centres (n, 2) in
    um, writing order, and the label image (block id of every pixel, -1 outside). Shared by the job and the preview."""
    cen_px, labels, _ = gs.segment(H, pitch, radius, aperture=T,
                                   ridge_cost=gs.DEFAULT_RIDGE_COST if ridge_cost is None else ridge_cost)
    centres = np.column_stack((X[0, cen_px[:, 1]], Y[cen_px[:, 0], 0]))
    return centres, route_order(ts, centres, work_dir, ctx, route_solvers, route_time_limit, route_info), labels


def phase_quality_layout(ts, H, T, X, Y, pitch, radius, work_dir, ctx, route_solvers=rs.ROUTE_AUTO,
                         route_time_limit=rs.DEFAULT_TIME_LIMIT, route_info=None):
    """Phase-quality-guided fields (phase_quality.py): field centres (n, 2) in um, writing order and the label image.
    Shared by the job and the preview. The segmentation does not depend on the writing order: the order comes from
    route_solvers.solve_route (exact through pyconcorde if installed and requested, else the TipSlicer TSP order improved
    by 2-opt). If `route_info` is a dict it is filled with the route metadata (solver, optimal, length, detail)."""
    cen_px, labels, _ = pqg.segment(H, pitch, radius, aperture=T)
    centres = np.column_stack((X[0, cen_px[:, 1]], Y[cen_px[:, 0], 0]))
    return centres, route_order(ts, centres, work_dir, ctx, route_solvers, route_time_limit, route_info), labels


def height_map_layout(ts, T, X, Y, pitch, method, field_size, radius, work_dir, ctx, size_um=None, T_seg=None,
                      route_solvers=rs.ROUTE_AUTO, route_time_limit=rs.DEFAULT_TIME_LIMIT, route_info=None):
    """Field centres (n, 2) and their writing order for the footprint image T on the grid X, Y (y grows with the row).
    The job and the slicer preview both call this, so the preview is the plan that is written."""
    rows, cols = T.shape
    w, h = size_um if size_um else (cols * pitch, rows * pitch)      # physical size (the preview works on a reduced T)
    if method == "standard":                               # grid centred on the map, row by row (y, then x ascending)
        nx = int(np.ceil(w / field_size - 1e-9))
        ny = int(np.ceil(h / field_size - 1e-9))
        cands = [((i - (nx - 1) / 2.0) * field_size, (j - (ny - 1) / 2.0) * field_size)
                 for j in range(ny) for i in range(nx)]
        centres = np.array([c for c in cands if (T * in_field(c[0], c[1], X, Y, field_size / 2.0)).sum() > 0])
        return centres, np.arange(len(centres))
    if T_seg is not None:                                              # adaptive, height-based
        centres = height_based_centres(ts, T, T_seg, X, Y, pitch, radius)
    else:
        centres = ts.getOptimalStitchCentres(T, radius, radius, X, Y)   # adaptive: edge voting
    return centres, route_order(ts, centres, work_dir, ctx, route_solvers, route_time_limit, route_info)


def stitch_height_map(ts, H, pitch, method, r, work_dir, h5_dir, ctx):
    """Slice the height map H (um, row 0 = +y if r['top_plus_y']) into fields and write one H5 stack per field."""
    import scipy.ndimage as ndi
    if r["top_plus_y"]:                                    # y must increase with the row index below
        H = np.flipud(H)
    rows, cols = H.shape
    x = (np.arange(cols) - (cols - 1) / 2.0) * pitch
    y = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    X, Y = np.meshgrid(x, y)
    H = H + r["base"]                                      # solid base (DeScribe: BaseSliceCount x slice distance)
    order = 0 if r["nearest"] else 1                       # nearest keeps every pixel a flat plateau (DOE)

    def height_slice(Xl, Yl, z, _extra):
        coords = np.array([(Yl - y[0]) / pitch, (Xl - x[0]) / pitch])
        Hs = ndi.map_coordinates(H, coords, order=order, mode="nearest")
        inside = (coords[0] > -0.5) & (coords[0] < rows - 0.5) & (coords[1] > -0.5) & (coords[1] < cols - 0.5)
        return inside & (Hs > z + 1e-6)                    # tolerance: heights are multiples of the slice distance

    T = (H > 0).astype(np.uint8)                           # footprint = what the first layer has to write
    seg = r["seg"] if method == "adaptive" else None
    T_seg = (H > seg + 1e-6).astype(np.uint8) if seg is not None else None
    labels = None
    if method == "adaptive" and r["gradient"]:                                         # a label-image segmentation
        if r["mode"] == "phase_quality_guided":
            centres, order_idx, labels = phase_quality_layout(ts, H, T, X, Y, pitch, r["radius"], work_dir, ctx,
                                                              r["route_solvers"], r["route_time"], r["route_info"])
        else:
            centres, order_idx, labels = gradient_layout(ts, H, T, X, Y, pitch, r["radius"], work_dir, ctx, r["ridge"],
                                                         r["route_solvers"], r["route_time"], r["route_info"])
        cells = gs.LabelCells(labels, X[0, 0], Y[0, 0], pitch)
    else:
        centres, order_idx = height_map_layout(ts, T, X, Y, pitch, method, r["field_size"], r["radius"], work_dir, ctx, T_seg=T_seg,
                                              route_solvers=r["route_solvers"], route_time_limit=r["route_time"],
                                              route_info=r["route_info"])
        cells = Cells(ts, centres) if method == "adaptive" else None
    ctx.log(f"{len(centres)} fields")

    zz = z_levels(0, H.max() - 1e-6, r["slicing"], seg, r["coarse"])
    k = r.get("shear")
    pad = shear_pad(r, float(H.max()), ctx)                  # 0 without block shear
    for n, idx in enumerate(order_idx):
        ctx.check()
        Xloc, Yloc = block_grid(centres[idx, 0], centres[idx, 1], r["frame_half"] + pad, r["hashing"])
        if method == "standard":
            def base_mask(dx, dy, xc=centres[idx, 0], yc=centres[idx, 1], Xl=Xloc, Yl=Yloc):
                return in_field(xc, yc, Xl - dx, Yl - dy, r["field_size"] / 2.0)
        else:
            def base_mask(dx, dy, idx=idx, Xl=Xloc, Yl=Yloc):
                return cells.mask(idx, Xl - dx, Yl - dy)
        path = str(Path(h5_dir) / f"block_{n:05d}.h5")
        if k is None:
            ts.formulaToStack(path, Xloc, Yloc, zz, height_slice, extraParams=0,
                              writingMask=_dilate(base_mask(0.0, 0.0), r["overlap_iter"]))
        else:                                                # leaning block: the outline depends on z
            mask_z = sheared_mask_fn(base_mask, k, 0.0)
            ts.formulaToStack(path, Xloc, Yloc, zz,
                              lambda Xa, Ya, z, e, m=mask_z: height_slice(Xa, Ya, z, e) * _dilate(m(z), r["overlap_iter"]),
                              extraParams=0)
        ctx.progress("Slicing fields", n + 1, len(order_idx))
    return len(centres), centres, order_idx, labels


def stitch_stl(ts, stl_path, method, r, work_dir, h5_dir, ctx, name):
    """STL input: low-resolution stack of the whole object, field centres per Z layer, 3D ordering."""
    import tables
    try:
        mesh, Zmins, Zmaxs = ts.loadStl(str(stl_path), mulDimension=r["stl_scale"], centred=False)
    except ImportError as e:
        raise RuntimeError(f"An STL input needs numpy-stl ({e}).\nInstall with: pip install numpy-stl") from e
    low_h, low_s = 1.0, 1.0
    X, Y, Z = ts.getMeshXYZ(mesh, low_h, low_s)
    low_h5 = str(Path(h5_dir) / f"{name}_lowres.h5")
    ctx.log("slicing the whole object at low resolution")
    ts.stlToStack(mesh, low_h5, X, Y, Z, Zmins, Zmaxs)

    am_z = int((Z.max() - Z.min()) / r["fov_z"]) + 1
    z_range = (Z.max() - Z.min()) / am_z
    ctx.log(f"{am_z} Z layer(s) of {z_range:.2f} um")
    centre_coords = []
    for k in range(am_z):
        ground, Xp, Yp = ts.getProjection(low_h5, k * z_range, (k + 1) * z_range + r["overlap_z"])
        if method == "adaptive":
            seg = r["seg"]
            if seg is not None and k * z_range <= seg < (k + 1) * z_range + r["overlap_z"]:      # height-based
                with tables.open_file(low_h5, mode="r") as h:
                    zs = h.root.Z[:]
                    cross = h.root.data[int(np.argmin(np.abs(zs - seg)))].astype(np.uint8)
                centres = height_based_centres(ts, ground, cross, Xp, Yp, Xp[0, 1] - Xp[0, 0], r["radius"])
            else:
                centres = ts.getOptimalStitchCentres(ground, r["radius"], r["radius"], Xp, Yp)
        else:                                              # standard on an STL: regular squares of field_size
            try:
                centres = ts.getSquareStitchCentres(ground, r["field_size"] / 2.0, Xp, Yp)
            except Exception:                              # fewer than 4 squares: one field around the object
                mat = np.argwhere(ground > 0)
                centres = np.array([[Xp[mat[:, 0], mat[:, 1]].mean(), Yp[mat[:, 0], mat[:, 1]].mean()]])
        centre_coords.append(np.column_stack((centres, np.ones(len(centres)) * k * z_range)))

    grow = 1000                                            # Z scaled up so the layers stay together in the ordering
    pp = np.asarray([p for cc in centre_coords for p in cc * np.array([1, 1, grow])])
    info = {}                                              # same solver as the height-map modes (pyconcorde if installed, else 2-opt)
    order = route_order(ts, pp[:, :2] if am_z == 1 else pp, work_dir, ctx, r["route_solvers"], r["route_time"], info)
    ordered = pp[order] * np.array([1, 1, 1 / grow])
    if ordered[0, 2] > ordered[-1, 2]:
        ordered = ordered[::-1]
    layers = [ordered[np.argwhere(np.abs(ordered[:, 2] - z0) < 1)[:, 0]] for z0 in np.unique(ordered[:, 2])]
    info["length_um"] = pqg.travel_distance(np.vstack(layers), np.arange(len(ordered)))      # the stage path as it is written
    r["route_info"].update(info)

    n = 0
    for z0, layer in zip(np.unique(ordered[:, 2]), layers):
        zz = z_levels(z0, z0 + z_range + r["slicing"], r["slicing"], r["seg"] if method == "adaptive" else None, r["coarse"])
        centres = layer[:, :2]
        cells = Cells(ts, centres)
        k = r.get("shear")
        pad = shear_pad(r, float(z_range + r["slicing"]), ctx)       # 0 without block shear
        for i in range(len(centres)):
            ctx.check()
            Xloc, Yloc = block_grid(centres[i, 0], centres[i, 1], r["frame_half"] + pad, r["hashing"])
            if k is None:
                MM = _dilate(cells.mask(i, Xloc, Yloc), r["overlap_iter"])
            else:                                            # leaning block: the outline depends on z (from the layer base z0)
                def base_mask(dx, dy, i=i, Xl=Xloc, Yl=Yloc):
                    return cells.mask(i, Xl - dx, Yl - dy)
                mask_z = sheared_mask_fn(base_mask, k, float(z0))
                MM = lambda z, m=mask_z: _dilate(m(z), r["overlap_iter"])
            ts.stlToStack(mesh, str(Path(h5_dir) / f"block_{n:05d}.h5"), Xloc, Yloc, zz, Zmins, Zmaxs,
                          inFillMaster=low_h5, writingMask=MM)
            n += 1
            ctx.progress("Slicing fields", n, len(ordered))
    return n


# --------------------------------------------------------------------------
# Package files: job, data, fields, recipes
# --------------------------------------------------------------------------
def _num(v):
    return f"{round(v, 3):g}"


def write_job_file(path, data_name, rec: Recipe, notes):
    """<name>_job.gwl, line by line the job file of a DeScribe DOE package; only the values come from the recipe.
    Everything this app adds is a comment at the end."""
    lines = ["% File generated by NanoSlicer (job layout of DeScribe 2.6)", "",
             "% System initialization", "InvertZAxis 1", "",
             "% Writing configuration", "GalvoScanMode", "ContinuousMode",
             f"PiezoSettlingTime {_num(rec.piezo_settling_time)}", f"GalvoAcceleration {_num(rec.galvo_acceleration)}",
             f"StageVelocity {_num(rec.stage_velocity)}", "",
             "% Scan field offsets", "XOffset 0", "YOffset 0", "ZOffset 0", "",
             "% Writing parameters", "PowerScaling 1.0", "",
             "% Solid hatch lines writing parameters",
             f"var $solidLaserPower = {_num(rec.laser_power)}", f"var $solidScanSpeed = {_num(rec.scan_speed)}", "",
             "% Base writing parameters", "var $baseLaserPower = $solidLaserPower", "var $baseScanSpeed = $solidScanSpeed", "",
             f"var $interfacePos = {_num(rec.interface_pos)}", "",
             "% Include slicer output", f"include {os.path.basename(data_name)}", "",
             "% Notes from NanoSlicer (comments only, they do not change the job above):"]
    lines += ["%   " + n for n in notes]
    lines += ["%   The field files write with ScanSpeed $solidScanSpeed and LaserPower $solidLaserPower, in galvo",
              f"%   coordinates relative to the field centre; {os.path.basename(data_name)} moves the stage and calls "
              "FindInterfaceAt.", ""]
    with open(path, "w", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")


def write_data_file(path, files_dir, fields, header, field_size, interface_pos, grid, global_interface=False):
    """<name>_data.gwl in the layout of a DeScribe data file: documented comment header, relative stage moves, then
    '% FIELD i|j', FindInterfaceAt and an include for every field. No variables, no per-block logic.
    global_interface: FindInterfaceAt only before the first field (the z reference found there stays valid for the
    rest); a comment marks where the later calls were left out."""
    half = field_size / 2.0
    L = ["% " + h if h else "% " for h in header]
    f0 = fields[0]
    L += ["MoveStageY " + _num(f0["yc"] - half), "MoveStageX " + _num(f0["xc"] - half),
          "MoveStageY " + _num(half), "MoveStageX " + _num(half)]
    prev = None
    for f in fields:
        if prev is not None:
            dx, dy = f["xc"] - prev["xc"], f["yc"] - prev["yc"]
            if abs(dy) < 1e-6:
                L.append("MoveStageX " + _num(dx))
            elif grid:         # new row: back to the first field's corner, one row up, half a field in
                L += ["MoveStageX " + _num(f["xc"] - half - prev["xc"]), "MoveStageY " + _num(dy), "MoveStageX " + _num(half)]
            else:
                L += ["MoveStageX " + _num(dx), "MoveStageY " + _num(dy)]
        L.append("% FIELD " + f["label"])
        if interface_pos >= 0:
            if prev is None or not global_interface:
                L.append("FindInterfaceAt $interfacePos")
            else:
                L.append("% FindInterfaceAt $interfacePos (Global interfaceFound: first field only)")
        L.append("include " + files_dir + "\\" + f["file"])
        prev = f
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\r\n".join(L) + "\r\n")


def global_interface_gwl(text, command="FindInterfaceAt", comment=True):
    """Post-process the text of a data file: keep the first `command` call and turn every later one into a comment
    (comment=True) or drop it. Only those lines change; the line endings, the stage moves, the field order and the
    includes stay as they are. Returns (new text, number of calls found). A later call with a different argument is
    replaced too, so every field uses the z reference found by the first call."""
    pat = re.compile(r"^[ \t]*" + re.escape(command) + r"\b", re.I)
    out, seen = [], 0
    for line in text.splitlines(keepends=True):
        if pat.match(line):
            seen += 1
            if seen > 1:
                if comment:
                    body = line.rstrip("\r\n")
                    out.append("% " + body.strip() + " (Global interfaceFound: first field only)" + line[len(body):])
                continue
        out.append(line)
    return "".join(out), seen


def global_interface_file(path, output=None, command="FindInterfaceAt", comment=True):
    """global_interface_gwl on a file (written to `output`, default: in place). Returns the number of calls found."""
    with open(path, newline="", encoding="utf-8") as f:
        text = f.read()
    new, seen = global_interface_gwl(text, command, comment)
    with open(output or path, "w", newline="", encoding="utf-8") as f:
        f.write(new)
    return seen


def write_recipes(base, png_path, shape, pitch, levels, design_height, base_slices, rec: Recipe, block_size, split_mode):
    """<name>.recipe and <name>_job.recipe with the keys of the DeScribe DOE recipes."""
    rows, cols = shape
    hx, hy = cols * pitch / 2.0, rows * pitch / 2.0
    top = design_height + base_slices * rec.slicing_distance
    model = ["Version = 1.4", "Model.Type = Doe", f"Model.FilePath = {png_path}",
             f"Model.BoundingBox = Minimum:X:{-hx:g} Y:{-hy:g} Z:0 Maximum:X:{hx:g} Y:{hy:g} Z:{top:g}",
             f"Doe.LevelCount = {levels}", f"Doe.DesignHeight = {design_height:g}",
             f"Doe.PixelSize = X:{pitch:g} Y:{pitch:g}", f"Doe.BaseSliceCount = {base_slices}",
             f"Doe.FieldOverlap = {rec.overlap:g}"]
    job = [f"Slicing.DistanceMax = {rec.slicing_distance:g}", f"Filling.HatchingDistance = {rec.hatching_distance:g}",
           f"Splitting.Mode = {split_mode}", f"Splitting.BlockSize = X:{block_size:g} Y:{block_size:g} Z:50",
           f"Exposure.GalvoAcceleration = {rec.galvo_acceleration:g}", f"Exposure.FindInterfaceAt = {rec.interface_pos:g}",
           f"Exposure.ShellLaserPower = {rec.laser_power:g}", f"Exposure.ShellScanSpeed = {rec.scan_speed:g}",
           f"Exposure.CoreLaserPower = {rec.laser_power:g}", f"Exposure.CoreScanSpeed = {rec.scan_speed:g}"]
    with open(f"{base}.recipe", "w") as f:
        f.write("\n".join(model + ["Doe.UseTiling = False", "Doe.TilingCount = X:1 Y:1"]) + "\n")
    with open(f"{base}_job.recipe", "w") as f:
        f.write("\n".join(model + job) + "\n")


def build_data_header(in_path, H, pitch, levels, design_height, base, base_slices, rec: Recipe, method):
    """Comment lines (without the leading '% ') of the data file, in the order of DeScribe's header."""
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    L = ["File generated by NanoSlicer (data layout of DeScribe 2.6)", "", "Creation time", "  " + now, "",
         "Source file", "  Type: " + ("Doe" if H is not None and levels else "Height map" if H is not None else "Stl"),
         "  Path: " + str(in_path), ""]
    if H is not None:
        top = H.max() + base
        hx, hy = H.shape[1] * pitch / 2, H.shape[0] * pitch / 2
        volume = float(((H + base) * pitch ** 2).sum()) * 1e-9                 # mm^3
        L += ["Volume", f"  {volume:.5f} mm³", "", "Bounding box", f"  Minimum X: {-hx:g} Y: {-hy:g} Z: 0",
              f"  Maximum X: {hx:g} Y: {hy:g} Z: {top:g}", "",
              "Doe" if levels else "Height map", f"  Resolution X: {H.shape[1]} Y: {H.shape[0]}"]
        if levels:
            L += [f"  LevelCount: {levels}", f"  DesignHeight: {design_height:g}", f"  LevelHeight: {H.max() / (levels - 1):g}"]
        L += [f"  PixelSize: X:{pitch:g} Y:{pitch:g}", f"  Field overlap: {rec.overlap:g}",
              f"  BaseSliceCount: {base_slices}", f"  BuildHeight: {top:g}", ""]
    L += ["Processing", f"  SliceDistance: {rec.slicing_distance:g}", f"  HatchLineDistance: {rec.hatching_distance:g}", "",
          "Splitting"]
    if method == "standard":
        L.append(f"  FieldSize X: {rec.field_size:g} Y: {rec.field_size:g}")
    else:
        L += ["  Adaptive stitching (TipSlicer edge voting)", f"  Vote and write radius: {rec.adaptive_radius:g}"]
    return L + [""]


def write_fields(ts, h5_dir, files_dir, name, rec: Recipe, grid, ctx, zscaler=None):
    """One field file per H5 stack. Returns the list of fields and the total written length per laser code."""
    h5s, _ = ts.findFiles(r"\d\.h5$", str(h5_dir))             # ends in a digit: skips *_lowres.h5
    if not h5s:
        raise RuntimeError("no field stacks were written")
    import tables
    cents = []                                                 # field centres from the stacks give the grid indices
    for ff in h5s:
        with tables.open_file(ff, mode="r") as h:
            gc = h.root.data.attrs.GalvoCentre
            cents.append((round(float(gc[0]), 2), round(float(gc[1]), 2)))
    xs, ys = sorted({c[0] for c in cents}), sorted({c[1] for c in cents})
    fields, total = [], None
    for n, ff in enumerate(h5s):
        ctx.check()
        xc, yc = cents[n]
        tag = f"{xs.index(xc)}_{ys.index(yc)}" if grid else f"{n:05d}"
        fname = f"{name}_{tag}.gwl"
        fields.append({"xc": xc, "yc": yc, "file": fname, "label": tag.replace("_", "|") if grid else str(n)})
        dists = ts.stackToGwl(ff, str(Path(files_dir) / fname), int(rec.shells), 1, 1,
                              atAngles=rec.hatch_angle,
                              codeSpeeds=[rec.scan_speed] * 3, codeIntens=[rec.laser_power] * 3, writeHeader=False,
                              writeColourH5=False, doTopBottom=True, fieldMode=True, ZscalerFunc=zscaler)
        if len(dists) > 0:
            total = dists if total is None else total + dists
        ctx.progress("Writing GWL files", n + 1, len(h5s))
    return fields, total


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def export_sliced_pngs(out_dir, name, rec: Recipe, js: JobSettings, H, in_path, plan, ctx, max_side=1100):
    """For an adaptive job: <name>_sliced_standard_square.png and <name>_sliced_adaptive.png, the same structure divided
    both ways, so the layouts can be compared. The adaptive image shows the field centres and writing order the job used
    (`plan`); for an STL both layouts are computed from the projection of the solid."""
    import slicer_preview as sp                            # imported here: slicer_preview itself imports this module
    ctx.log("writing the sliced-PNG comparison (standard square and adaptive)")
    if H is None:
        fp = sp.stl_footprint(in_path, js.stl_scale, max_side)
        adaptive = sp.compute_layout(fp, "adaptive", rec, ctx)
    else:
        fp = sp.design_footprint(H, js.pixel_size_um, js.base_um, js.image_top_is_plus_y, max_side, min_side=1000)
        adaptive = sp.Layout(np.asarray(plan[0], float), np.asarray(plan[1]), "adaptive", fp, rec.field_size, rec.adaptive_radius,
                             labels=plan[2], segmentation="height_gradient_guided" if plan[2] is not None else "footprint_edges")
    standard = sp.compute_layout(fp, "standard", rec, ctx)
    paths = [Path(out_dir) / f"{name}_sliced_standard_square.png", Path(out_dir) / f"{name}_sliced_adaptive.png"]
    for lay, p in zip((standard, adaptive), paths):
        sp.render(lay, rec.overlap).save(p)
    return paths


def export_comparison_png(out_dir, name, rec: Recipe, js: JobSettings, H, ctx, five=False, phase_plan=None):
    """<name>_sliced_comparison.png: the slicing modes side by side: standard square, threshold-based adaptive and
    height-gradient-guided; with five=True also footprint-edge adaptive and phase-quality-guided, plus a metrics table."""
    import slicer_preview as sp
    ctx.log("writing the slicing comparison")
    modes = ("standard", "footprint", "threshold", "gradient", "phase") if five else None
    img, _ = sp.comparison_image(H, js.pixel_size_um, js.base_um, js.image_top_is_plus_y, rec, js.segment_height_um, ctx=ctx,
                                 modes=modes, phase_plan=phase_plan)
    path = Path(out_dir) / f"{name}_sliced_comparison.png"
    img.save(path)
    return path


def export_path_png(out_dir, name, rec: Recipe, js: JobSettings, H, plan, max_side=1100):
    """<name>_writing_path.png: the fields in the order they are written (green = first, magenta = last)."""
    import slicer_preview as sp
    fp = sp.design_footprint(H, js.pixel_size_um, js.base_um, js.image_top_is_plus_y, max_side, min_side=1000)
    lay = sp.Layout(np.asarray(plan[0], float), np.asarray(plan[1]), "adaptive", fp, rec.field_size, rec.adaptive_radius,
                    labels=plan[2], segmentation=js.segmentation_mode)
    path = Path(out_dir) / f"{name}_writing_path.png"
    sp.render_path(lay).save(path)
    return path


def settings_from_design(design, output_dir, *, name="", method="standard", base_um=0.2, key=None, value_unit="um",
                         image_top_is_plus_y=True, keep_intermediate=False, segment_height_um=None,
                         segmentation_mode=None, ridge_cost=None, route_solvers=rs.ROUTE_AUTO,
                         route_time_limit=rs.DEFAULT_TIME_LIMIT, global_interface_found=False) -> JobSettings:
    """JobSettings for a state.Design (what the Quantizer sent, or a design built from a file)."""
    if design.heights_um is None:                              # an STL: the solid is the input
        return JobSettings(input_file=design.stl_path or design.source_path, output_dir=output_dir, name=name, method=method,
                           keep_intermediate=keep_intermediate, segment_height_um=segment_height_um,
                           segmentation_mode=segmentation_mode, ridge_cost=ridge_cost, route_solvers=route_solvers,
                           route_time_limit=route_time_limit, global_interface_found=global_interface_found)
    model = design.png_path if design.png_path and Path(design.png_path).is_file() else None
    return JobSettings(input_file=design.source_path, output_dir=output_dir, name=name, method=method,
                       pixel_size_um=design.pixel_size_um, height_max_um=design.height_max_um, levels=design.levels,
                       base_um=base_um, key=key, value_unit=value_unit, image_top_is_plus_y=image_top_is_plus_y,
                       keep_intermediate=keep_intermediate, heights_um=design.heights_um, model_png=model,
                       segment_height_um=segment_height_um, segmentation_mode=segmentation_mode, ridge_cost=ridge_cost,
                       route_solvers=route_solvers, route_time_limit=route_time_limit,
                       global_interface_found=global_interface_found)


def check_scan_field(rec: Recipe, method, ctx):
    diag = (rec.field_size if method == "standard" else 2 * rec.adaptive_radius) * SQRT2
    if diag > rec.scan_field_diameter + 1e-6:
        ctx.log(f"WARNING: the {'field' if method == 'standard' else 'stack frame'} diagonal ({diag:.0f} um) is larger "
                f"than the scan field of {rec.objective} ({rec.scan_field_diameter:.0f} um)")


def generate_job_files(js: JobSettings, rec: Recipe, progress=None, log=None, cancel=None) -> dict:
    """Slice the input and write the job files. Returns a report dict (files, number of fields, print time).
    Raises ValueError for bad settings, RuntimeError for a missing dependency, JobCancelled when cancelled."""
    ctx = _Ctx(progress, log, cancel)
    rec = Recipe.from_dict(rec.to_dict()).validate()
    if js.method not in METHODS:
        raise ValueError(f"slicing method must be one of {METHODS}, got {js.method!r}")
    in_path = Path(js.input_file).expanduser()
    given = js.heights_um is not None                      # heights handed over by the Quantizer
    if not given and not in_path.is_file():
        raise ValueError(f"input file not found: {in_path}")
    ext = in_path.suffix.lower()
    if not given and ext not in INPUT_EXTS:
        raise ValueError(f"input must be one of {INPUT_EXTS}, got {ext!r}")
    if not js.output_dir:
        raise ValueError("choose an output folder")
    is_stl = ext == ".stl" and not given

    H = None
    if given:
        if not js.pixel_size_um or js.pixel_size_um <= 0:
            raise ValueError("pixel size must be > 0")
        H = np.asarray(js.heights_um, dtype=np.float64)
        if H.ndim != 2 or js.base_um is None or js.base_um + H.min() <= 0:
            raise ValueError("the lowest column has zero height: set a base thickness > 0")
    elif not is_stl:
        if not js.pixel_size_um or js.pixel_size_um <= 0:
            raise ValueError("pixel size must be > 0")
        if ext in pq.IMAGE_EXTS and not js.height_max_um:
            raise ValueError("height max (um) is required for images: grey values are relative")
        if js.base_um is None or js.base_um < 0:
            raise ValueError("base thickness must be >= 0")
        H = hms.heights_from_array(hms.load_height_array(in_path, js.key), js.height_max_um or None,
                                   js.levels or None, js.value_unit)
        if js.base_um + H.min() <= 0:
            raise ValueError("the lowest column has zero height: set a base thickness > 0")
    mode = js.segmentation_mode or ("height_threshold" if js.segment_height_um is not None else "footprint_edges")
    if mode not in SEGMENTATIONS:
        raise ValueError(f"segmentation mode must be one of {SEGMENTATIONS}, got {mode!r}")
    if mode != "footprint_edges" and js.method != "adaptive":
        raise ValueError("height-based segmentation belongs to the adaptive slicing method")
    if mode in LABEL_MODES:
        if H is None:
            raise ValueError(f"{mode} segmentation needs a height map (it reads the heights), not an STL")
        if js.ridge_cost is not None and js.ridge_cost < 0:
            raise ValueError("ridge cost must be >= 0")
    if mode == "height_threshold" and js.segment_height_um is None:
        raise ValueError("height_threshold segmentation needs a segmentation height")
    seg = js.segment_height_um if mode == "height_threshold" else None
    if seg is not None:
        if seg < 0:
            raise ValueError("the segmentation height must be >= 0")
        if H is not None and not (H + js.base_um > seg + 1e-6).any():
            raise ValueError(f"nothing of the structure is higher than {seg:g} µm (peak {float((H + js.base_um).max()):g} µm); "
                             "lower the segmentation height")
    check_scan_field(rec, js.method, ctx)

    ts = tipslicer()
    name = (js.name.strip() or in_path.stem).replace(" ", "_")
    out_dir = Path(js.output_dir).expanduser().resolve()
    files_dir = out_dir / f"{name}_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    tmp = None
    if js.keep_intermediate:
        work_dir = out_dir / "work"
        work_dir.mkdir(exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="nanoslicer_")        # removed below; never a folder of yours
        work_dir = Path(tmp)
    h5_dir = work_dir / "h5"
    h5_dir.mkdir(exist_ok=True)

    hashing = rec.hatching_distance
    standard = js.method == "standard"
    frame_half = (rec.field_size / SQRT2 if standard else rec.adaptive_radius) + (rec.overlap if standard else 0.0)
    r = {"top_plus_y": js.image_top_is_plus_y, "base": js.base_um, "nearest": bool(js.levels),
         "field_size": rec.field_size, "radius": rec.adaptive_radius, "frame_half": frame_half, "hashing": hashing,
         "slicing": rec.slicing_distance, "overlap_iter": int(round(rec.overlap / hashing)),
         "fov_z": 50.0, "overlap_z": 0.0, "stl_scale": js.stl_scale,
         "seg": seg, "coarse": rec.base_slicing_distance if seg is not None else 0.0,
         "gradient": mode in LABEL_MODES, "mode": mode, "ridge": js.ridge_cost,
         "route_solvers": tuple(js.route_solvers), "route_time": js.route_time_limit, "route_info": {},
         "shear": shear_vector(rec), "scan_field": rec.scan_field_diameter}
    zscaler = None
    if seg is not None and rec.base_power_multiplier != 1.0:                 # laser power below the segmentation height
        zscaler = lambda z: rec.base_power_multiplier if z < seg - 1e-9 else 1.0
    base_slices = int(np.ceil((js.base_um or 0) / rec.slicing_distance - 1e-9))     # whole slices, at least one for a base
    t0 = time.time()
    try:
        # ---- slicing
        if is_stl:
            n_fields = stitch_stl(ts, in_path, js.method, r, work_dir, h5_dir, ctx, name)
            plan = None
        else:
            n_fields, centres_used, order_used, labels_used = stitch_height_map(ts, H, js.pixel_size_um, js.method, r, work_dir, h5_dir, ctx)
            plan = (centres_used, order_used, labels_used)

        # ---- package files
        files = []
        if H is not None and js.levels:
            png = out_dir / f"{name}.png"
            src_png = Path(js.model_png) if js.model_png and Path(js.model_png).is_file() else (
                in_path if ext == ".png" and not given else None)
            if src_png is not None:
                shutil.copyfile(src_png, png)
            else:
                grey = H / H.max() * (255 if js.levels <= 256 else 65535)
                Image.fromarray(np.round(grey).astype(np.uint8 if js.levels <= 256 else np.uint16)).save(png)
            write_recipes(str(out_dir / name), png, H.shape, js.pixel_size_um, js.levels, js.height_max_um or float(H.max()),
                          base_slices, rec, rec.field_size if standard else 2 * rec.adaptive_radius,
                          "Rectangular" if standard else "Adaptive (edge voting, TipSlicer)")
            files += [png, out_dir / f"{name}.recipe", out_dir / f"{name}_job.recipe"]

        fields, total = write_fields(ts, h5_dir, files_dir, name, rec, standard and not is_stl, ctx, zscaler)
        header = build_data_header(in_path, H, js.pixel_size_um, js.levels, js.height_max_um or (float(H.max()) if H is not None else 0),
                                   js.base_um, base_slices, rec, js.method)
        data_name, job_name = out_dir / f"{name}_data.gwl", out_dir / f"{name}_job.gwl"
        write_data_file(data_name, files_dir.name, fields, header, rec.field_size, rec.interface_pos, standard and not is_stl,
                        global_interface=js.global_interface_found)
        notes = [f"Source: {in_path.name}", f"Recipe: {rec.name} ({rec.objective}, {rec.mode}, {rec.resin})",
                 f"Slicing method: {js.method}",
                 f"slicing {rec.slicing_distance:g} um, hatching {rec.hatching_distance:g} um, shells {rec.shells}, "
                 f"speed {rec.scan_speed:g} um/s, power {rec.laser_power:g}, interface {rec.interface_pos:g} um"]
        if r["route_info"]:
            ri = r["route_info"]
            ctx.log(f"Route ({mode if js.method == 'adaptive' else 'standard'}): {ri['solver']}" + ("" if ri["optimal"] or not ri["detail"] else f" ({ri['detail']})"))
            if mode != "phase_quality_guided":
                notes.append(f"Writing order: {ri['solver']}" + (" (proven optimal open path)" if ri["optimal"] else " (improved, not proven optimal)")
                             + f", {ri['length_um'] / 1000:.1f} mm of stage travel")
        if r["shear"] is not None:
            notes.append(f"Block shear angle: {rec.block_shear_angle:g} deg from the substrate normal towards "
                         f"{rec.block_shear_direction:g} deg in xy (the seams between fields lean with the height; "
                         f"the fields themselves and their order are unchanged)")
        if js.global_interface_found and rec.interface_pos >= 0:
            notes.append("Global interfaceFound: ON - FindInterfaceAt runs once, at the first field; the other fields reuse that "
                         "interface position (global_interface_found)")
        if mode == "height_gradient_guided":
            notes.append("Segmentation: height-gradient-guided (block borders follow the height steps of the design)")
        if mode == "phase_quality_guided":
            route = r["route_info"]
            notes.append("Segmentation: phase-quality-guided (minimum-cost seams on the height-gradient map)")
            if route:
                notes.append(f"Writing order: {route['solver']}" + (" (proven optimal open path)" if route["optimal"] else " (improved, not proven optimal)")
                             + f", {route['length_um'] / 1000:.1f} mm of stage travel")
        if seg is not None:
            notes.append(f"Height-based segmentation: fields follow the structure at z = {seg:g} um; below it slicing "
                         f"{rec.base_slicing_distance or rec.slicing_distance:g} um, laser power x{rec.base_power_multiplier:g}")
        write_job_file(job_name, data_name, rec, notes)
        files += [job_name, data_name, files_dir]
        if js.method == "adaptive" and js.export_sliced_png:
            files += export_sliced_pngs(out_dir, name, rec, js, H, in_path, plan, ctx)
        if mode in LABEL_MODES and plan[2] is not None:
            lab_png = out_dir / f"{name}_segmentation_labels.png"                    # block id of every pixel (16 bit)
            gs.save_label_image(plan[2], lab_png)
            files.append(lab_png)
            if js.export_sliced_png:
                files.append(export_comparison_png(out_dir, name, rec, js, H, ctx, five=mode == "phase_quality_guided",
                                                   phase_plan=(plan[0], plan[1], plan[2], dict(r["route_info"]))
                                                   if mode == "phase_quality_guided" else None))
                if mode == "phase_quality_guided":
                    files.append(export_path_png(out_dir, name, rec, js, H, plan))
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    hours = float((np.asarray(total) / rec.scan_speed).sum() / 3600.0) if total is not None else None
    return {"name": name, "output_dir": str(out_dir), "method": js.method, "fields": n_fields,
            "files": [str(f) for f in files], "print_hours": hours, "seconds": time.time() - t0,
            "intermediate_dir": str(work_dir) if js.keep_intermediate else None,
            "route": dict(r["route_info"]) or None,
            "interface_calls": 0 if rec.interface_pos < 0 else (1 if js.global_interface_found else n_fields)}
