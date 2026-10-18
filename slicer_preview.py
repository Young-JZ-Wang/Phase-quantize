#!/usr/bin/env python3
"""
slicer_preview.py
-----------------
Shows how a slicing method divides a structure into writing fields, before any job file is written.

    footprint  =  where there is material, seen from above (a design's height map, or an STL's projection)
    layout     =  field centres and writing order, from the same code the job uses (job_generator.height_map_layout)
    render     =  the structure with the field borders, the overlap bands, the centres and the writing path

No UI code in here; the Job File Generator tab only displays the image. Large structures are reduced to at most
`max_side` pixels per side, which is also noted in the caption (the job itself works at full resolution).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

import gradient_segmentation as gs
import phase_quality as pqg
import route_solvers as rs
import height_map_stl as hms
import job_generator as jg
from recipes import Recipe


@dataclass
class Footprint:
    T: np.ndarray                    # (rows, cols) uint8, 1 = material; row 0 = smallest y
    bg: np.ndarray                   # (rows, cols) uint8 grey image of the structure, same orientation as T
    X: np.ndarray                    # pixel-centre coordinates in um, same shape as T
    Y: np.ndarray
    pitch: float                     # um per pixel of T
    size_um: tuple                   # (width, height) of the structure in um
    reduced: int = 1                 # reduction factor against the full-resolution input
    kind: str = "design"             # "design" (height map) or "stl"
    stl_origin: tuple = (0.0, 0.0)   # lower-left corner of an STL's bounding box (um)
    T_seg: Optional[np.ndarray] = None   # cross-section at the segmentation height (height-based adaptive slicing)
    H: Optional[np.ndarray] = None       # heights (um, base included) on the same grid as T (height-gradient-guided slicing)
    source: Optional[str] = None     # STL path
    stl_scale: float = 1.0


@dataclass
class Layout:
    centres: np.ndarray              # (n, 2) um, y up
    order: np.ndarray                # writing order (indices into centres)
    method: str
    footprint: Footprint
    field_size: float
    radius: float
    segment_height: Optional[float] = None   # height-based adaptive slicing: z of the cross-section the fields follow
    labels: Optional[np.ndarray] = None      # block id of every pixel (height-gradient-guided): the blocks are not cells
    segmentation: str = "footprint_edges"
    route: Optional[dict] = None             # writing-order metadata (solver, optimal, length_um, detail) when it was optimised


# --------------------------------------------------------------------------
# Footprints
# --------------------------------------------------------------------------
def _block_max(a, f):
    """Reduce a 2-D array by the integer factor f with max pooling (padding with zeros)."""
    if f == 1:
        return a
    rows, cols = a.shape
    pr, pc = (-rows) % f, (-cols) % f
    a = np.pad(a, ((0, pr), (0, pc)))
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).max(axis=(1, 3))


def _grid(rows, cols, pitch, origin=None):
    if origin is None:                                             # centred on 0, like the job
        x = (np.arange(cols) - (cols - 1) / 2.0) * pitch
        y = (np.arange(rows) - (rows - 1) / 2.0) * pitch
    else:
        x = origin[0] + (np.arange(cols) + 0.5) * pitch
        y = origin[1] + (np.arange(rows) + 0.5) * pitch
    return np.meshgrid(x, y)


def design_footprint(heights_um, pixel_size_um, base_um=0.0, image_top_is_plus_y=True, max_side=800, min_side=0,
                     segment_height_um=None) -> Footprint:
    """Footprint of a height map (rows x cols, um). Material wherever height + base > 0. A small map is repeated up to
    at least `min_side` pixels (smooth borders for exported images); 0 keeps its own grid."""
    H = np.asarray(heights_um, dtype=np.float64)
    if H.ndim != 2 or not pixel_size_um or pixel_size_um <= 0:
        raise ValueError("a height map and a pixel size are needed for the preview")
    if image_top_is_plus_y:                                        # y must grow with the row, as in the job
        H = np.flipud(H)
    rows, cols = H.shape
    f = max(1, int(np.ceil(max(rows, cols) / float(max_side))))
    T = _block_max(((H + (base_um or 0.0)) > 0).astype(np.uint8), f)
    T_seg = None
    if segment_height_um is not None:                              # structure above the segmentation height
        T_seg = _block_max(((H + (base_um or 0.0)) > segment_height_um + 1e-6).astype(np.uint8), f)
    Hg = np.pad(H + (base_um or 0.0), ((0, (-rows) % f), (0, (-cols) % f)), mode="edge")[::f, ::f] if f > 1 else H + (base_um or 0.0)
    peak = H.max() if H.max() > 0 else 1.0
    bg = np.pad(H, ((0, (-rows) % f), (0, (-cols) % f)), mode="edge")[::f, ::f] if f > 1 else H
    bg = np.round(bg / peak * 255).astype(np.uint8)
    pitch = pixel_size_um * f
    u = int(np.ceil(min_side / max(T.shape))) if min_side and max(T.shape) < min_side else 1
    if u > 1:                                                      # same structure on a finer grid
        T, bg, pitch = np.repeat(np.repeat(T, u, 0), u, 1), np.repeat(np.repeat(bg, u, 0), u, 1), pitch / u
        if T_seg is not None:
            T_seg = np.repeat(np.repeat(T_seg, u, 0), u, 1)
        Hg = np.repeat(np.repeat(Hg, u, 0), u, 1)
    X, Y = _grid(*T.shape, pitch)
    return Footprint(T=T, bg=bg, X=X, Y=Y, pitch=pitch, size_um=(cols * pixel_size_um, rows * pixel_size_um), reduced=f,
                     T_seg=T_seg, H=Hg)


def _rasterize(polys, shape):
    """Union of the triangles (N, 3, 2), given in pixel coordinates, as a uint8 image. Each triangle is filled on its
    own: one fillPoly call over all of them would use the even-odd rule, and a solid's top and bottom faces (same
    projection) would cancel each other out."""
    p = np.asarray(polys, dtype=np.float64)
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    p = p[area > 1e-6]                                             # vertical walls project to lines: nothing to fill
    try:
        import cv2
        img = np.zeros(shape, np.uint8)
        pts = np.round(p).astype(np.int32)
        for t in pts:
            cv2.fillConvexPoly(img, t, 1)
        return img
    except ImportError:
        im = Image.new("L", (shape[1], shape[0]), 0)
        d = ImageDraw.Draw(im)
        for t in p:
            d.polygon([tuple(v) for v in t], fill=1)
        return np.asarray(im, np.uint8)


def stl_footprint(path, stl_scale=1.0, max_side=800) -> Footprint:
    """Footprint of an STL: its projection onto the xy plane."""
    tri = hms.read_stl_triangles(path) * stl_scale
    lo, hi = tri.reshape(-1, 3).min(axis=0), tri.reshape(-1, 3).max(axis=0)
    w, h = hi[0] - lo[0], hi[1] - lo[1]
    if w <= 0 or h <= 0:
        raise ValueError("the STL has no extent in x or y")
    pitch = max(w, h) / float(max_side)
    cols, rows = int(np.ceil(w / pitch)) + 1, int(np.ceil(h / pitch)) + 1
    xy = (tri[..., :2] - lo[:2]) / pitch
    T = _rasterize(xy, (rows, cols))
    X, Y = _grid(rows, cols, pitch, origin=(lo[0], lo[1]))
    return Footprint(T=T, bg=(T * 90).astype(np.uint8), X=X, Y=Y, pitch=pitch, size_um=(w, h), reduced=1, kind="stl",
                     stl_origin=(float(lo[0]), float(lo[1])), source=str(path), stl_scale=stl_scale)


def stl_cross_section(fp: Footprint, z_um):
    """Cross-section of the STL at height z (um above its lowest point) on the footprint grid (needs TipSlicer)."""
    ts = jg.tipslicer()
    tri = hms.read_stl_triangles(fp.source) * fp.stl_scale
    tri[..., 2] -= tri[..., 2].min()                                # like TipSlicer.loadStl (z starts at 0)
    zmin, zmax = tri[..., 2].min(axis=1), tri[..., 2].max(axis=1)
    return (np.asarray(ts.singleSliceStl(tri, zmin, zmax, z_um, fp.X, fp.Y)) > 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _square_tiling(fp: Footprint, field_size):
    """STL, standard method: TipSlicer's regular square tiling (cells of field_size, grid from the corner)."""
    x0 = np.round(fp.stl_origin[0] - 5.0)
    y0 = np.round(fp.stl_origin[1] - 5.0)
    xs = np.arange(x0, fp.X.max() + 5.0, field_size)
    ys = np.arange(y0, fp.Y.max() + 5.0, field_size)
    cands = [(x, y) for y in ys for x in xs]
    keep = [c for c in cands if (fp.T * jg.in_field(c[0], c[1], fp.X, fp.Y, field_size / 2.0)).sum() > 0]
    return np.array(keep)


def compute_layout(fp: Footprint, method, recipe: Recipe, ctx=None, segment_height_um=None,
                   segmentation_mode=None, route_solvers=rs.ROUTE_AUTO, route_time_limit=rs.DEFAULT_TIME_LIMIT) -> Layout:
    """Field centres and writing order of `method` ("standard" / "adaptive") for the footprint. With a segmentation
    height the adaptive fields follow the structure at that height (height-based slicing); with segmentation_mode
    "height_gradient_guided" the block borders are attracted to every height step. route_solvers chooses the writing-order
    optimiser of every adaptive mode (route_solvers.ROUTE_AUTO: pyconcorde if installed, else 2-opt; ROUTE_FAST: 2-opt)."""
    if method not in jg.METHODS:
        raise ValueError(f"slicing method must be one of {jg.METHODS}")
    ctx = ctx or jg._Ctx()
    if method == "adaptive" and segmentation_mode in jg.LABEL_MODES:
        if fp.H is None:
            raise ValueError(f"{segmentation_mode} segmentation needs a height map, not an STL")
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nanoslicer_preview_") as tmp:
            if segmentation_mode == "phase_quality_guided":
                route = {}
                centres, order, labels = jg.phase_quality_layout(jg.tipslicer(), fp.H, fp.T, fp.X, fp.Y, fp.pitch,
                                                                 recipe.adaptive_radius, tmp, ctx, route_solvers,
                                                                 route_time_limit, route)
                return Layout(np.asarray(centres, float), np.asarray(order), method, fp, recipe.field_size,
                              recipe.adaptive_radius, None, labels, segmentation_mode, route)
            else:
                route = {}
                centres, order, labels = jg.gradient_layout(jg.tipslicer(), fp.H, fp.T, fp.X, fp.Y, fp.pitch,
                                                            recipe.adaptive_radius, tmp, ctx, None, route_solvers,
                                                            route_time_limit, route)
        return Layout(np.asarray(centres, float), np.asarray(order), method, fp, recipe.field_size, recipe.adaptive_radius,
                      None, labels, segmentation_mode, route)
    route = {}
    if method == "standard" and fp.kind == "stl":
        centres = _square_tiling(fp, recipe.field_size)        # the job orders STL fields with the same solver
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nanoslicer_preview_") as tmp:
            order = jg.route_order(jg.tipslicer(), centres, tmp, ctx, route_solvers, route_time_limit, route) if len(centres) > 2 \
                else np.arange(len(centres))
    else:
        ts = jg.tipslicer() if method == "adaptive" else None
        seg = segment_height_um if method == "adaptive" else None
        T_seg = None
        if seg is not None:
            if fp.kind == "stl":
                T_seg = fp.T_seg if fp.T_seg is not None else stl_cross_section(fp, seg)
            elif fp.T_seg is None:
                raise ValueError("build the footprint with segment_height_um to preview height-based slicing")
            else:
                T_seg = fp.T_seg
        import tempfile
        with tempfile.TemporaryDirectory(prefix="nanoslicer_preview_") as tmp:
            centres, order = jg.height_map_layout(ts, fp.T, fp.X, fp.Y, fp.pitch, method, recipe.field_size,
                                                  recipe.adaptive_radius, tmp, ctx, size_um=fp.size_um, T_seg=T_seg,
                                                  route_solvers=route_solvers, route_time_limit=route_time_limit,
                                                  route_info=route)
    if len(centres) == 0:
        raise ValueError("the structure has no material to write")
    return Layout(np.asarray(centres, float), np.asarray(order), method, fp, recipe.field_size, recipe.adaptive_radius,
                  segment_height_um if method == "adaptive" else None, route=route or None)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
BORDER = (255, 70, 70)
OVERLAP = (255, 170, 0)
CENTRE = (255, 235, 59)
PATH = (0, 200, 220)
FIRST = (60, 220, 90)
LAST = (230, 60, 230)


def field_labels(layout: Layout):
    """(rows, cols) int array: index of the field that owns each pixel, -1 outside the structure."""
    from scipy.spatial import cKDTree
    fp = layout.footprint
    if layout.labels is not None:                                   # blocks that are not nearest-centre cells
        lab = layout.labels if layout.labels.shape == fp.T.shape else gs.resize_labels(layout.labels, fp.T.shape)
        return np.where(fp.T > 0, lab, -1)
    tree = cKDTree(layout.centres)
    pts = np.column_stack((fp.X.ravel(), fp.Y.ravel()))
    # standard fields are squares (nearest centre in the max-norm), adaptive fields are Voronoi cells (Euclidean)
    _, idx = tree.query(pts, p=np.inf if layout.method == "standard" else 2)
    lab = idx.reshape(fp.T.shape)
    return np.where(fp.T > 0, lab, -1)


BORDER_ALPHA = 0.7                    # borders are blended over the image, not painted on it
OUTLINE_ALPHA = 0.3                   # the edge of the structure itself, fainter than the field borders


def _border_segments(layout: Layout):
    """Lines along which the fields touch, in pixel coordinates of the y-up footprint grid: the grid lines of a standard
    layout, the finite Voronoi ridges of an adaptive one."""
    fp = layout.footprint
    rows, cols = fp.T.shape
    to_px = lambda x, y: ((x - fp.X[0, 0]) / fp.pitch, (y - fp.Y[0, 0]) / fp.pitch)
    c = layout.centres
    segs = []
    if layout.method == "standard":
        half = layout.field_size / 2.0
        for x in np.unique(np.round(np.concatenate((c[:, 0] - half, c[:, 0] + half)), 6)):
            segs.append((to_px(x, fp.Y[0, 0] - fp.pitch), to_px(x, fp.Y[-1, 0] + fp.pitch)))
        for y in np.unique(np.round(np.concatenate((c[:, 1] - half, c[:, 1] + half)), 6)):
            segs.append((to_px(fp.X[0, 0] - fp.pitch, y), to_px(fp.X[0, -1] + fp.pitch, y)))
        return segs
    from scipy.spatial import Voronoi
    span = 50.0 * max(fp.size_um)
    mid = c.mean(axis=0)
    far = mid + span * np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]])
    vor = Voronoi(np.vstack((c, far)))                           # far points make every real cell finite
    n = len(c)
    for (i, j), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        if i < n and j < n and -1 not in ridge:
            a, b = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            segs.append((to_px(*a), to_px(*b)))
    return segs


def _dash_mask(shape, segments, dash, gap):
    """Pixels of the segments that fall on a dash (dash and gap lengths in pixels, counted from each segment's start)."""
    m = np.zeros(shape, bool)
    for (x0, y0), (x1, y1) in segments:
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 1e-9:
            continue
        t = np.linspace(0.0, 1.0, int(length * 2) + 2)
        on = ((t * length) % (dash + gap)) < dash
        xs = np.round(x0 + (x1 - x0) * t[on]).astype(int)
        ys = np.round(y0 + (y1 - y0) * t[on]).astype(int)
        ok = (xs >= 0) & (xs < shape[1]) & (ys >= 0) & (ys < shape[0])
        m[ys[ok], xs[ok]] = True
    return m


def render(layout: Layout, overlap_um=0.0, dashed=True, border_alpha=BORDER_ALPHA, path=True) -> Image.Image:
    """The structure with field borders (dashed, semi-transparent red), overlap bands (orange), centres (yellow),
    writing path (cyan), first field (green) and last field (magenta). Top of the image is +y."""
    import scipy.ndimage as ndi
    fp = layout.footprint
    lab = field_labels(layout)
    inside = lab >= 0
    shared = np.zeros(lab.shape, bool)                             # pixels at a border between two fields
    outline = np.zeros(lab.shape, bool)                            # pixels at the edge of the structure
    for axis in (0, 1):
        a = np.moveaxis(lab, axis, 0)
        sh = np.zeros_like(a, bool)
        ol = np.zeros_like(a, bool)
        d = a[1:] != a[:-1]
        both = (a[1:] >= 0) & (a[:-1] >= 0)
        sh[1:] |= d & both
        sh[:-1] |= d & both
        ol[1:] |= d & ~both
        ol[:-1] |= d & ~both
        shared |= np.moveaxis(sh, 0, axis)
        outline |= np.moveaxis(ol, 0, axis)
    shared &= inside
    outline &= inside

    rgb = np.repeat((fp.bg.astype(np.float64) * 0.55 + 25)[..., None], 3, axis=2)
    rgb[~inside] = (20, 20, 20)
    big = max(lab.shape)
    ov_px = int(round(overlap_um / fp.pitch)) if overlap_um and overlap_um > 0 else 0
    if ov_px >= 1:
        band = ndi.binary_dilation(shared, iterations=ov_px) & inside
        rgb[band] = rgb[band] * 0.4 + np.array(OVERLAP) * 0.6
    thick = max(1, int(round(big / 450.0)))                        # line width and dash length follow the image size
    border = shared
    if dashed:
        dash = max(4, int(round(big / 70.0)))
        if layout.labels is not None:                              # curved borders: a dash pattern that works in any direction
            yy, xx = np.indices(lab.shape)
            border = shared & (((xx + 2 * yy) // max(2, dash // 2)) % 2 == 0)
        else:
            near = ndi.binary_dilation(shared, iterations=2)       # keep only dashes that lie on a real field border
            border = _dash_mask(lab.shape, _border_segments(layout), dash, max(3, dash // 2)) & near
    if thick > 1:
        border = ndi.binary_dilation(border, iterations=thick - 1)
    border &= inside
    rgb[outline] = rgb[outline] * (1.0 - OUTLINE_ALPHA) + np.array(BORDER) * OUTLINE_ALPHA
    rgb[border] = rgb[border] * (1.0 - border_alpha) + np.array(BORDER) * border_alpha

    img = Image.fromarray(np.flipud(rgb).astype(np.uint8))          # top = +y
    rows, cols = lab.shape
    d = ImageDraw.Draw(img)
    px = (layout.centres[:, 0] - fp.X[0, 0]) / fp.pitch
    py = rows - 1 - (layout.centres[:, 1] - fp.Y[0, 0]) / fp.pitch
    route = [(px[i], py[i]) for i in layout.order]
    if path and len(route) > 1:
        d.line(route, fill=PATH, width=1)
    r = max(1, int(min(rows, cols) / 330))
    for x, y in zip(px, py):
        d.rectangle([x - r, y - r, x + r, y + r], fill=CENTRE)
    for i, colour in ((layout.order[0], FIRST), (layout.order[-1], LAST)):
        x, y = px[i], py[i]
        d.rectangle([x - 2 * r, y - 2 * r, x + 2 * r, y + 2 * r], fill=colour)
    return img


def caption(layout: Layout, overlap_um=0.0) -> str:
    fp = layout.footprint
    if layout.method == "standard":
        what = f"standard: {len(layout.centres)} rectangular fields of {layout.field_size:g} µm"
    else:
        what = f"adaptive: {len(layout.centres)} fields, write radius {layout.radius:g} µm"
    parts = [what, f"{fp.size_um[0]:g} × {fp.size_um[1]:g} µm"]
    if layout.segmentation == "height_gradient_guided":
        parts.insert(1, "height-gradient-guided, block borders follow the height steps")
    if layout.segmentation == "phase_quality_guided":
        parts.insert(1, "phase-quality-guided, minimum-cost seams on the height steps")
        if layout.route:
            parts.append("route: " + layout.route["solver"] + (" (optimal)" if layout.route["optimal"] else ""))
    if layout.segment_height is not None:
        parts.insert(1, f"height-based, fields follow the structure at z = {layout.segment_height:g} µm")
    if overlap_um:
        parts.append(f"overlap {overlap_um:g} µm")
    if fp.reduced > 1:
        parts.append(f"preview at 1/{fp.reduced} resolution")
    if fp.kind == "stl":
        parts.append("projection of the STL; taller objects are also split in z")
    return "  ·  ".join(parts)


def render_path(layout: Layout) -> Image.Image:
    """The writing order alone: the structure dimmed, the stage path as a line that runs from green (first field) to
    magenta (last field), the field centres as dots."""
    fp = layout.footprint
    rows, cols = fp.T.shape
    rgb = np.repeat((fp.bg.astype(np.float64) * 0.35 + 18)[..., None], 3, axis=2)
    img = Image.fromarray(np.flipud(rgb).astype(np.uint8))
    d = ImageDraw.Draw(img)
    px = (layout.centres[:, 0] - fp.X[0, 0]) / fp.pitch
    py = rows - 1 - (layout.centres[:, 1] - fp.Y[0, 0]) / fp.pitch
    n = len(layout.order)
    width = max(1, int(max(rows, cols) / 450))
    for j in range(n - 1):
        t = j / max(n - 2, 1)
        colour = tuple(int(a + (b - a) * t) for a, b in zip(FIRST, LAST))
        i0, i1 = layout.order[j], layout.order[j + 1]
        d.line([(px[i0], py[i0]), (px[i1], py[i1])], fill=colour, width=width)
    r = max(1, int(min(rows, cols) / 330))
    for x, y in zip(px, py):
        d.rectangle([x - r, y - r, x + r, y + r], fill=CENTRE)
    return img


# --------------------------------------------------------------------------
# Comparison of the slicing modes
# --------------------------------------------------------------------------
MODE_TITLES = {"standard": "Standard square fields", "footprint": "Adaptive, footprint edges",
               "threshold": "Adaptive, threshold at z = {z:g} um", "gradient": "Adaptive, height-gradient-guided",
               "phase": "Adaptive, phase-quality-guided"}


def layout_metrics(layouts: dict) -> dict:
    """Seam metrics (phase_quality.seam_metrics) of every layout, on the same grid and with the same seam cost, so the
    modes compare. Needs layouts built from a design (the footprint carries the heights)."""
    first = next(iter(layouts.values()))
    fp = first.footprint
    g = gs.height_gradient(fp.H, fp.pitch)
    out = {}
    for k, lay in layouts.items():
        m = pqg.seam_metrics(field_labels(lay), g, fp.pitch, lay.centres, lay.order)
        m["route_solver"] = lay.route["solver"] if lay.route else None            # set for the modes whose order was optimised here
        m["route_optimal"] = bool(lay.route["optimal"]) if lay.route else None
        out[k] = m
    return out


def _metrics_panel(metrics: dict, size, max_side):
    """A table of the metrics as an image of the given size."""
    from PIL import ImageFont
    w, h = size
    try:
        font = ImageFont.load_default(size=max(10, w // 40))
    except (ImportError, TypeError):
        font = None
    short = {"standard": "standard", "footprint": "footprint", "threshold": "threshold", "gradient": "gradient", "phase": "phase"}
    head = ("mode", "fields", "seam mm", "cost", "on step", "islands", "travel")
    rows = [head] + [(short.get(k, k), str(m["blocks"]), f"{m['seam_length_um'] / 1000:.1f}", f"{m['seam_cost_mean']:.3f}",
                      f"{m['seam_on_step'] * 100:.0f}%", str(m["extra_pieces"]), f"{m.get('travel_um', 0) / 1000:.1f}")
                     for k, m in metrics.items()]
    img = Image.new("RGB", size, (24, 24, 24))
    d = ImageDraw.Draw(img)
    fracs = (0.0, 0.18, 0.31, 0.47, 0.59, 0.73, 0.85)                     # column starts; every cell fits its column
    line = max(14, h // (len(rows) + 5))
    y = line // 2
    for r, row in enumerate(rows):
        for x, cell in zip(fracs, row):
            d.text((int(w * x) + 4, y), cell, fill=(160, 200, 255) if r == 0 else (230, 230, 230), font=font)
        y += line
    routes = [f"route ({short.get(k, k)}): {m['route_solver']}" + (", optimal" if m.get("route_optimal") else "")
              for k, m in metrics.items() if m.get("route_solver")]
    note = (("\n".join(routes) + "\n" if routes else "") +
            "cost: mean seam cost exp(-k gradH), lower = seams on the steps\n"
            "on step: share of the seam on a strong height step\n"
            "islands: extra pieces of a block   travel: stage travel in mm")
    d.text((4, y + line // 2), note, fill=(150, 150, 150), font=font)
    return img


def comparison_image(heights_um, pixel_size_um, base_um, image_top_is_plus_y, recipe: Recipe, segment_height_um=None,
                     max_side=700, ctx=None, modes=None, route_solvers=rs.ROUTE_AUTO, route_time_limit=rs.DEFAULT_TIME_LIMIT,
                     phase_plan=None):
    """The same structure divided several ways, side by side. By default three: standard square fields, the
    threshold-based adaptive slicing (edge voting on the cross-section at `segment_height_um`, default just above the base)
    and the height-gradient-guided slicing. With modes=("standard", "footprint", "threshold", "gradient", "phase") all five,
    on two rows with a metrics table as the sixth cell. Returns (PIL image, {mode: Layout}).
    The phase-quality writing order comes from route_solvers, or, with phase_plan = (centres_um, order, labels, route_info),
    from a job that already solved it (nothing is solved twice and the route shown is the one that is written)."""
    modes = tuple(modes) if modes else ("standard", "threshold", "gradient")
    z = base_um if segment_height_um is None else segment_height_um
    fp = design_footprint(heights_um, pixel_size_um, base_um, image_top_is_plus_y, max_side=max_side, segment_height_um=z)
    opt = {"route_solvers": route_solvers, "route_time_limit": route_time_limit}
    build = {"standard": lambda: compute_layout(fp, "standard", recipe, ctx),
             "footprint": lambda: compute_layout(fp, "adaptive", recipe, ctx, **opt),
             "threshold": lambda: compute_layout(fp, "adaptive", recipe, ctx, segment_height_um=z, **opt),
             "gradient": lambda: compute_layout(fp, "adaptive", recipe, ctx, segmentation_mode="height_gradient_guided", **opt),
             "phase": lambda: (Layout(np.asarray(phase_plan[0], float), np.asarray(phase_plan[1]), "adaptive", fp, recipe.field_size,
                                      recipe.adaptive_radius, None, phase_plan[2], "phase_quality_guided", phase_plan[3])
                               if phase_plan else
                               compute_layout(fp, "adaptive", recipe, ctx, segmentation_mode="phase_quality_guided",
                                              route_solvers=route_solvers, route_time_limit=route_time_limit))}
    unknown = [m for m in modes if m not in build]
    if unknown:
        raise ValueError(f"unknown comparison mode(s) {unknown}; choose from {list(build)}")
    layouts = {m: build[m]() for m in modes}
    titles = {m: MODE_TITLES[m].format(z=z) for m in modes}
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=max(12, max_side // 40))
    except (ImportError, TypeError):
        font = None
    panels = [render(layouts[m], recipe.overlap) for m in modes]
    bar, gap = max(20, max_side // 22), 8
    w, h = panels[0].size
    ncols = 3 if len(modes) > 3 else len(modes)
    nrows = 1 if len(modes) <= 3 else 2
    out = Image.new("RGB", (ncols * w + (ncols - 1) * gap, nrows * (h + bar) + (nrows - 1) * gap), (24, 24, 24))
    d = ImageDraw.Draw(out)
    for i, m in enumerate(modes):
        x, y = (i % ncols) * (w + gap), (i // ncols) * (h + bar + gap)
        out.paste(panels[i], (x, y + bar))
        d.text((x + 6, y + 3), f"{titles[m]}: {len(layouts[m].centres)} fields", fill=(230, 230, 230), font=font)
    if len(modes) > 3:
        x, y = (len(modes) % ncols) * (w + gap), (len(modes) // ncols) * (h + bar + gap)
        d.text((x + 6, y + 3), "Seam metrics", fill=(230, 230, 230), font=font)
        out.paste(_metrics_panel(layout_metrics(layouts), (w, h), max_side), (x, y + bar))
    return out, layouts
