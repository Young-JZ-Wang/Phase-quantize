#!/usr/bin/env python3
"""
gradient_segmentation.py
------------------------
Height-gradient-guided segmentation: divide a quantized height map into writing fields so that the stitching seams lie
along the existing height steps and not through smooth areas.

Unlike the other adaptive modes this one reads the full height map H(x, y), not a binary silhouette:

    1. gradient   gradH = |grad H| of the actual height values (every quantized step is visible, not only the outline of
                  one thresholded level), normalised to [0, 1]
    2. centres    writing centres that cover the whole aperture within the write radius (a hexagonal lattice whose
                  phase puts the seeds in smooth places, so the steps fall between centres)
    3. blocks     every pixel goes to the centre it reaches at the lowest travel cost, where crossing a pixel costs
                  1 + ridge_cost * gradH. Floods stop at the steps, so the borders between blocks settle on them; in
                  smooth areas the cost is uniform and the borders are the usual nearest-centre ones.
                  A centre only competes inside its own square writing frame (|dx|, |dy| <= radius), so no block can
                  leave its frame; the disc cover guarantees every pixel has at least one centre whose frame holds it.

Heights and the optical design are never touched; only the block masks are.
"""
import numpy as np
from PIL import Image

DEFAULT_RIDGE_COST = 25.0      # cost, in pixels of travel, of crossing a pixel of full (normalised) gradient
SMOOTH_WEIGHT = 0.2            # weight of the "centre sits in a smooth area" term when choosing centres


def height_gradient(H, pixel_size_um):
    """|grad H| of the height values (um per um), normalised to [0, 1] by the 99th percentile of the non-zero
    gradients (so one very steep wrap does not flatten all the other steps). An all-flat map gives zeros."""
    H = np.asarray(H, dtype=np.float64)
    gy, gx = np.gradient(H, pixel_size_um)
    g = np.hypot(gx, gy)
    nz = g[g > 1e-12]
    if nz.size == 0:
        return np.zeros_like(g)
    return np.clip(g / np.percentile(nz, 99), 0.0, 1.0)


def _disc(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.float64)


def cover_centres(T, g, radius_px):
    """Pixel centres (row, col) of discs of `radius_px` that together cover every pixel of the aperture T. The next centre
    is the uncovered pixel whose disc covers the most still-uncovered aperture, with a small preference for smooth
    surroundings (low gradient)."""
    from scipy.signal import fftconvolve
    if radius_px < 3:
        raise ValueError("the write radius is under 3 pixels: use a smaller pixel size or a larger radius")
    P = 2 * radius_px + 1
    disc = _disc(radius_px)
    small = _disc(max(1, radius_px // 3))
    In = np.pad((np.asarray(T) > 0).astype(np.float64), P)                       # padded: windows never need clipping
    smooth = 1.0 - np.clip(fftconvolve(np.pad(np.asarray(g, float), P), small, "same") / small.sum(), 0, 1)
    cover = fftconvolve(In, disc, "same") / disc.sum()                            # uncovered aperture inside each disc
    centres = []
    r = radius_px
    while In.any():
        score = (cover + SMOOTH_WEIGHT * smooth) * (In > 0)
        cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
        centres.append((cy - P, cx - P))
        sl = (slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
        removed = In[sl] * disc                                                   # pixels this disc newly covers
        In[sl] -= removed
        sl2 = (slice(cy - 2 * r, cy + 2 * r + 1), slice(cx - 2 * r, cx + 2 * r + 1))
        cover[sl2] -= fftconvolve(removed, disc, "full") / disc.sum()             # their share in every disc around
    return np.array(centres, dtype=int)


def lattice_centres(T, g, radius_px, shrink=0.92, phases=5, nudge_frac=0.08):
    """Writing centres (row, col) that cover every pixel of the aperture T within `radius_px`: a hexagonal lattice (the
    most economical covering) whose phase is the one that puts the seeds in the smoothest places, each seed then moved
    to the smoothest pixel nearby. Whatever the lattice leaves uncovered (odd-shaped apertures) is covered by the greedy
    cover_centres, so the guarantee holds for any aperture."""
    import scipy.ndimage as ndi
    T = np.asarray(T) > 0
    rows, cols = T.shape
    re_ = shrink * radius_px
    dx, dy = np.sqrt(3.0) * re_, 1.5 * re_
    _, (iy, ix) = ndi.distance_transform_edt(~T, return_indices=True)               # nearest aperture pixel
    ys = np.arange(-dy, rows + dy, dy)

    def seeds(fy, fx):
        pts = []
        for j, y in enumerate(ys + fy):
            xs = np.arange(-dx, cols + dx, dx) + fx + (dx / 2 if j % 2 else 0.0)
            pts.append(np.column_stack((np.full(xs.shape, y), xs)))
        p = np.rint(np.vstack(pts)).astype(int)
        p = np.column_stack((np.clip(p[:, 0], 0, rows - 1), np.clip(p[:, 1], 0, cols - 1)))
        p = np.column_stack((iy[p[:, 0], p[:, 1]], ix[p[:, 0], p[:, 1]]))              # snap onto the aperture
        return np.unique(p, axis=0)

    options = [seeds(fy, fx) for fy in np.linspace(0, dy, phases, endpoint=False) for fx in np.linspace(0, dx, phases, endpoint=False)]
    fewest = min(len(o) for o in options)
    pick = min((o for o in options if len(o) <= fewest * 1.02), key=lambda o: g[o[:, 0], o[:, 1]].sum())   # smoothest phase
    r = max(1, int(round(nudge_frac * radius_px)))
    out = []
    for cy, cx in pick:                                                             # nudge to the smoothest pixel nearby
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, rows), max(cx - r, 0), min(cx + r + 1, cols)
        w = np.where(T[y0:y1, x0:x1], g[y0:y1, x0:x1], np.inf)
        k = np.unravel_index(int(np.argmin(w)), w.shape)
        out.append((y0 + k[0], x0 + k[1]))
    out = np.unique(np.array(out, dtype=int), axis=0)
    seed_img = np.zeros(T.shape, bool)
    seed_img[out[:, 0], out[:, 1]] = True
    uncovered = T & (ndi.distance_transform_edt(~seed_img) > radius_px)
    if uncovered.any():
        out = np.vstack((out, cover_centres(uncovered, g, radius_px)))
    return _prune(T, out, radius_px)


def _prune(T, centres, radius_px):
    """Drop every centre whose disc covers nothing that another centre does not also cover (duplicated seeds at the
    edges); the coverage of the aperture is unchanged."""
    from scipy.signal import fftconvolve
    r = radius_px
    P = r + 1
    disc = _disc(r) > 0
    Tp = np.pad(np.asarray(T) > 0, P)
    seeds = np.zeros(Tp.shape)
    for cy, cx in centres:
        seeds[cy + P, cx + P] += 1
    count = np.rint(fftconvolve(seeds, disc.astype(float), "same")).astype(int)            # discs covering each pixel
    keep = np.ones(len(centres), bool)
    unique = [int(((count[cy + P - r:cy + P + r + 1, cx + P - r:cx + P + r + 1] == 1) & disc &
                   Tp[cy + P - r:cy + P + r + 1, cx + P - r:cx + P + r + 1]).sum()) for cy, cx in centres]
    for k in np.argsort(unique):                                                           # least useful first
        cy, cx = centres[k]
        sl = (slice(cy + P - r, cy + P + r + 1), slice(cx + P - r, cx + P + r + 1))
        if not ((count[sl] == 1) & disc & Tp[sl]).any():
            keep[k] = False
            count[sl] -= disc
    return centres[keep]


def assign_blocks(T, g, centres_px, radius_px, ridge_cost=DEFAULT_RIDGE_COST):
    """Label image (rows, cols) int32: index of the block of every aperture pixel, -1 outside the aperture."""
    from skimage.graph import MCP_Geometric
    rows, cols = T.shape
    cost = 1.0 + ridge_cost * np.asarray(g, float)
    cost[np.asarray(T) == 0] = np.inf
    best = np.full((rows, cols), np.inf)
    lab = np.full((rows, cols), -1, np.int32)
    L = int(radius_px)                                                             # the square writing frame
    for k, (cy, cx) in enumerate(centres_px):
        y0, y1, x0, x1 = max(cy - L, 0), min(cy + L + 1, rows), max(cx - L, 0), min(cx + L + 1, cols)
        sub = cost[y0:y1, x0:x1]
        costs, _ = MCP_Geometric(sub, fully_connected=True).find_costs([(cy - y0, cx - x0)])
        win_best, win_lab = best[y0:y1, x0:x1], lab[y0:y1, x0:x1]
        better = costs < win_best
        win_best[better] = costs[better]
        win_lab[better] = k
    left = (lab < 0) & (np.asarray(T) > 0)                                         # cut off by gaps in the aperture
    if left.any():
        from scipy.spatial import cKDTree
        _, near = cKDTree(np.asarray(centres_px, float)).query(np.argwhere(left).astype(float))
        lab[left] = near
    return lab


def compact(centres_px, labels):
    """Drop the centres that ended up with no pixel (their neighbours took all their area: such a field would be written
    empty) and renumber the labels 0..n-1. Returns (centres, labels)."""
    used = np.unique(labels[labels >= 0])
    if len(used) == len(centres_px):
        return centres_px, labels
    remap = np.full(len(centres_px), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return centres_px[used], np.where(labels >= 0, remap[np.maximum(labels, 0)], -1).astype(np.int32)


def segment(H, pixel_size_um, radius_um, aperture=None, ridge_cost=DEFAULT_RIDGE_COST):
    """Height-gradient-guided segmentation of the height map H (um, rows grow with y).

    Returns (centres_px, labels, gradient): the writing centres as (row, col), the block id of every pixel (-1 outside
    the aperture, default aperture = the whole map) and the normalised gradient that guided the borders."""
    H = np.asarray(H, dtype=np.float64)
    T = np.ones(H.shape, np.uint8) if aperture is None else (np.asarray(aperture) > 0).astype(np.uint8)
    if not T.any():
        raise ValueError("the aperture is empty")
    # half a pixel of margin: a fine writing-grid point is looked up by its nearest pixel, which may be up to half a pixel
    # away, and it must still lie inside the frame (|dx|, |dy| <= radius) of the block that pixel belongs to
    radius_px = int(np.floor(radius_um / pixel_size_um - 0.5))
    g = height_gradient(H, pixel_size_um)
    centres = lattice_centres(T, g, radius_px)
    centres, labels = compact(centres, assign_blocks(T, g, centres, radius_px, ridge_cost))
    return centres, labels, g


class LabelCells:
    """Block masks sampled from a label image, with the same .mask(idx, Xloc, Yloc) as the job's Cells."""

    def __init__(self, labels, x0, y0, pitch):
        self.labels, self.x0, self.y0, self.pitch = labels, x0, y0, pitch

    def mask(self, idx, Xloc, Yloc):
        rows, cols = self.labels.shape
        ix = np.round((Xloc - self.x0) / self.pitch).astype(int)
        iy = np.round((Yloc - self.y0) / self.pitch).astype(int)
        ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
        out = np.zeros(Xloc.shape, np.uint8)
        out[ok] = self.labels[iy[ok], ix[ok]] == idx
        return out


def block_mask(labels, idx):
    """Binary mask (bool, same grid as the label image) of block `idx`."""
    return np.asarray(labels) == idx


def save_label_image(labels, path):
    """16-bit PNG: block id + 1 (0 = outside the aperture); row 0 = the largest y, like an image of the design."""
    if labels.max() + 1 > 65535:
        raise ValueError("more than 65535 blocks")
    Image.fromarray(np.flipud(labels + 1).astype(np.uint16)).save(path)


def resize_labels(labels, shape):
    """Nearest-neighbour resize of a label image (for drawing it on a finer grid)."""
    rows = np.minimum((np.arange(shape[0]) * labels.shape[0]) // shape[0], labels.shape[0] - 1)
    cols = np.minimum((np.arange(shape[1]) * labels.shape[1]) // shape[1], labels.shape[1] - 1)
    return labels[np.ix_(rows, cols)]
