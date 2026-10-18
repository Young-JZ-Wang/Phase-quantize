#!/usr/bin/env python3
"""
phase_quality.py
----------------
Phase-quality-guided segmentation (segmentation_mode = "phase_quality_guided"), seam metrics and block ordering.

Where the stitching seams cause the least optical damage, for a phase mask / DOE
---------------------------------------------------------------------------------
A stitching seam is a place where two writing fields meet. For a phase element what matters at a seam is phase
continuity: a seam that runs through a smooth phase region splits one optical feature in two, and the small
misregistration of the fields shows up as a phase error in an area where the design had none. A seam that runs along an
existing phase step (a wrap, or any quantized height transition) puts the field change where the structure already has a
discontinuity, so the seam hides in it.

How this extends the two adaptive-stitching papers
--------------------------------------------------
* Adaptive stitching (Dehaeck, Scheid, Lambert 2018) chooses the writing centres by edge voting on a BINARY mask of
  the structure and takes the Voronoi cells of the centres as the fields. It sees only the outline of one silhouette,
  and the borders between fields are wherever the Voronoi construction puts them.
* Miniscope3D (2020) segments a microlens array by HEIGHT-FUNCTION OWNERSHIP: every xy point belongs to the microlens
  whose local height function is highest there, so the seams fall on the lens boundaries. That needs the design to be a
  set of separate local height functions; a wrapped, quantized DOE has none, so there is nothing to compare.
* Here the structure that matters is the internal phase-step pattern of ONE height map H(x, y). So the same goal
  (seams on the existing discontinuities) is reached with the FULL height-gradient map instead of a silhouette or
  an ownership rule:

      gradH     = |grad H| of the actual height values, normalised to [0, 1]   (every quantized level is visible)
      seam cost = eps + (1 - eps) * exp(-k * gradH)                           (cheap on a step, dear in smooth areas)

  and the fields are the minimum-cost partition under the hard rules of the printer:
      * every aperture pixel belongs to exactly one block,
      * a pixel can only belong to a block whose square writing frame (|dx|, |dy| <= radius) contains it,
      * a block is one connected piece (disconnected islands are merged into a neighbour).
  The paper's writing-order optimisation is kept: the centres are visited in a short (travelling-salesman) order,
  improved by 2-opt, because the stage travel between fields is what the order is for.

Algorithm
---------
1. writing centres  gradient_segmentation.lattice_centres (cover the aperture within the write radius, seeds in smooth places)
2. start            gradient_segmentation.assign_blocks (travel cost with the gradient as a barrier)
3. refine           graph cut: for every pair of neighbouring blocks the shared border is re-optimised as a
                    minimum s-t cut in a band around it (alpha-beta swap), exactly, with scipy's max-flow. Repeated for
                    a few sweeps. Each swap can only lower the total seam cost.
4. clean up         pieces that got cut off from their block are merged into the neighbour they touch most

The height values and the optical design are never touched; only the block masks (and the block order) change.
"""
import numpy as np
import scipy.ndimage as ndi

import gradient_segmentation as gs

SEAM_K = 4.0          # sharpness of the seam cost: exp(-k * gradH); a full step costs e^-4 = 2% of a smooth pixel edge
SEAM_EPS = 0.05       # floor of the seam cost, so a seam is never free (this keeps seams short and straight)
HIGH_GRADIENT = 0.5   # an edge counts as "on a phase step" when its normalised gradient is at least this
SCALE = 1000          # costs are scaled to integers for the max-flow
INF = 1 << 28


def seam_cost(g, k=SEAM_K, eps=SEAM_EPS):
    """Cost of cutting at pixels with normalised gradient g: ~1 in smooth areas, ~eps on strong steps."""
    return eps + (1.0 - eps) * np.exp(-k * np.asarray(g, dtype=np.float64))


def _edge_costs(g, k, eps):
    """Cost of a seam between horizontal neighbours (rows, cols-1) and vertical neighbours (rows-1, cols)."""
    c = seam_cost(g, k, eps)
    return 0.5 * (c[:, :-1] + c[:, 1:]), 0.5 * (c[:-1] + c[1:])


# --------------------------------------------------------------------------
# graph-cut refinement
# --------------------------------------------------------------------------
def _neighbour_pairs(lab):
    a = np.concatenate((lab[:, :-1][lab[:, :-1] != lab[:, 1:]], lab[:-1][lab[:-1] != lab[1:]]))
    b = np.concatenate((lab[:, 1:][lab[:, :-1] != lab[:, 1:]], lab[1:][lab[:-1] != lab[1:]]))
    ok = (a >= 0) & (b >= 0)
    if not ok.any():
        return []
    return [tuple(p) for p in np.unique(np.sort(np.column_stack((a[ok], b[ok])), axis=1), axis=0)]


def _swap(lab, a, b, centres, L, ch, cv, band):
    """Re-optimise the border between blocks a and b: returns True if some pixel changed block."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    rows, cols = lab.shape
    (ay, ax), (by, bx) = centres[a], centres[b]
    y0, y1 = max(min(ay, by) - L, 0), min(max(ay, by) + L + 1, rows)
    x0, x1 = max(min(ax, bx) - L, 0), min(max(ax, bx) + L + 1, cols)
    sub = lab[y0:y1, x0:x1]
    is_a, is_b = sub == a, sub == b
    region = is_a | is_b
    seam = (is_a & (ndi.binary_dilation(is_b, structure=ndi.generate_binary_structure(2, 1)))) | \
           (is_b & (ndi.binary_dilation(is_a, structure=ndi.generate_binary_structure(2, 1))))
    if not seam.any():
        return False
    free = region & ndi.binary_dilation(seam, iterations=band)
    m = int(free.sum())
    if m == 0:
        return False
    idx = np.full(sub.shape, -1, np.int64)
    idx[free] = np.arange(m)
    S, T = m, m + 1
    yy, xx = np.indices(sub.shape)
    ok_a = (np.abs(yy + y0 - ay) <= L) & (np.abs(xx + x0 - ax) <= L)                 # inside a's / b's writing frame
    ok_b = (np.abs(yy + y0 - by) <= L) & (np.abs(xx + x0 - bx) <= L)
    fa, fb = region & ~free & is_a, region & ~free & is_b                            # fixed pixels, terminals
    rs, cs, ws = [], [], []

    def add(u, v, w, both=True):
        rs.append(u); cs.append(v); ws.append(w)
        if both:
            rs.append(v); cs.append(u); ws.append(w)

    for w_full, sl_p, sl_q in ((ch[y0:y1, x0:x1 - 1] if x1 - 1 > x0 else None, (slice(None), slice(0, -1)), (slice(None), slice(1, None))),
                               (cv[y0:y1 - 1, x0:x1] if y1 - 1 > y0 else None, (slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        if w_full is None:
            continue
        w = np.maximum(1, np.rint(w_full * SCALE)).astype(np.int64)
        fp, fq = free[sl_p], free[sl_q]
        both = fp & fq
        add(idx[sl_p][both], idx[sl_q][both], w[both])                                # free - free
        for p_free, q_fixed_a, q_fixed_b, ip, wsel in ((fp, fa[sl_q], fb[sl_q], idx[sl_p], w),
                                                       (fq, fa[sl_p], fb[sl_p], idx[sl_q], w)):
            ma, mb = p_free & q_fixed_a, p_free & q_fixed_b
            add(np.full(ma.sum(), S), ip[ma], wsel[ma], both=False)                   # next to a fixed a pixel: leaving a costs w
            add(ip[mb], np.full(mb.sum(), T), wsel[mb], both=False)                   # next to a fixed b pixel
    fr_idx = idx[free]
    must_b, must_a = ~ok_a[free], ~ok_b[free]                                         # outside a's frame -> b, and vice versa
    add(fr_idx[must_b], np.full(must_b.sum(), T), np.full(must_b.sum(), INF), both=False)
    add(np.full(must_a.sum(), S), fr_idx[must_a], np.full(must_a.sum(), INF), both=False)
    rows_ = np.concatenate(rs).astype(np.int32)
    cols_ = np.concatenate(cs).astype(np.int32)
    data = np.concatenate(ws).astype(np.int32)
    cap = csr_matrix((data, (rows_, cols_)), shape=(m + 2, m + 2), dtype=np.int32)
    res = maximum_flow(cap, S, T)
    resid = (cap - res.flow).tocsr()
    resid.data = (resid.data > 0).astype(np.int8)
    resid.eliminate_zeros()
    reach = np.zeros(m + 2, bool)
    reach[breadth_first_order(resid, S, directed=True, return_predecessors=False)] = True
    new = np.where(reach[idx[free]], a, b)
    old = sub[free]
    if np.array_equal(new, old):
        return False
    sub[free] = new
    return True


def refine(labels, centres_px, g, radius_px, k=SEAM_K, eps=SEAM_EPS, sweeps=4, band=6):
    """Graph-cut refinement of a block label image (see the module docstring, step 3). Returns a new label image."""
    lab = np.array(labels, copy=True)
    ch, cv = _edge_costs(g, k, eps)
    for _ in range(sweeps):
        changed = False
        for a, b in _neighbour_pairs(lab):
            changed |= _swap(lab, a, b, centres_px, int(radius_px), ch, cv, band)
        if not changed:
            break
    return lab


# --------------------------------------------------------------------------
# connected, printable blocks
# --------------------------------------------------------------------------
def merge_islands(lab, centres_px, radius_px):
    """Make every block one connected piece: a piece that is not the main one goes to the block it touches most, as long
    as that block's writing frame holds the whole piece (otherwise it stays). Returns (labels, number of pieces moved)."""
    lab = np.array(lab, copy=True)
    rows, cols = lab.shape
    L = int(radius_px)
    nb = len(centres_px)
    moved = 0
    for k in range(nb):
        cy, cx = centres_px[k]
        y0, y1, x0, x1 = max(cy - L - 1, 0), min(cy + L + 2, rows), max(cx - L - 1, 0), min(cx + L + 2, cols)
        win = lab[y0:y1, x0:x1]                                            # a block lives inside its own frame
        cc, n = ndi.label(win == k)
        if n <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        main = cc[cy - y0, cx - x0] if cc[cy - y0, cx - x0] > 0 else int(np.argmax(sizes)) + 1
        for piece in range(1, n + 1):
            if piece == main:
                continue
            pm = cc == piece
            neigh = win[ndi.binary_dilation(pm) & ~pm]
            neigh = neigh[(neigh >= 0) & (neigh != k)]
            if neigh.size == 0:
                continue
            ys, xs = np.nonzero(pm)
            ys, xs = ys + y0, xs + x0
            counts = np.bincount(neigh, minlength=nb)
            for cand in np.argsort(-counts):
                if counts[cand] == 0:
                    break
                fy, fx = centres_px[cand]
                if np.abs(ys - fy).max() <= L and np.abs(xs - fx).max() <= L:     # the whole piece fits that block's frame
                    win[pm] = cand
                    moved += 1
                    break
    return lab, moved


def segment(H, pixel_size_um, radius_um, aperture=None, k=SEAM_K, eps=SEAM_EPS, sweeps=4):
    """Phase-quality-guided segmentation of the height map H (um, rows grow with y).

    Returns (centres_px, labels, gradient) like gradient_segmentation.segment: the writing centres (row, col), the block id
    of every pixel (-1 outside the aperture) and the normalised gradient."""
    H = np.asarray(H, dtype=np.float64)
    T = np.ones(H.shape, np.uint8) if aperture is None else (np.asarray(aperture) > 0).astype(np.uint8)
    if not T.any():
        raise ValueError("the aperture is empty")
    radius_px = int(np.floor(radius_um / pixel_size_um - 0.5))          # half a pixel of slack, see gradient_segmentation
    g = gs.height_gradient(H, pixel_size_um)
    centres = gs.lattice_centres(T, g, radius_px)
    lab = gs.assign_blocks(T, g, centres, radius_px)
    lab = refine(lab, centres, g, radius_px, k, eps, sweeps)
    lab, _ = merge_islands(lab, centres, radius_px)
    centres, lab = gs.compact(centres, lab)                              # a block the swaps emptied is not written
    return centres, lab, g


# --------------------------------------------------------------------------
# writing order
# --------------------------------------------------------------------------
def two_opt(points, order, max_passes=30):
    """2-opt improvement of an open writing path (Euclidean stage travel). Returns the improved order."""
    pts = np.asarray(points, dtype=float)
    o = list(np.asarray(order, dtype=int))
    n = len(o)
    if n < 4:
        return np.array(o)
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a = o[i - 1]
            ob = np.array(o)
            j = np.arange(i + 1, n)
            nxt = np.append(ob[j[:-1] + 1], -1)                                   # the pixel after position j (none at the end)
            old = D[a, ob[i]] + np.where(nxt >= 0, D[ob[j], np.where(nxt >= 0, nxt, 0)], 0.0)
            new = D[a, ob[j]] + np.where(nxt >= 0, D[ob[i], np.where(nxt >= 0, nxt, 0)], 0.0)
            gain = old - new
            jbest = int(np.argmax(gain))
            if gain[jbest] > 1e-9:
                jj = i + 1 + jbest
                o[i:jj + 1] = o[i:jj + 1][::-1]
                improved = True
        if not improved:
            break
    return np.array(o)


def travel_distance(centres_um, order):
    """Total stage travel (um) between consecutive fields in the writing order."""
    p = np.asarray(centres_um, dtype=float)[np.asarray(order, dtype=int)]
    return float(np.sqrt(((p[1:] - p[:-1]) ** 2).sum(1)).sum()) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def seam_metrics(labels, g, pixel_size_um, centres_um=None, order=None, k=SEAM_K, eps=SEAM_EPS, high=HIGH_GRADIENT) -> dict:
    """Numbers that compare segmentations of the same structure (smaller is better unless noted):

    blocks              number of blocks
    seam_length_um      total length of the borders between blocks
    seam_cost_mean      mean of the seam cost along the borders (the same cost for every mode, so they compare)
    seam_cost_total_um  seam length weighted by the cost: the quantity the graph cut minimises
    seam_on_step        fraction of the border lying on a strong height step (gradient >= `high`); LARGER is better
    disconnected_blocks blocks made of more than one piece
    extra_pieces        pieces beyond the first, summed over the blocks
    tiny_islands        extra pieces under 50 pixels
    travel_um           stage travel along the writing order (if centres and order are given)
    """
    lab = np.asarray(labels)
    g = np.asarray(g, dtype=np.float64)
    cost = seam_cost(g, k, eps)
    hx = (lab[:, :-1] != lab[:, 1:]) & (lab[:, :-1] >= 0) & (lab[:, 1:] >= 0)
    vx = (lab[:-1] != lab[1:]) & (lab[:-1] >= 0) & (lab[1:] >= 0)
    eg = np.concatenate((np.maximum(g[:, :-1], g[:, 1:])[hx], np.maximum(g[:-1], g[1:])[vx]))
    ec = np.concatenate((0.5 * (cost[:, :-1] + cost[:, 1:])[hx], 0.5 * (cost[:-1] + cost[1:])[vx]))
    n_edges = len(eg)
    ids = np.unique(lab[lab >= 0])
    extra = tiny = disc = 0
    boxes = ndi.find_objects(lab + 1)                                          # each block's bounding box
    for k_ in ids:
        cc, n = ndi.label(lab[boxes[k_]] == k_)
        if n > 1:
            sizes = np.sort(np.bincount(cc.ravel())[1:])[::-1]
            disc += 1
            extra += n - 1
            tiny += int((sizes[1:] < 50).sum())
    out = {"blocks": int(len(ids)), "seam_length_um": float(n_edges * pixel_size_um),
           "seam_cost_mean": float(ec.mean()) if n_edges else 0.0,
           "seam_cost_total_um": float(ec.sum() * pixel_size_um),
           "seam_on_step": float((eg >= high).mean()) if n_edges else 0.0,
           "disconnected_blocks": int(disc), "extra_pieces": int(extra), "tiny_islands": int(tiny)}
    if centres_um is not None and order is not None:
        out["travel_um"] = travel_distance(centres_um, order)
    return out
