from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage as ndi
from PIL import Image

pytest.importorskip("skimage")
import gradient_segmentation as gs  # noqa: E402


def stairs(n=240, step_um=0.2, levels=6, width=40):
    """Vertical stripes whose height rises one level per stripe (0 ... levels-1), then falls back: many internal steps."""
    x = np.arange(n)
    lev = (x // width) % levels
    return np.tile(lev * step_um, (n, 1)) + 0.2


def test_gradient_sees_every_quantized_step_not_only_the_outline():
    H = stairs()
    g = gs.height_gradient(H, 1.0)
    assert g.min() == 0 and g.max() == 1
    edges = [x for x in range(1, H.shape[1]) if H[0, x] != H[0, x - 1]]
    assert len(edges) >= 5
    for x in edges:                                                    # level 1->2, 2->3 ... all as strong as 0->1
        up = H[0, x] > H[0, x - 1]
        if up:
            assert g[:, x - 1:x + 1].max() > 0.5, x
    # the thresholded silhouette (what the older mode sees) has a single outline, at the first step only
    silhouette = (H > 0.2 + 1e-6).astype(float)
    outline = np.abs(np.gradient(silhouette, axis=1)).max(axis=0) > 0
    assert outline.sum() < 2 * len(edges) // 2 + 2 and (g.max(axis=0) > 0.5).sum() > outline.sum()


def test_gradient_normalisation_and_flat_map():
    assert not gs.height_gradient(np.ones((20, 20)), 1.0).any()                 # no division by zero on a flat map
    H = np.zeros((20, 20)); H[:, 10:] = 5.0                                      # one huge wrap ...
    H[:, 15:] = 5.2                                                              # ... and one small step
    g = gs.height_gradient(H, 1.0)
    assert 0 <= g.min() and g.max() <= 1 and g[:, 14:16].max() > 0               # the small step is still there
    # the gradient is that of the actual heights: the same shape at another height scale
    assert np.allclose(gs.height_gradient(3 * stairs(), 1.0), gs.height_gradient(stairs() - 0.2, 1.0))


def test_centres_cover_the_aperture_within_the_write_radius():
    H = stairs(200)
    T = np.ones(H.shape, np.uint8)
    T[:60, :60] = 0                                                              # an odd aperture: a corner is missing
    g = gs.height_gradient(H, 1.0)
    c = gs.lattice_centres(T, g, 30)
    assert T[c[:, 0], c[:, 1]].all()                                             # centres lie in the aperture
    seed = np.zeros(T.shape, bool); seed[c[:, 0], c[:, 1]] = True
    assert (ndi.distance_transform_edt(~seed)[T > 0]).max() <= 30                # every pixel within the write radius
    ideal = T.sum() / (1.5 * np.sqrt(3) * 30 ** 2)
    assert len(c) < 2.2 * ideal + 5                                              # economical (edge discs cost extra)
    with pytest.raises(ValueError, match="3 pixels"):
        gs.lattice_centres(T, g, 2)


def test_blocks_partition_the_aperture_and_stay_in_their_frame():
    H = stairs(240)
    cen, lab, g = gs.segment(H, 1.0, 30.0)
    assert lab.shape == H.shape and (lab >= 0).all() and lab.max() == len(cen) - 1       # every aperture pixel has a block
    ys, xs = np.indices(H.shape)
    assert np.abs(ys - cen[lab, 0]).max() <= 30 and np.abs(xs - cen[lab, 1]).max() <= 30     # inside its own writing frame
    assert all((lab == k).any() for k in range(len(cen)))                                  # no empty block
    # masks are disjoint and add up to the aperture
    total = np.zeros(H.shape, int)
    for k in range(len(cen)):
        total += gs.block_mask(lab, k)
    assert (total == 1).all()
    # blocks are not fragmented
    for k in range(len(cen)):
        cc, n = ndi.label(lab == k)
        assert np.bincount(cc.ravel())[1:].max() >= 0.9 * (lab == k).sum()


def test_aperture_outside_stays_unlabelled():
    H = stairs(200)
    yy, xx = np.indices(H.shape)
    T = ((xx - 100) ** 2 + (yy - 100) ** 2 < 90 ** 2).astype(np.uint8)                   # a circular aperture
    cen, lab, _ = gs.segment(H, 1.0, 25.0, aperture=T)
    assert (lab[T == 0] == -1).all() and (lab[T > 0] >= 0).all()
    with pytest.raises(ValueError):
        gs.segment(H, 1.0, 25.0, aperture=np.zeros_like(T))


def test_borders_prefer_height_steps_over_plain_nearest_centre():
    """Sparse steps (every 60 px, most of the map smooth): the gradient-guided borders sit on them clearly more often."""
    H = np.zeros((300, 300)); H[:, 60:] += 0.2; H[:, 150:] += 0.2; H[:, 240:] += 0.2
    H += 0.2
    from scipy.spatial import cKDTree
    cen, lab, g = gs.segment(H, 1.0, 40.0)
    step = ndi.binary_dilation(g > 0.05, iterations=1)
    yy, xx = np.indices(H.shape)
    near = cKDTree(cen).query(np.column_stack((yy.ravel(), xx.ravel())))[1].reshape(H.shape)

    def on_step(l):
        b = np.zeros(l.shape, bool); b[1:] |= l[1:] != l[:-1]; b[:, 1:] |= l[:, 1:] != l[:, :-1]
        return (b & step).sum() / b.sum()
    assert on_step(lab) > on_step(near) + 0.1


def test_ridge_cost_zero_is_plain_nearest_centre_blocks():
    H = stairs(120)
    T = np.ones(H.shape, np.uint8)
    cen = gs.lattice_centres(T, gs.height_gradient(H, 1.0), 25)
    from scipy.spatial import cKDTree
    lab = gs.assign_blocks(T, gs.height_gradient(H, 1.0), cen, 25, ridge_cost=0.0)
    yy, xx = np.indices(H.shape)
    near = cKDTree(cen).query(np.column_stack((yy.ravel(), xx.ravel())))[1].reshape(H.shape)
    assert (lab == near).mean() > 0.97                                            # without the gradient, the Voronoi cells


def test_deterministic():
    H = stairs(150)
    a, b = gs.segment(H, 1.0, 25.0), gs.segment(H, 1.0, 25.0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_label_image_and_cells_sampling(tmp_path):
    H = stairs(100)
    cen, lab, _ = gs.segment(H, 2.0, 30.0)
    gs.save_label_image(lab, tmp_path / "l.png")
    img = np.asarray(Image.open(tmp_path / "l.png"))
    assert img.dtype == np.uint16 and np.array_equal(np.flipud(img) - 1, lab)       # id + 1, row 0 = the largest y
    cells = gs.LabelCells(lab, 0.0, 0.0, 2.0)                                       # x0, y0 of pixel (0, 0), pitch 2 um
    Xl, Yl = np.meshgrid(np.arange(0, 200, 0.5), np.arange(0, 200, 0.5))            # a finer block grid than the pixels
    m = cells.mask(3, Xl, Yl)
    assert m.shape == Xl.shape and m.sum() > 0 and m.dtype == np.uint8
    assert m[Xl.shape[0] // 2, 0] in (0, 1)
    out = cells.mask(3, Xl + 1e4, Yl)                                                # entirely outside the label image
    assert out.sum() == 0
    assert gs.resize_labels(lab, (300, 300)).shape == (300, 300)


# ------------------------------------------------------------------ job, preview, comparison
cv2 = pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
import job_generator as jg  # noqa: E402
import slicer_preview as sp  # noqa: E402
from recipes import default_recipe  # noqa: E402
from state import Design  # noqa: E402


def design(n=200):
    H = stairs(n, width=34)
    return Design(source_path=str(Path("x") / "steps.npy"), origin="quantizer", heights_um=H - 0.2, pixel_size_um=1.0,
                  levels=6, step_um=0.2, height_max_um=1.0)


def rec():
    r = default_recipe(); r.hatching_distance = 0.5
    return r


def job(tmp_path, **kw):
    js = jg.settings_from_design(design(), str(tmp_path), method="adaptive", base_um=0.2,
                                 segmentation_mode="height_gradient_guided", **kw)
    return jg.generate_job_files(js, rec())


def test_job_writes_blocks_labels_and_comparison(tmp_path):
    rep = job(tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert "steps_segmentation_labels.png" in names and "steps_sliced_comparison.png" in names
    assert "steps_sliced_adaptive.png" in names and "steps_sliced_standard_square.png" in names
    lab = np.flipud(np.asarray(Image.open(tmp_path / "steps_segmentation_labels.png"))).astype(int) - 1
    assert lab.shape == (200, 200) and (lab >= 0).all() and lab.max() + 1 == rep["fields"]
    assert Image.open(tmp_path / "steps_sliced_comparison.png").size[0] >= 3 * 200
    assert "height-gradient-guided" in (tmp_path / "steps_job.gwl").read_text()
    # job and data files keep their structure
    data = (tmp_path / "steps_data.gwl").read_text(encoding="utf-8")
    assert data.count("% FIELD") == rep["fields"] and data.count("include steps_files") == rep["fields"]


def test_job_covers_the_whole_footprint_with_no_gaps(tmp_path):
    rep = job(tmp_path, keep_intermediate=True)
    import tables
    area = 0
    for f in sorted((tmp_path / "work" / "h5").glob("block_*.h5")):
        with tables.open_file(f) as h:
            area += float(h.root.data[0].sum()) * 0.5 * 0.5                           # slice 0, hatching grid 0.5 um
    footprint = 200.0 * 200.0
    assert footprint <= area <= footprint * 1.02                                     # all written; seams overlap only slightly
    assert rep["fields"] > 4


def test_a_pixels_whole_area_stays_inside_its_blocks_frame():
    """A fine writing-grid point is looked up by its nearest pixel, up to half a pixel away. For no point to fall outside
    the frame of the block that owns its pixel, every pixel must be at most (radius - half a pixel) from its centre."""
    pitch, radius = 2.0, 70.71
    cen, lab, _ = gs.segment(stairs(240, width=23) - 0.2, pitch, radius)
    ys, xs = np.indices(lab.shape)
    reach = np.maximum(np.abs(ys - cen[lab, 0]), np.abs(xs - cen[lab, 1])) * pitch + pitch / 2.0
    assert reach.max() <= radius


def test_no_gap_at_coarse_pixels(tmp_path):
    """2 um pixels with a 70.71 um frame: a half pixel of slack must not leave hairline gaps between blocks."""
    d = Design(source_path=str(tmp_path / "c.npy"), origin="quantizer", heights_um=stairs(120, width=17) - 0.2,
               pixel_size_um=2.0, levels=6, step_um=0.2, height_max_um=1.0)
    js = jg.settings_from_design(d, str(tmp_path), method="adaptive", base_um=0.2, segmentation_mode="height_gradient_guided",
                                 keep_intermediate=True)
    jg.generate_job_files(js, rec())
    import tables
    area = 0.0
    for f in sorted((tmp_path / "work" / "h5").glob("block_*.h5")):
        with tables.open_file(f) as h:
            area += float(h.root.data[0].sum()) * 0.5 * 0.5
    assert 240.0 ** 2 <= area <= 240.0 ** 2 * 1.02


def test_field_files_stay_inside_the_writing_frame(tmp_path):
    job(tmp_path)
    r = default_recipe().adaptive_radius
    for f in (tmp_path / "steps_files").iterdir():
        pts = np.array([[float(v) for v in l.split()] for l in f.read_text().replace("\r", "").split("\n")
                        if len(l.split()) == 3 and not l.startswith("%")])
        assert np.abs(pts[:, :2]).max() <= r + 1e-6


def test_heights_are_untouched(tmp_path):
    d = design()
    before = d.heights_um.copy()
    job(tmp_path)
    jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "b"), method="adaptive", base_um=0.2,
                                                  segmentation_mode="height_gradient_guided"), rec())
    assert np.array_equal(d.heights_um, before)
    for f in (tmp_path / "steps_files").iterdir():                                    # every layer's z is one of the design's
        zs = {round(float(l.split()[2]), 3) for l in f.read_text().replace("\r", "").split("\n")
              if len(l.split()) == 3 and not l.startswith("%")}
        assert zs <= {round(0.2 * k, 3) for k in range(8)}


def test_validation(tmp_path):
    d = design()
    with pytest.raises(ValueError, match="adaptive"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="standard",
                                                      segmentation_mode="height_gradient_guided"), rec())
    with pytest.raises(ValueError, match="segmentation mode"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="adaptive", segmentation_mode="turbo"), rec())
    with pytest.raises(ValueError, match="needs a segmentation height"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="adaptive", segmentation_mode="height_threshold"), rec())
    with pytest.raises(ValueError, match="ridge cost"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="adaptive", base_um=0.2,
                                                      segmentation_mode="height_gradient_guided", ridge_cost=-1), rec())
    stl = tmp_path / "s.stl"
    import height_map_stl as hms
    hms.heights_to_stl(d.heights_um, stl, pixel_size_um=1.0, base_um=0.2)
    d2 = Design(source_path=str(stl), origin="file", stl_path=str(stl))
    with pytest.raises(ValueError, match="height map"):
        jg.generate_job_files(jg.settings_from_design(d2, str(tmp_path / "o"), method="adaptive",
                                                      segmentation_mode="height_gradient_guided"), rec())


def test_default_adaptive_mode_is_unchanged(tmp_path):
    """footprint_edges (and the implied height_threshold) behave as before: no label image, no comparison image."""
    d = design()
    jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "a"), method="adaptive", base_um=0.2), rec())
    assert not list((tmp_path / "a").glob("*labels*")) and not list((tmp_path / "a").glob("*comparison*"))
    jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "b"), method="adaptive", base_um=0.2, segment_height_um=0.4), rec())
    assert not list((tmp_path / "b").glob("*labels*"))


def test_preview_matches_the_job_and_is_labelled():
    d = design()
    fp = sp.design_footprint(d.heights_um, 1.0, 0.2, True)
    lay = sp.compute_layout(fp, "adaptive", rec(), segmentation_mode="height_gradient_guided")
    assert lay.labels is not None and lay.segmentation == "height_gradient_guided"
    assert "height-gradient-guided" in sp.caption(lay)
    img = sp.render(lay)
    assert img.size == fp.T.shape[::-1]
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        centres, _, labels = jg.gradient_layout(jg.tipslicer(), np.flipud(d.heights_um) + 0.2, fp.T, fp.X, fp.Y, 1.0,
                                                default_recipe().adaptive_radius, tmp, jg._Ctx())
    assert np.allclose(np.sort(lay.centres, axis=0), np.sort(centres, axis=0)) and np.array_equal(lay.labels, labels)
    with pytest.raises(ValueError, match="height map"):
        sp.compute_layout(sp.Footprint(T=fp.T, bg=fp.bg, X=fp.X, Y=fp.Y, pitch=1.0, size_um=(200, 200)), "adaptive", rec(),
                          segmentation_mode="height_gradient_guided")


def test_comparison_image_has_the_three_layouts(tmp_path):
    img, lays = sp.comparison_image(design().heights_um, 1.0, 0.2, True, rec(), max_side=200)
    assert set(lays) == {"standard", "threshold", "gradient"}
    assert lays["standard"].labels is None and lays["gradient"].labels is not None
    assert img.size[0] >= 3 * 200 and img.size[1] > 200
    # CLI: image + label image
    import nanoslicer as ns
    np.save(tmp_path / "h.npy", design().heights_um)
    ns.cli(["compare", str(tmp_path / "h.npy"), str(tmp_path / "c.png"), "--pixel-size", "1", "--base", "0.2",
            "--max-side", "200", "--labels", str(tmp_path / "l.png")])
    assert (tmp_path / "c.png").is_file() and (tmp_path / "l.png").is_file()
