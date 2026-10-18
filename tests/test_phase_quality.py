from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("skimage")
import gradient_segmentation as gs  # noqa: E402
import phase_quality as pq  # noqa: E402


def stairs(n=240, width=40, levels=6):
    return np.tile((np.arange(n) // width % levels) * 0.2, (n, 1)) + 0.2


def sparse(n=300):
    H = np.zeros((n, n)); H[:, 60:] += 0.2; H[:, 150:] += 0.2; H[:, 240:] += 0.2
    return H + 0.2


def test_seam_cost_is_cheap_on_steps_and_dear_in_smooth_areas():
    c = pq.seam_cost(np.array([0.0, 0.25, 0.5, 1.0]))
    assert c[0] == pytest.approx(1.0) and np.all(np.diff(c) < 0)                       # falls with the gradient
    assert c[-1] == pytest.approx(pq.SEAM_EPS + (1 - pq.SEAM_EPS) * np.exp(-pq.SEAM_K)) and c[-1] > pq.SEAM_EPS   # never free
    assert pq.seam_cost(0.0, k=0) == 1.0


def test_metrics_on_a_hand_made_segmentation():
    lab = np.zeros((4, 6), int); lab[:, 3:] = 1                                        # two blocks, one vertical seam, 4 edges
    g = np.zeros((4, 6)); g[:, 2:4] = 1.0                                               # the seam lies on a step
    m = pq.seam_metrics(lab, g, 2.0, centres_um=[[0, 0], [3, 4]], order=[0, 1])
    assert m["blocks"] == 2 and m["seam_length_um"] == pytest.approx(8.0)              # 4 edges of 2 um
    assert m["seam_on_step"] == 1.0 and m["seam_cost_mean"] == pytest.approx(pq.seam_cost(1.0))
    assert m["seam_cost_total_um"] == pytest.approx(8.0 * pq.seam_cost(1.0))
    assert m["travel_um"] == pytest.approx(5.0) and m["extra_pieces"] == 0 and m["disconnected_blocks"] == 0
    flat = pq.seam_metrics(lab, np.zeros((4, 6)), 2.0)
    assert flat["seam_on_step"] == 0.0 and flat["seam_cost_mean"] == pytest.approx(1.0)  # the same seam in a smooth area
    lab2 = lab.copy(); lab2[0, 0] = 1                                                   # a 1-pixel island of block 1 inside block 0
    m2 = pq.seam_metrics(lab2, g, 2.0)
    assert m2["disconnected_blocks"] == 1 and m2["extra_pieces"] == 1 and m2["tiny_islands"] == 1
    assert pq.seam_metrics(np.full((3, 3), -1), np.zeros((3, 3)), 1.0)["blocks"] == 0   # nothing inside the aperture


def test_refinement_lowers_the_seam_cost_and_keeps_every_rule():
    H = stairs(240, width=34)
    T = np.ones(H.shape, np.uint8)
    g = gs.height_gradient(H, 1.0)
    R = 30
    cen = gs.lattice_centres(T, g, R)
    start = gs.assign_blocks(T, g, cen, R)
    lab = pq.refine(start, cen, g, R, sweeps=3)
    before, after = pq.seam_metrics(start, g, 1.0), pq.seam_metrics(lab, g, 1.0)
    assert after["seam_cost_total_um"] < before["seam_cost_total_um"]                  # the graph cut found cheaper seams
    ys, xs = np.indices(H.shape)
    assert (lab >= 0).all()                                                             # every aperture pixel has a block
    assert np.abs(ys - cen[lab, 0]).max() <= R and np.abs(xs - cen[lab, 1]).max() <= R  # inside the writing frame
    assert not np.array_equal(lab, start)


def test_refinement_never_raises_the_cost_sweep_by_sweep():
    H = sparse(200)
    T = np.ones(H.shape, np.uint8); g = gs.height_gradient(H, 1.0); R = 28
    cen = gs.lattice_centres(T, g, R)
    lab = gs.assign_blocks(T, g, cen, R)
    last = pq.seam_metrics(lab, g, 1.0)["seam_cost_total_um"]
    for _ in range(3):
        lab = pq.refine(lab, cen, g, R, sweeps=1)
        now = pq.seam_metrics(lab, g, 1.0)["seam_cost_total_um"]
        assert now <= last + 1e-9
        last = now


def test_segment_is_a_connected_deterministic_partition_inside_the_frame():
    H = stairs(240, width=34)
    cen, lab, g = pq.segment(H, 2.0, 70.71)
    assert (lab >= 0).all() and lab.max() == len(cen) - 1 and all((lab == k).any() for k in range(len(cen)))
    m = pq.seam_metrics(lab, g, 2.0)
    assert m["extra_pieces"] == 0 and m["tiny_islands"] == 0                           # printable: no islands
    ys, xs = np.indices(lab.shape)
    reach = np.maximum(np.abs(ys - cen[lab, 0]), np.abs(xs - cen[lab, 1])) * 2.0 + 1.0
    assert reach.max() <= 70.71                                                         # a pixel's whole area is in the frame
    cen2, lab2, _ = pq.segment(H, 2.0, 70.71)
    assert np.array_equal(cen, cen2) and np.array_equal(lab, lab2)
    total = sum(gs.block_mask(lab, k).astype(int) for k in range(len(cen)))
    assert (total == 1).all()                                                           # the per-block masks partition the aperture


def test_phase_quality_beats_the_nearest_centre_blocks_on_the_seam_cost():
    H = sparse(300)
    cen, lab, g = pq.segment(H, 1.0, 40.0)
    yy, xx = np.indices(H.shape)
    from scipy.spatial import cKDTree
    near = cKDTree(cen).query(np.column_stack((yy.ravel(), xx.ravel())))[1].reshape(H.shape)
    a, b = pq.seam_metrics(lab, g, 1.0), pq.seam_metrics(near, g, 1.0)
    assert a["seam_cost_total_um"] < 0.8 * b["seam_cost_total_um"]
    assert a["seam_on_step"] > b["seam_on_step"] + 0.1
    # and it is no worse than its own starting point, the height-gradient-guided blocks
    c2, lab2, g2 = gs.segment(H, 1.0, 40.0)
    assert a["seam_cost_total_um"] <= pq.seam_metrics(lab2, g2, 1.0)["seam_cost_total_um"] + 1e-6


def test_aperture_and_empty_blocks():
    H = stairs(200)
    yy, xx = np.indices(H.shape)
    T = ((xx - 100) ** 2 + (yy - 100) ** 2 < 90 ** 2).astype(np.uint8)
    cen, lab, _ = pq.segment(H, 1.0, 25.0, aperture=T)
    assert (lab[T == 0] == -1).all() and (lab[T > 0] >= 0).all()
    with pytest.raises(ValueError):
        pq.segment(H, 1.0, 25.0, aperture=np.zeros_like(T))
    c, l = gs.compact(np.array([[1, 1], [5, 5], [9, 9]]), np.array([[0, 2], [2, -1]], dtype=np.int32))   # block 1 is empty
    assert c.tolist() == [[1, 1], [9, 9]] and l.tolist() == [[0, 1], [1, -1]]


def test_islands_are_merged_into_the_neighbour_that_holds_them():
    lab = np.zeros((40, 40), np.int32); lab[:, 20:] = 1
    lab[5:8, 30:33] = 0                                       # a piece of block 0 inside block 1 (inside both frames)
    fixed, moved = pq.merge_islands(lab, np.array([[20, 10], [20, 30]]), 25)
    assert moved == 1 and (fixed[5:8, 30:33] == 1).all()
    assert pq.seam_metrics(fixed, np.zeros((40, 40)), 1.0)["extra_pieces"] == 0
    # it stays when the neighbour's frame cannot hold it
    far = np.zeros((40, 60), np.int32); far[:, 20:] = 1; far[5:8, 22:25] = 0
    kept, moved2 = pq.merge_islands(far, np.array([[20, 10], [20, 45]]), 15)
    assert moved2 == 0 and (kept[5:8, 22:25] == 0).all()


def test_two_opt_shortens_a_bad_order_and_keeps_every_field():
    rng = np.random.default_rng(1)
    pts = rng.random((60, 2)) * 1000
    bad = rng.permutation(60)
    good = pq.two_opt(pts, bad)
    assert sorted(good.tolist()) == list(range(60))
    assert pq.travel_distance(pts, good) < 0.6 * pq.travel_distance(pts, bad)
    assert pq.travel_distance(pts, pq.two_opt(pts, good)) <= pq.travel_distance(pts, good) + 1e-9    # stable
    assert pq.two_opt(pts[:3], [2, 0, 1]).tolist() == [2, 0, 1]                                         # nothing to improve
    assert pq.travel_distance(pts, [0]) == 0.0


# ------------------------------------------------------------------ job, preview, comparison
cv2 = pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
import job_generator as jg  # noqa: E402
import slicer_preview as sp  # noqa: E402
from recipes import default_recipe  # noqa: E402
from state import Design  # noqa: E402


def design(n=200):
    return Design(source_path=str(Path("x") / "steps.npy"), origin="quantizer", heights_um=stairs(n, width=34) - 0.2,
                  pixel_size_um=1.0, levels=6, step_um=0.2, height_max_um=1.0)


def rec():
    r = default_recipe(); r.hatching_distance = 0.5
    return r


def job(tmp_path, **kw):
    js = jg.settings_from_design(design(), str(tmp_path), method="adaptive", base_um=0.2,
                                 segmentation_mode="phase_quality_guided", **kw)
    return jg.generate_job_files(js, rec())


def test_job_writes_the_phase_quality_outputs(tmp_path):
    rep = job(tmp_path, keep_intermediate=True)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"steps_segmentation_labels.png", "steps_sliced_comparison.png", "steps_writing_path.png",
            "steps_sliced_adaptive.png", "steps_sliced_standard_square.png"} <= names
    lab = np.flipud(np.asarray(Image.open(tmp_path / "steps_segmentation_labels.png"))).astype(int) - 1
    assert (lab >= 0).all() and lab.max() + 1 == rep["fields"]
    assert Image.open(tmp_path / "steps_sliced_comparison.png").size[0] >= 3 * 200
    assert "phase-quality-guided" in (tmp_path / "steps_job.gwl").read_text()
    data = (tmp_path / "steps_data.gwl").read_text(encoding="utf-8")
    assert data.count("% FIELD") == rep["fields"] and data.count("include steps_files") == rep["fields"]
    # every field file writes something (no empty block), inside the frame, and the aperture is fully covered
    r = default_recipe().adaptive_radius
    for f in (tmp_path / "steps_files").iterdir():
        pts = np.array([[float(v) for v in l.split()] for l in f.read_text().replace("\r", "").split("\n")
                        if len(l.split()) == 3 and not l.startswith("%")])
        assert len(pts) > 0 and np.abs(pts[:, :2]).max() <= r + 1e-6
    import tables
    area = 0.0
    for f in sorted((tmp_path / "work" / "h5").glob("block_*.h5")):
        with tables.open_file(f) as h:
            area += float(h.root.data[0].sum()) * 0.5 * 0.5
    assert 200.0 ** 2 <= area <= 200.0 ** 2 * 1.02


def test_the_job_follows_the_preview_plan(tmp_path):
    d = design()
    fp = sp.design_footprint(d.heights_um, 1.0, 0.2, True)
    lay = sp.compute_layout(fp, "adaptive", rec(), segmentation_mode="phase_quality_guided")
    assert lay.segmentation == "phase_quality_guided" and "phase-quality-guided" in sp.caption(lay)
    rep = job(tmp_path)
    assert rep["fields"] == len(lay.centres)
    assert sorted(lay.order.tolist()) == list(range(len(lay.centres)))


def test_validation(tmp_path):
    d = design()
    with pytest.raises(ValueError, match="adaptive"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="standard",
                                                      segmentation_mode="phase_quality_guided"), rec())
    stl = tmp_path / "s.stl"
    import height_map_stl as hms
    hms.heights_to_stl(d.heights_um, stl, pixel_size_um=1.0, base_um=0.2)
    d2 = Design(source_path=str(stl), origin="file", stl_path=str(stl))
    with pytest.raises(ValueError, match="height map"):
        jg.generate_job_files(jg.settings_from_design(d2, str(tmp_path / "o"), method="adaptive",
                                                      segmentation_mode="phase_quality_guided"), rec())
    assert "phase_quality_guided" in jg.SEGMENTATIONS and "phase_quality_guided" in jg.LABEL_MODES


def test_all_five_modes_compare_with_metrics_and_paths():
    d = design()
    img, lays = sp.comparison_image(d.heights_um, 1.0, 0.2, True, rec(), max_side=200,
                                    modes=("standard", "footprint", "threshold", "gradient", "phase"))
    assert list(lays) == ["standard", "footprint", "threshold", "gradient", "phase"]
    assert img.size[0] >= 3 * 200 and img.size[1] >= 2 * 200                           # 3 x 2 cells
    met = sp.layout_metrics(lays)
    assert set(met) == set(lays)
    for k, m in met.items():
        assert m["blocks"] == len(lays[k].centres) and m["travel_um"] > 0 and m["seam_length_um"] > 0
        assert 0 <= m["seam_on_step"] <= 1 and 0 < m["seam_cost_mean"] <= 1
        assert {"extra_pieces", "tiny_islands", "disconnected_blocks"} <= set(m)
    # the same cost for every mode, and the phase-quality seams are the cheapest of the adaptive ones
    assert met["phase"]["seam_cost_mean"] <= min(met["footprint"]["seam_cost_mean"], met["threshold"]["seam_cost_mean"])
    assert met["phase"]["extra_pieces"] == 0
    pimg = sp.render_path(lays["phase"])
    assert pimg.size == lays["phase"].footprint.T.shape[::-1]
    with pytest.raises(ValueError, match="unknown comparison mode"):
        sp.comparison_image(d.heights_um, 1.0, 0.2, True, rec(), modes=("standard", "nope"))
    three, l3 = sp.comparison_image(d.heights_um, 1.0, 0.2, True, rec(), max_side=200)             # the default is unchanged
    assert set(l3) == {"standard", "threshold", "gradient"} and three.size[1] < 2 * 200


def test_compare_command_writes_everything(tmp_path):
    import json
    import nanoslicer as ns
    np.save(tmp_path / "h.npy", design().heights_um)
    ns.cli(["compare", str(tmp_path / "h.npy"), str(tmp_path / "c.png"), "--pixel-size", "1", "--base", "0.2", "--max-side", "200",
            "--metrics", str(tmp_path / "m.json"), "--path-image", str(tmp_path / "p.png"), "--labels", str(tmp_path / "l.png")])
    assert all((tmp_path / n).is_file() for n in ("c.png", "m.png".replace("m.png", "m.json"), "p.png", "l.png"))
    assert set(json.loads((tmp_path / "m.json").read_text())) == {"standard", "footprint", "threshold", "gradient", "phase"}
    ns.cli(["compare", str(tmp_path / "h.npy"), str(tmp_path / "c3.png"), "--pixel-size", "1", "--base", "0.2", "--max-side", "200", "--three"])
    assert Image.open(tmp_path / "c3.png").size[1] < Image.open(tmp_path / "c.png").size[1]
