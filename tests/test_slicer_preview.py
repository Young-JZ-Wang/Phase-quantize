import numpy as np
import pytest

import height_map_stl as hms
import slicer_preview as sp
from recipes import default_recipe


def heights(n=120):
    h = np.zeros((n, n))
    h[:, n // 2:] = 0.2
    return h


def test_design_footprint_orientation_and_size():
    h = heights(); h[0, :] = 0.4                                      # top image row high
    fp = sp.design_footprint(h, 1.0, base_um=0.2, image_top_is_plus_y=True)
    assert fp.T.shape == (120, 120) and fp.size_um == (120.0, 120.0) and fp.reduced == 1
    assert fp.bg[-1].min() == 255 and fp.bg[0].max() < 255              # +y (last row of the y-up array) holds the high row
    fp2 = sp.design_footprint(h, 1.0, base_um=0.2, image_top_is_plus_y=False)
    assert fp2.bg[0].min() == 255
    with pytest.raises(ValueError):
        sp.design_footprint(h, None)


def test_large_design_is_reduced_but_keeps_its_physical_size():
    fp = sp.design_footprint(np.zeros((1000, 800)), 2.0, base_um=0.2, max_side=250)
    assert fp.reduced == 4 and fp.T.shape == (250, 200) and fp.size_um == (1600.0, 2000.0) and fp.pitch == 8.0


def test_standard_layout_matches_the_job_grid():
    fp = sp.design_footprint(heights(), 1.0, base_um=0.2)
    lay = sp.compute_layout(fp, "standard", default_recipe())
    assert lay.centres.tolist() == [[-50, -50], [50, -50], [-50, 50], [50, 50]]      # 120 um -> 2 x 2 fields of 100 um
    assert lay.order.tolist() == [0, 1, 2, 3]                                       # row by row
    r = default_recipe(); r.field_size = 40
    assert len(sp.compute_layout(fp, "standard", r).centres) == 9
    # a reduced preview of the same structure plans the same grid
    big = np.zeros((600, 600))
    assert (sp.compute_layout(sp.design_footprint(big, 2.0, 0.2, max_side=100), "standard", default_recipe()).centres.tolist()
            == sp.compute_layout(sp.design_footprint(big, 2.0, 0.2, max_side=2000), "standard", default_recipe()).centres.tolist())


def test_labels_cover_the_structure_and_render_marks_borders():
    fp = sp.design_footprint(heights(), 1.0, base_um=0.2)
    lay = sp.compute_layout(fp, "standard", default_recipe())
    lab = sp.field_labels(lay)
    assert set(np.unique(lab)) == {0, 1, 2, 3} and (lab >= 0).all()
    img = sp.render(lay, overlap_um=4.0)
    a = np.asarray(img)
    assert img.size == (120, 120)
    assert (np.abs(a.astype(int) - np.array(sp.OVERLAP)) < 90).all(axis=2).any()     # overlap band
    assert (a == np.array(sp.FIRST)).all(axis=2).any() and (a == np.array(sp.LAST)).all(axis=2).any()


def test_borders_are_dashed_and_transparent():
    h = np.zeros((120, 120)); h[:, 60:] = 0.2; h[:60, :60] = 0.4                       # structure to see through the lines
    lay = sp.compute_layout(sp.design_footprint(h, 1.0, base_um=0.2), "standard", default_recipe())
    solid = np.asarray(sp.render(lay, dashed=False, border_alpha=1.0)).astype(int)
    clear = np.asarray(sp.render(lay, dashed=False, border_alpha=0.0)).astype(int)     # no borders at all
    dashed = np.asarray(sp.render(lay)).astype(int)                                     # the default look
    n_solid = (solid == sp.BORDER).all(axis=2).sum()
    on = (dashed != clear).any(axis=2)                                                  # pixels the default draws a border on
    assert n_solid > 0 and 0.3 * n_solid < on.sum() < 0.7 * n_solid                    # dashes: part of each line only
    assert not (dashed == sp.BORDER).all(axis=2).any()                                  # never a solid red pixel
    assert not np.array_equal(dashed[on], solid[on])
    # transparent: the structure still shows through (pixels over different heights differ)
    assert len({tuple(p) for p in dashed[on]}) > 1
    # along one border line the dashes alternate with gaps
    col = 59 if on[:, 59].any() else 60
    line = on[:, col] if on[:, col].sum() > 20 else on[:, 59]
    assert 0 < line.sum() < len(line)


def test_adaptive_layout_and_caption():
    pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
    fp = sp.design_footprint(np.zeros((300, 300)), 1.0, base_um=0.2)
    lay = sp.compute_layout(fp, "adaptive", default_recipe())
    assert len(lay.centres) >= 4 and sorted(lay.order.tolist()) == list(range(len(lay.centres)))
    assert "adaptive" in sp.caption(lay) and f"{len(lay.centres)} fields" in sp.caption(lay)
    assert sp.render(lay).size == (300, 300)


def test_adaptive_preview_matches_what_the_job_plans(tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
    import job_generator as jg
    h = np.zeros((200, 200)); h[:, 100:] = 0.2
    lay = sp.compute_layout(sp.design_footprint(h, 1.0, base_um=0.2), "adaptive", default_recipe())
    ts = jg.tipslicer()
    fp = sp.design_footprint(h, 1.0, base_um=0.2)
    centres, _ = jg.height_map_layout(ts, fp.T, fp.X, fp.Y, 1.0, "adaptive", 100.0, default_recipe().adaptive_radius,
                                      str(tmp_path), jg._Ctx())
    assert np.allclose(np.sort(lay.centres, axis=0), np.sort(centres, axis=0))


def test_stl_footprint_and_standard_tiling(tmp_path):
    a = np.zeros((60, 100)); a[:, 50:] = 0.2
    rep = hms.heights_to_stl(a, tmp_path / "s.stl", pixel_size_um=2.0, base_um=0.2)
    fp = sp.stl_footprint(rep["output"])
    assert fp.kind == "stl" and fp.size_um == pytest.approx((200.0, 120.0)) and fp.T.sum() > 0.95 * fp.T.size * 0.98
    lay = sp.compute_layout(fp, "standard", default_recipe())
    assert len(lay.centres) >= 2 and "STL" in sp.caption(lay)
    assert sp.render(lay).size == fp.T.shape[::-1]


def test_read_ascii_and_binary_stl(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 1, 1], [2, 1, 1], [1, 2, 1]]], float)
    hms.write_binary_stl(tmp_path / "b.stl", tri)
    assert np.allclose(hms.read_stl_triangles(tmp_path / "b.stl"), tri)
    lines = ["solid x"]
    for tr in tri:
        lines += ["facet normal 0 0 1", "outer loop"] + [f"vertex {v[0]} {v[1]} {v[2]}" for v in tr] + ["endloop", "endfacet"]
    (tmp_path / "a.stl").write_text("\n".join(lines + ["endsolid x"]) + "\n")
    assert np.allclose(hms.read_stl_triangles(tmp_path / "a.stl"), tri)
    (tmp_path / "bad.stl").write_text("nothing here")
    with pytest.raises(ValueError):
        hms.read_stl_triangles(tmp_path / "bad.stl")


def test_no_material_is_an_error():
    fp = sp.design_footprint(np.zeros((20, 20)), 1.0, base_um=0.0)
    with pytest.raises(ValueError):
        sp.compute_layout(fp, "standard", default_recipe())
    with pytest.raises(ValueError):
        sp.compute_layout(fp, "turbo", default_recipe())


def test_height_based_preview_matches_the_job_and_is_labelled():
    pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
    import tempfile
    import job_generator as jg
    y, x = np.mgrid[0:200, 0:200]
    h = np.zeros((200, 200)); h[((x - 60) ** 2 + (y - 60) ** 2) < 25 ** 2] = 0.4
    fp = sp.design_footprint(h, 1.0, base_um=0.2, segment_height_um=0.3)
    assert fp.T_seg is not None and 0 < fp.T_seg.sum() < fp.T.sum()
    lay = sp.compute_layout(fp, "adaptive", default_recipe(), segment_height_um=0.3)
    assert lay.segment_height == 0.3 and "height-based" in sp.caption(lay) and "z = 0.3" in sp.caption(lay)
    plain = sp.compute_layout(fp, "adaptive", default_recipe())
    assert plain.segment_height is None and "height-based" not in sp.caption(plain)
    with tempfile.TemporaryDirectory() as tmp:
        centres, _ = jg.height_map_layout(jg.tipslicer(), fp.T, fp.X, fp.Y, 1.0, "adaptive", 100.0,
                                          default_recipe().adaptive_radius, tmp, jg._Ctx(), T_seg=fp.T_seg)
    assert np.allclose(np.sort(lay.centres, axis=0), np.sort(centres, axis=0))
    # standard slicing ignores the segmentation height
    assert sp.compute_layout(fp, "standard", default_recipe(), segment_height_um=0.3).segment_height is None
    with pytest.raises(ValueError):                                               # footprint built without the cross-section
        sp.compute_layout(sp.design_footprint(h, 1.0, 0.2), "adaptive", default_recipe(), segment_height_um=0.3)


def test_stl_cross_section_at_a_height(tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
    a = np.zeros((60, 60)); a[:, 30:] = 1.0                                         # right half is 1 um taller
    rep = hms.heights_to_stl(a, tmp_path / "s.stl", pixel_size_um=1.0, base_um=1.0)  # base 1 um, right half 2 um
    fp = sp.stl_footprint(rep["output"], max_side=120)
    low, high = sp.stl_cross_section(fp, 0.5), sp.stl_cross_section(fp, 1.5)
    assert low.sum() > 0.95 * fp.T.sum() * 0.95                                     # at 0.5 um the whole solid
    assert 0.4 * low.sum() < high.sum() < 0.6 * low.sum()                           # at 1.5 um only the taller half
    lay = sp.compute_layout(fp, "adaptive", default_recipe(), segment_height_um=1.5)
    assert lay.segment_height == 1.5 and len(lay.centres) >= 1
