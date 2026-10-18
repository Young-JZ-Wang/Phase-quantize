import numpy as np
import pytest
import scipy.io
from PIL import Image

import height_map_stl as hms


def read_tris(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    rec = np.frombuffer(raw[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    assert len(rec) == n
    return rec["v"].astype(np.float64)


@pytest.fixture
def staircase():
    """4 x 6 map with three plateaus: values 0, 1, 2."""
    a = np.zeros((4, 6))
    a[:, 2:4] = 1
    a[:, 4:] = 2
    return a


def test_png_to_stl_volume_and_bbox(tmp_path, staircase):
    p = tmp_path / "m.png"
    Image.fromarray((staircase * 100).astype(np.uint8)).save(p)
    rep = hms.height_map_to_stl(p, tmp_path / "m.stl", pixel_size_um=2.0, height_max_um=1.0, base_um=0.5)
    tri = read_tris(rep["output"])
    # columns: 8 px at 0.5, 8 px at 1.0, 8 px at 1.5 um, each 2 x 2 um
    assert rep["volume_mm3"] * 1e9 == pytest.approx(4 * (8 * 0.5 + 8 * 1.0 + 8 * 1.5), rel=1e-6)
    assert tri[..., 0].min() == pytest.approx(-6) and tri[..., 0].max() == pytest.approx(6)     # 6 px * 2 um / 2
    assert tri[..., 1].min() == pytest.approx(-4) and tri[..., 1].max() == pytest.approx(4)
    assert tri[..., 2].min() == 0 and tri[..., 2].max() == pytest.approx(1.5)
    assert rep["footprint_um"] == [12.0, 8.0] and rep["shape"] == [4, 6]


def test_npy_and_mat_give_the_same_solid_as_the_png(tmp_path, staircase):
    Image.fromarray((staircase * 100).astype(np.uint8)).save(tmp_path / "m.png")
    np.save(tmp_path / "m.npy", staircase)
    scipy.io.savemat(tmp_path / "m.mat", {"h": staircase, "scalar": np.array([1.0])})
    kw = dict(pixel_size_um=2.0, height_max_um=1.0, base_um=0.5)
    ref = read_tris(hms.height_map_to_stl(tmp_path / "m.png", tmp_path / "a.stl", **kw)["output"])
    for ext in ("npy", "mat"):
        out = read_tris(hms.height_map_to_stl(tmp_path / f"m.{ext}", tmp_path / f"{ext}.stl", **kw)["output"])
        assert np.array_equal(out, ref), ext


def test_values_as_heights_with_unit(tmp_path, staircase):
    np.save(tmp_path / "h_m.npy", staircase * 1e-6)                     # metres, like the quantizer's *_height_m.npy
    rep = hms.height_map_to_stl(tmp_path / "h_m.npy", pixel_size_um=1.0, value_unit="m", base_um=0.25)
    assert rep["height_um"] == pytest.approx(2.25)                      # peak 2 um + base
    assert rep["output"].endswith("h_m.stl")                            # default name next to the input


def test_levels_snap_heights(tmp_path):
    np.save(tmp_path / "ramp.npy", np.tile(np.linspace(0, 1, 40), (3, 1)))
    a = hms.load_height_array(tmp_path / "ramp.npy")
    q = hms.heights_from_array(a, 1.0, levels=5)
    assert sorted(np.unique(q)) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])


def test_merge_changes_triangle_count_not_volume(tmp_path, staircase):
    np.save(tmp_path / "m.npy", staircase)
    kw = dict(pixel_size_um=1.0, height_max_um=1.0, base_um=0.2)
    a = hms.height_map_to_stl(tmp_path / "m.npy", tmp_path / "a.stl", merge=True, **kw)
    b = hms.height_map_to_stl(tmp_path / "m.npy", tmp_path / "b.stl", merge=False, **kw)
    assert a["triangles"] < b["triangles"]
    assert a["volume_mm3"] == pytest.approx(b["volume_mm3"], rel=1e-9)


def test_row0_orientation(tmp_path):
    a = np.zeros((2, 2)); a[0, :] = 1                                  # top image row high
    np.save(tmp_path / "m.npy", a)
    for plus_y in (True, False):
        tri = read_tris(hms.height_map_to_stl(tmp_path / "m.npy", tmp_path / "o.stl", pixel_size_um=1.0, height_max_um=1.0,
                                              base_um=0.1, image_top_is_plus_y=plus_y)["output"])
        high = tri[tri[..., 2] > 1.0]
        assert (high[..., 1].mean() > 0) == plus_y


@pytest.mark.parametrize("bad", ["image_no_height", "zero_base", "missing_key", "ambiguous_mat", "wrong_ext", "complex", "flat"])
def test_errors(tmp_path, staircase, bad):
    np.save(tmp_path / "m.npy", staircase)
    Image.fromarray(np.zeros((3, 3), np.uint8)).save(tmp_path / "i.png")
    scipy.io.savemat(tmp_path / "two.mat", {"a": staircase, "b": staircase})
    np.save(tmp_path / "c.npy", staircase * (1 + 1j))
    np.save(tmp_path / "flat.npy", np.ones((3, 3)))
    cases = {
        "image_no_height": lambda: hms.height_map_to_stl(tmp_path / "i.png", pixel_size_um=1.0),
        "zero_base": lambda: hms.height_map_to_stl(tmp_path / "m.npy", pixel_size_um=1.0, height_max_um=1.0, base_um=0.0),
        "missing_key": lambda: hms.height_map_to_stl(tmp_path / "two.mat", pixel_size_um=1.0, height_max_um=1.0, key="zz"),
        "ambiguous_mat": lambda: hms.height_map_to_stl(tmp_path / "two.mat", pixel_size_um=1.0, height_max_um=1.0),
        "wrong_ext": lambda: hms.height_map_to_stl(tmp_path / "m.txt", pixel_size_um=1.0, height_max_um=1.0),
        "complex": lambda: hms.height_map_to_stl(tmp_path / "c.npy", pixel_size_um=1.0, height_max_um=1.0),
        "flat": lambda: hms.height_map_to_stl(tmp_path / "flat.npy", pixel_size_um=1.0, height_max_um=1.0),
    }
    with pytest.raises(ValueError):
        cases[bad]()
