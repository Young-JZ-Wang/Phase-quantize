import numpy as np
import pytest
from PIL import Image

from state import AppState, Design, design_from_file


def design(**kw):
    base = dict(source_path="/x/mask.png", origin="quantizer", heights_um=np.zeros((4, 6)), pixel_size_um=2.0, levels=6,
                step_um=0.2, height_max_um=1.0, design_id=3)
    base.update(kw)
    return Design(**base)


def test_design_geometry_and_summary():
    d = design()
    assert d.shape == (4, 6) and d.footprint_um == (12.0, 8.0)
    s = d.summary()
    assert "6 × 4 px" in s and "12 × 8 µm" in s and "6 levels of 200.0 nm" in s and "height 1 µm" in s
    assert design(pixel_size_um=None).footprint_um is None
    assert design(heights_um=None).shape is None and design(heights_um=None).summary().startswith("STL")


def test_publish_notifies_and_files_attach_to_the_same_design_only():
    st, seen = AppState(), []
    st.subscribe(lambda s: seen.append(s.design))
    d = design()
    st.publish(d)
    assert seen == [d] and st.design is d
    st.files_written(3, png_path="a.png", stl_path="a.stl")                 # same design_id
    assert (d.png_path, d.stl_path) == ("a.png", "a.stl") and len(seen) == 2
    st.files_written(99, png_path="stale.png", stl_path="stale.stl")        # an older / other quantizer result
    assert d.png_path == "a.png" and len(seen) == 2
    st.publish(design(design_id=4, origin="file"))
    st.files_written(4, png_path="b.png")                                    # only quantizer designs take exports
    assert st.design.png_path is None


def test_design_from_file_height_map_and_stl(tmp_path):
    np.save(tmp_path / "h.npy", np.arange(12.0).reshape(3, 4))
    d = design_from_file(tmp_path / "h.npy", pixel_size_um=1.5, height_max_um=1.0, levels=3)
    assert d.origin == "file" and d.levels == 3 and d.step_um == pytest.approx(0.5) and d.height_max_um == pytest.approx(1.0)
    assert sorted(np.unique(d.heights_um)) == pytest.approx([0, 0.5, 1.0])
    s = design_from_file(tmp_path / "solid.stl")
    assert s.heights_um is None and s.stl_path.endswith("solid.stl")


def test_design_from_file_errors(tmp_path):
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(tmp_path / "i.png")
    with pytest.raises(ValueError, match="height max"):
        design_from_file(tmp_path / "i.png", pixel_size_um=1.0)
    d = design_from_file(tmp_path / "i.png", pixel_size_um=1.0, height_max_um=2.0)
    assert d.png_path.endswith("i.png") and d.height_max_um == pytest.approx(2.0)
