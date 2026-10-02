import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import phase_mask_quantizer as pq

EXAMPLES = Path(__file__).parent.parent / "examples"
TWO_PI = 2 * np.pi


def test_quantize_levels_and_wrap():
    phi = np.array([[0.0, TWO_PI, -TWO_PI / 6, 1.0]])
    q = pq.quantize(phi, 6)
    assert q.min() >= 0 and q.max() <= 5
    assert q[0, 0] == q[0, 1] == 0                      # 2*pi wraps to level 0
    assert q[0, 2] == 5                                 # -2pi/6 wraps to the top level


def test_equal_level_heights():
    d = pq.build_design(np.linspace(0, TWO_PI, 500, endpoint=False).reshape(1, -1),
                        wavelength_nm=532, levels=4)
    assert d["step"] == pytest.approx(d["depth_2pi"] / 4)
    assert set(np.unique(d["q"])) == {0, 1, 2, 3}


def test_step_mode_matches_phlatcam_formula():
    """PhlatCam: hq = dh * floor(h/dh), h = phi * lam / (2 pi dn)."""
    phi = np.random.default_rng(0).random((64, 64)) * TWO_PI
    d = pq.build_design(phi, wavelength_nm=532, step_nm=200)
    dn = pq.n_fused_silica(532e-9) - 1.0
    ref = 0.2e-6 * np.floor(phi * 0.532e-6 / (TWO_PI * dn) / 0.2e-6)
    assert np.allclose(d["heights"], ref)
    assert d["levels"] == 6                              # 1.04 um depth / 0.2 um step -> 6 levels


def test_step_mode_round_and_bounds():
    phi = np.random.default_rng(1).random((32, 32)) * TWO_PI
    d = pq.build_design(phi, wavelength_nm=532, step_nm=200, rounding="round")
    assert d["q"].max() <= d["levels"] - 1
    with pytest.raises(ValueError):
        pq.build_design(phi, wavelength_nm=532, step_nm=5000)   # step larger than the 2*pi depth


def test_material_checks():
    phi = np.zeros((4, 4))
    with pytest.raises(ValueError):                       # air in air: no index step
        pq.build_design(phi, wavelength_nm=532, material="air")
    with pytest.raises(ValueError):
        pq.build_design(phi, wavelength_nm=532, levels=1)
    d = pq.build_design(phi, wavelength_nm=532, n_index=1.5, n_substrate=1.46)
    assert d["n"] == 1.5 and d["n_substrate"] == 1.46
    assert d["depth_2pi"] == pytest.approx(532e-9 / 0.5)


def test_load_units_and_invert(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.full((3, 3), 0.25))
    assert pq.load_phase(p, unit="waves")[0, 0] == pytest.approx(TWO_PI / 4)
    assert pq.load_phase(p, unit="deg")[0, 0] == pytest.approx(np.deg2rad(0.25))
    a = pq.build_design(np.full((3, 3), 1.0), wavelength_nm=532, levels=8)
    b = pq.build_design(np.full((3, 3), 1.0), wavelength_nm=532, levels=8, invert=True)
    assert a["q"][0, 0] + b["q"][0, 0] == 8              # conjugate mask mirrors the level


@pytest.mark.parametrize("ext", [".png", ".bmp"])
def test_image_input(tmp_path, ext):
    g = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    p = tmp_path / f"g{ext}"
    Image.fromarray(g).save(p)
    phi = pq.load_phase(p)
    assert phi.shape == (4, 256)
    assert phi[0, 0] == 0 and phi[0, 255] == pytest.approx(TWO_PI)


def test_image_16bit_and_rgb(tmp_path):
    Image.fromarray(np.full((2, 2), 65535, np.uint16)).save(tmp_path / "a.png")
    assert pq.load_phase(tmp_path / "a.png")[0, 0] == pytest.approx(TWO_PI)
    Image.fromarray(np.full((2, 2, 3), 255, np.uint8)).save(tmp_path / "b.png")
    assert pq.load_phase(tmp_path / "b.png")[0, 0] == pytest.approx(TWO_PI)


def test_bad_inputs(tmp_path):
    (tmp_path / "x.txt").write_text("nope")
    with pytest.raises(ValueError):
        pq.load_phase(tmp_path / "x.txt")
    np.save(tmp_path / "v.npy", np.zeros(5))
    with pytest.raises(ValueError):
        pq.load_phase(tmp_path / "v.npy")


def test_mat_input(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    scipy_io.savemat(tmp_path / "m.mat", {"phi": np.ones((5, 7)), "scalar": 3})
    assert pq.load_phase(tmp_path / "m.mat").shape == (5, 7)
    scipy_io.savemat(tmp_path / "two.mat", {"a": np.ones((3, 3)), "b": np.ones((3, 3))})
    with pytest.raises(ValueError):
        pq.load_phase(tmp_path / "two.mat")              # ambiguous without a variable name
    assert pq.load_phase(tmp_path / "two.mat", "b").shape == (3, 3)


def test_stl_is_closed_and_consistently_oriented():
    """Every directed edge a->b must be matched by an opposite b->a (closed, outward-facing surface).

    Random level maps include diagonal-only contacts where 4 faces meet at one edge, so this checks
    orientation balance rather than 'exactly two triangles per edge'.
    """
    q = np.random.default_rng(2).integers(0, 4, (6, 7)).astype(np.uint16)
    tris = pq.heightfield_to_triangles(q, step=1.0, pitch=1.0, base=2.0)
    edges = Counter()
    for t in np.round(tris, 6):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[(tuple(t[a]), tuple(t[b]))] += 1
    assert all(edges[(b, a)] == n for (a, b), n in edges.items())


def test_stl_volume_matches_heightfield():
    q = np.array([[0, 1], [2, 3]], np.uint16)
    tris = pq.heightfield_to_triangles(q, step=1.0, pitch=2.0, base=1.0).astype(np.float64)
    vol = np.einsum("ij,ij->", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6   # divergence theorem
    assert vol == pytest.approx(4.0 * (q.sum() + 4 * 1.0))                          # pixel area * (height + base)


def test_export_files_and_report(tmp_path):
    report = pq.run(EXAMPLES / "lens_phase.png", 2.0, 532, "both", levels=4, output_dir=tmp_path)
    names = [Path(f).name for f in report["files"]]
    pngs = [f for f in report["files"] if f.endswith(".png") and "_continuous" not in f]
    assert len(pngs) == 1                                 # a single 8-bit image, no 16-bit map
    assert any(n.endswith(".stl") for n in names)
    rep = json.loads(Path(report["files"][-1]).read_text())
    assert rep["footprint_um"] == [512.0, 512.0]          # 256 px * 2 um
    img = Image.open(pngs[0])
    arr = np.asarray(img)
    assert img.mode == "L" and arr.dtype == np.uint8
    assert len(np.unique(arr)) == 4 and arr.max() == 255   # discrete, normalized by its own max
    stl = Path(next(f for f in report["files"] if f.endswith(".stl")))
    n_tri = int.from_bytes(stl.read_bytes()[80:84], "little")
    assert n_tri == report["stl_triangles"] and stl.stat().st_size == 84 + 50 * n_tri


def test_output_name_has_no_distance(tmp_path):
    d = pq.build_design(np.zeros((10, 20)), wavelength_nm=532, levels=6)
    name = pq.output_path(d, tmp_path / "foo.npy", 1.0).name
    assert name == "phHeight_foo_0.96umMaxH_1.00um_20um_q192nm_lam532"
    assert "mm" not in name


def test_png_only_needs_no_pitch(tmp_path):
    rep = pq.run(EXAMPLES / "vortex_phase.png", None, 532, "png", levels=8, output_dir=tmp_path)
    assert not any(f.endswith(".stl") for f in rep["files"])
    with pytest.raises(ValueError):
        pq.run(EXAMPLES / "vortex_phase.png", None, 532, "stl", output_dir=tmp_path)


def test_png_normalized_by_quantized_max(tmp_path):
    """Same as PhlatCam: png = round(255 * hq / max(hq)), exactly `levels` (or fewer) distinct values."""
    rep = pq.run(EXAMPLES / "smooth_phase.png", 1.0, 532, "png", step_nm=200, output_dir=tmp_path)
    arr = np.asarray(Image.open(next(f for f in rep["files"] if f.endswith(".png") and "_continuous" not in f)))
    assert set(np.unique(arr)) <= {round(255 * k / 5) for k in range(6)}


def test_too_many_levels_for_8bit(tmp_path):
    with pytest.raises(ValueError):
        pq.run(EXAMPLES / "smooth_phase.png", 1.0, 532, "png", levels=300, output_dir=tmp_path)


def test_phase_scale_doubles_phase_like_phlatcam_matlab():
    """MATLAB: phMm_mod = mod(2*phMm, 2*pi), then floor-quantize by dh."""
    phi = np.random.default_rng(3).random((40, 40)) * TWO_PI
    d = pq.build_design(phi, wavelength_nm=532, step_nm=200, phase_scale=2.0)
    dn = pq.n_fused_silica(532e-9) - 1.0
    ref = 0.2e-6 * np.floor(np.mod(2 * phi, TWO_PI) * 0.532e-6 / (TWO_PI * dn) / 0.2e-6)
    assert np.allclose(d["heights"], ref)


# ---------------------------------------------------------------- resampling / two-stage export
def test_original_mode_is_untouched():
    phi = np.random.default_rng(4).random((20, 30)) * TWO_PI
    assert pq.resample_phase(phi, "original", 8) is phi
    a = pq.build_design(phi, wavelength_nm=532, step_nm=200)
    b = pq.build_design(phi, wavelength_nm=532, step_nm=200, resample="original", factor=4)
    assert np.array_equal(a["q"], b["q"]) and b["pitch_scale"] == 1.0


@pytest.mark.parametrize("mode,factor,shape", [("up", 1.5, (30, 45)), ("up", 2, (40, 60)),
                                               ("up", 8, (160, 240)), ("down", 2, (10, 15)),
                                               ("down", 4, (5, 8))])
def test_resampled_shape(mode, factor, shape):
    d = pq.build_design(np.zeros((20, 30)), wavelength_nm=532, levels=4, resample=mode, factor=factor)
    assert d["q"].shape == shape
    assert d["resample"]["source_shape"] == [20, 30]


def test_resample_keeps_wrapped_phase_smooth():
    """A wrapped ramp must not gain spurious in-between values at the 2*pi jump."""
    ramp = np.mod(np.tile(np.linspace(0, 3 * TWO_PI, 60), (8, 1)), TWO_PI)
    up = pq.resample_phase(ramp, "up", 4)
    dphi = np.angle(np.exp(1j * np.diff(up, axis=1)))          # wrapped step between neighbours
    assert np.abs(dphi).max() < 0.2                             # smooth: no jumps off the ramp


def test_resample_errors():
    for kwargs in ({"resample": "up", "factor": 0.5}, {"resample": "up", "factor": 1000},
                   {"resample": "down", "factor": 20}, {"resample": "sideways"}):
        with pytest.raises(ValueError):
            pq.build_design(np.zeros((10, 10)), wavelength_nm=532, levels=4, **kwargs)


def test_two_stage_export_and_footprint(tmp_path):
    rep = pq.run(EXAMPLES / "lens_phase.png", 2.0, 532, "both", step_nm=200,
                 resample="up", factor=2, output_dir=tmp_path)
    names = [Path(f).name for f in rep["files"]]
    cont = [n for n in names if "_continuous" in n]
    assert len(cont) == 2
    assert names.index(cont[0]) < names.index(next(n for n in names if n.endswith(".stl")))   # stage 1 first
    raw = np.load(next(f for f in rep["files"] if f.endswith("_continuous.npy")))
    assert raw.shape == (256, 256) and raw.dtype == np.float64        # original grid, exact values
    c16 = Image.open(next(f for f in rep["files"] if f.endswith("_continuous.png")))
    assert c16.size == (256, 256) and np.asarray(c16).dtype == np.uint16
    disc = next(f for f in rep["files"] if f.endswith(".png") and "_continuous" not in f)
    assert Image.open(disc).size == (512, 512)                         # discrete grid is 2x finer
    assert rep["footprint_um"] == [512.0, 512.0]                       # physical size unchanged
    assert rep["pixel_pitch_um"] == pytest.approx(1.0) and rep["source_pixel_pitch_um"] == 2.0
    assert rep["resample"] == {"mode": "up", "factor": 2.0, "source_shape": [256, 256]}
    assert "_up2x" in Path(disc).name


def test_down_sampling_stl_uses_new_pitch(tmp_path):
    rep = pq.run(EXAMPLES / "lens_phase.png", 1.0, 532, "stl", step_nm=200, resample="down",
                 factor=4, output_dir=tmp_path)
    assert rep["pixel_pitch_um"] == pytest.approx(4.0) and rep["footprint_um"] == [256.0, 256.0]


def test_continuous_export_can_be_skipped(tmp_path):
    rep = pq.run(EXAMPLES / "lens_phase.png", 1.0, 532, "png", levels=4,
                 export_continuous=False, output_dir=tmp_path)
    assert not any("_continuous" in Path(f).name for f in rep["files"])


def test_cli_resample_flags(tmp_path, capsys):
    pq.cli([str(EXAMPLES / "lens_phase.png"), "1", "532", "png", "--step-nm", "200", "--resample", "down",
            "--factor", "2", "--output-dir", str(tmp_path)])
    assert any(p.name.endswith("_down2x.png") for p in tmp_path.iterdir())
    assert any(p.name.endswith("_continuous.npy") for p in tmp_path.iterdir())
