import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("tables")
pytest.importorskip("matplotlib")

import job_generator as jg                                   # noqa: E402
from recipes import default_recipe                            # noqa: E402

PITCH = 1.0


def design(n=120):
    """n x n px, 3 equidistant levels (0, 0.2, 0.4 um); the left half is the lowest, a disc the highest."""
    y, x = np.mgrid[0:n, 0:n]
    q = np.zeros((n, n), int)
    q[:, n // 2:] = 1
    q[(x - 0.7 * n) ** 2 + (y - 0.4 * n) ** 2 < (0.15 * n) ** 2] = 2
    return q


@pytest.fixture(scope="module")
def png(tmp_path_factory):
    p = tmp_path_factory.mktemp("in") / "mask.png"
    Image.fromarray((design() * 100).astype(np.uint8)).save(p)          # 0 / 100 / 200 -> exact thirds of the peak
    return p


def fast_recipe(**kw):
    r = default_recipe()
    r.hatching_distance = 0.5                                           # coarse lines: fast tests
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def settings(png, out, **kw):
    base = dict(input_file=str(png), output_dir=str(out), method="standard", pixel_size_um=PITCH,
                height_max_um=0.4, levels=3, base_um=0.2)
    base.update(kw)
    return jg.JobSettings(**base)


def active(path):
    return [re.sub(r"\s+", " ", l.strip()) for l in Path(path).read_text(encoding="utf-8").replace("\r", "").split("\n")
            if l.strip() and not l.strip().startswith("%")]


def read_field(path):
    """[(x, y, z)] points of all lines of a field file, one array per 'write'."""
    lines, cur = [], []
    for l in Path(path).read_text().replace("\r", "").split("\n"):
        t = l.split()
        if len(t) == 3 and not l.startswith("%"):
            cur.append([float(v) for v in t])
        elif l.startswith("write") and cur:
            lines.append(np.array(cur)); cur = []
    return lines


@pytest.fixture(scope="module")
def standard_run(png, tmp_path_factory):
    out = tmp_path_factory.mktemp("std")
    prog = []
    rep = jg.generate_job_files(settings(png, out), fast_recipe(), progress=lambda *a: prog.append(a), log=lambda m: None)
    return out, rep, prog


def test_standard_writes_the_descriptive_package(standard_run):
    out, rep, prog = standard_run
    names = sorted(p.name for p in out.iterdir())
    assert names == ["mask.png", "mask.recipe", "mask_data.gwl", "mask_files", "mask_job.gwl", "mask_job.recipe"]
    assert rep["fields"] == 4                                             # 120 um / 100 um fields -> 2 x 2
    assert sorted(p.name for p in (out / "mask_files").iterdir()) == [f"mask_{i}_{j}.gwl" for i in (0, 1) for j in (0, 1)]
    assert not (out / "work").exists()                                     # intermediate files are not kept
    assert prog[0][0] == "Slicing fields" and prog[-1][:3] == ("Writing GWL files", 4, 4)


def test_job_file_follows_the_descibe_template_and_the_recipe(standard_run):
    out, _, _ = standard_run
    assert active(out / "mask_job.gwl") == [
        "InvertZAxis 1", "GalvoScanMode", "ContinuousMode", "PiezoSettlingTime 10", "GalvoAcceleration 2",
        "StageVelocity 200", "XOffset 0", "YOffset 0", "ZOffset 0", "PowerScaling 1.0",
        "var $solidLaserPower = 57.5", "var $solidScanSpeed = 40000", "var $baseLaserPower = $solidLaserPower",
        "var $baseScanSpeed = $solidScanSpeed", "var $interfacePos = 0.5", "include mask_data.gwl"]


def test_data_file_structure(standard_run):
    out, _, _ = standard_run
    a = active(out / "mask_data.gwl")
    assert a[:4] == ["MoveStageY -100", "MoveStageX -100", "MoveStageY 50", "MoveStageX 50"]   # corner, then half a field
    includes = [l for l in a if l.startswith("include")]
    assert includes == [f"include mask_files\\mask_{i}_{j}.gwl" for j in (0, 1) for i in (0, 1)]   # row by row
    assert a.count("FindInterfaceAt $interfacePos") == 4
    assert "MoveStageX -200" not in a and "MoveStageX 100" in a
    assert "% FIELD 1|0" in (out / "mask_data.gwl").read_text()


def test_recipe_files_have_the_descibe_keys(standard_run):
    out, _, _ = standard_run
    assert "Doe.LevelCount = 3" in (out / "mask.recipe").read_text()
    job = (out / "mask_job.recipe").read_text()
    for key in ("Splitting.Mode = Rectangular", "Splitting.BlockSize = X:100 Y:100 Z:50", "Exposure.CoreLaserPower = 57.5",
                "Exposure.CoreScanSpeed = 40000", "Slicing.DistanceMax = 0.2", "Filling.HatchingDistance = 0.5"):
        assert key in job, key


def test_field_files_hold_local_coordinates(standard_run):
    out, _, _ = standard_run
    for f in (out / "mask_files").iterdir():
        lines = read_field(f)
        pts = np.vstack(lines)
        assert np.abs(pts[:, :2]).max() <= 50.0 + 1e-6                     # within the 100 um field around its centre
        assert pts[:, 2].min() == 0 and pts[:, 2].max() <= 0.4 + 1e-9
        assert all(np.allclose(l[:, 1], l[0, 1]) for l in lines)           # recipe hatch angle 0 -> horizontal lines


def test_written_lines_reproduce_the_design(standard_run):
    """Rebuild the height from the field files + stage moves and compare it with the input levels."""
    out, _, _ = standard_run
    n, cell = 120, 1.0
    top = np.zeros((n, n))
    sx = sy = 0.0
    for l in (out / "mask_data.gwl").read_text(encoding="utf-8").replace("\r", "").split("\n"):
        m = re.match(r"MoveStage([XY]) (\S+)", l)
        if m:
            if m.group(1) == "X": sx += float(m.group(2))
            else: sy += float(m.group(2))
        m = re.match(r"include \S+\\(\S+)", l)
        if m:
            for s in read_field(out / "mask_files" / m.group(1)):
                for a, b in zip(s[:-1], s[1:]):
                    t = np.linspace(0, 1, 60)[:, None]
                    q = a + (b - a) * t
                    c = np.floor((q[:, 0] + sx + n / 2) / cell).astype(int)
                    r = np.floor((q[:, 1] + sy + n / 2) / cell).astype(int)
                    ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
                    np.maximum.at(top, (r[ok], c[ok]), q[ok, 2] + 0.2)      # each layer is 0.2 um thick
    expected = (design()[::-1] * 0.2 + 0.2)                                   # row 0 of the image is +y
    inner = (slice(3, -3), slice(3, -3))
    assert np.mean(np.isclose(top[inner], expected[inner], atol=1e-6)) > 0.97


def test_recipe_values_reach_the_job_file(png, tmp_path):
    rec = fast_recipe(laser_power=61.25, scan_speed=25000, interface_pos=-1, piezo_settling_time=20, name="custom")
    jg.generate_job_files(settings(png, tmp_path), rec)
    a = active(tmp_path / "mask_job.gwl")
    assert "var $solidLaserPower = 61.25" in a and "var $solidScanSpeed = 25000" in a
    assert "var $interfacePos = -1" in a and "PiezoSettlingTime 20" in a
    assert "FindInterfaceAt $interfacePos" not in active(tmp_path / "mask_data.gwl")        # interface search off


def test_adaptive_method(png, tmp_path):
    rep = jg.generate_job_files(settings(png, tmp_path, method="adaptive"), fast_recipe())
    assert rep["method"] == "adaptive" and rep["fields"] >= 1
    files = sorted(p.name for p in (tmp_path / "mask_files").iterdir())
    assert files == [f"mask_{i:05d}.gwl" for i in range(rep["fields"])]    # fields are numbered in writing order
    data = (tmp_path / "mask_data.gwl").read_text(encoding="utf-8")
    assert "Adaptive stitching" in data and "% FIELD 0" in data and "|" not in data.split("MoveStageY")[1]
    assert "Splitting.Mode = Adaptive" in (tmp_path / "mask_job.recipe").read_text()


def test_design_smaller_than_one_field(tmp_path):
    """Fewer than 4 centres: no Voronoi diagram is possible, the job must still be written (both methods)."""
    Image.fromarray((design(40)[:, :] * 100).astype(np.uint8)).save(tmp_path / "tiny.png")
    for method in ("standard", "adaptive"):
        rep = jg.generate_job_files(settings(tmp_path / "tiny.png", tmp_path / method, method=method), fast_recipe())
        assert rep["fields"] == 1, method
        pts = np.vstack(read_field(tmp_path / method / "tiny_files" / next(iter((tmp_path / method / "tiny_files").iterdir())).name))
        assert pts[:, 0].max() - pts[:, 0].min() > 35                              # the whole 40 um design is written


def test_keep_intermediate_files(png, tmp_path):
    jg.generate_job_files(settings(png, tmp_path, keep_intermediate=True), fast_recipe())
    assert any((tmp_path / "work" / "h5").glob("block_*.h5"))


def test_npy_input_without_levels(tmp_path):
    np.save(tmp_path / "h.npy", design().astype(float) * 0.2e-6)         # metres
    js = jg.JobSettings(input_file=str(tmp_path / "h.npy"), output_dir=str(tmp_path / "o"), pixel_size_um=1.0,
                        height_max_um=None, levels=None, value_unit="m", base_um=0.2)
    rep = jg.generate_job_files(js, fast_recipe())
    assert rep["fields"] == 4
    assert not (tmp_path / "o" / "h.png").exists()                       # model image / DOE recipes only for level maps


def test_cancel(png, tmp_path):
    with pytest.raises(jg.JobCancelled):
        jg.generate_job_files(settings(png, tmp_path), fast_recipe(), cancel=lambda: True)


def test_bad_settings(png, tmp_path):
    rec = fast_recipe()
    for kw, msg in ((dict(method="turbo"), "method"), (dict(pixel_size_um=None), "pixel size"),
                    (dict(height_max_um=None), "height max"), (dict(output_dir=""), "output"),
                    (dict(input_file=str(tmp_path / "missing.png")), "not found"), (dict(base_um=0.0), "zero height")):
        with pytest.raises(ValueError, match=msg):
            jg.generate_job_files(settings(png, tmp_path, **kw), rec)
    bad = fast_recipe(); bad.scan_speed = -1
    with pytest.raises(ValueError):
        jg.generate_job_files(settings(png, tmp_path), bad)


def test_scan_field_warning(png, tmp_path):
    msgs = []
    jg.generate_job_files(settings(png, tmp_path), fast_recipe(scan_field_diameter=120), log=msgs.append)
    assert any("WARNING" in m for m in msgs)


def test_stl_input(tmp_path):
    pytest.importorskip("stl")
    import height_map_stl as hms
    np.save(tmp_path / "d.npy", design(60).astype(float))
    hms.height_map_to_stl(tmp_path / "d.npy", tmp_path / "d.stl", pixel_size_um=1.0, height_max_um=0.4, levels=3, base_um=0.2)
    js = jg.JobSettings(input_file=str(tmp_path / "d.stl"), output_dir=str(tmp_path / "o"), method="adaptive")
    rep = jg.generate_job_files(js, fast_recipe(slicing_distance=0.2, hatching_distance=0.5))
    assert rep["fields"] >= 1 and (tmp_path / "o" / "d_job.gwl").is_file()


def test_heights_handed_over_from_the_quantizer_match_the_file_route(png, tmp_path):
    """The Quantizer's heights in memory give the same job as reading the exported PNG."""
    from state import Design
    H = design().astype(float) * 0.2
    d = Design(source_path=str(png), origin="quantizer", heights_um=H, pixel_size_um=PITCH, levels=3, step_um=0.2,
               height_max_um=0.4, png_path=str(png))
    a = jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "mem"), method="standard", base_um=0.2), fast_recipe())
    b = jg.generate_job_files(settings(png, tmp_path / "file"), fast_recipe())
    assert a["fields"] == b["fields"] == 4
    for name in ("mask_data.gwl", "mask_job.gwl"):
        assert [l for l in active(tmp_path / "mem" / name) if "Path" not in l] == [l for l in active(tmp_path / "file" / name) if "Path" not in l]
    for i in (0, 1):
        for j in (0, 1):
            fa = np.vstack(read_field(tmp_path / "mem" / "mask_files" / f"mask_{i}_{j}.gwl"))
            fb = np.vstack(read_field(tmp_path / "file" / "mask_files" / f"mask_{i}_{j}.gwl"))
            assert np.allclose(fa, fb)
    assert (tmp_path / "mem" / "mask.png").read_bytes() == Path(png).read_bytes()       # the exported PNG is the model image


def test_handed_over_heights_without_a_png_write_a_model_image(tmp_path):
    from state import Design
    d = Design(source_path=str(tmp_path / "q.npy"), origin="quantizer", heights_um=design().astype(float) * 0.2,
               pixel_size_um=PITCH, levels=3, step_um=0.2, height_max_um=0.4)
    rep = jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "o")), fast_recipe())
    img = np.asarray(Image.open(tmp_path / "o" / "q.png"))
    assert rep["fields"] == 4 and sorted(np.unique(img)) == [0, 127, 255] or sorted(np.unique(img)) == [0, 128, 255]


def test_handed_over_heights_need_a_pixel_size_and_base(tmp_path):
    from state import Design
    d = Design(source_path="q.npy", origin="quantizer", heights_um=design().astype(float) * 0.2, pixel_size_um=None)
    with pytest.raises(ValueError, match="pixel size"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path)), fast_recipe())
    d.pixel_size_um = 1.0
    with pytest.raises(ValueError, match="zero height"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), base_um=0.0), fast_recipe())


def test_adaptive_job_also_writes_the_two_sliced_pngs(png, tmp_path):
    rep = jg.generate_job_files(settings(png, tmp_path, method="adaptive"), fast_recipe())
    std, ada = tmp_path / "mask_sliced_standard_square.png", tmp_path / "mask_sliced_adaptive.png"
    assert std.is_file() and ada.is_file() and str(std) in rep["files"] and str(ada) in rep["files"]
    a, b = Image.open(std), Image.open(ada)
    assert a.size == b.size and min(a.size) >= 1000 and a.size[0] == a.size[1] and a.mode == b.mode == "RGB"   # drawn on a fine grid
    assert np.asarray(a).tobytes() != np.asarray(b).tobytes()                       # two different layouts
    import slicer_preview as sp
    fp = sp.design_footprint(design().astype(float) * 0.2, PITCH, 0.2, min_side=1000)
    expected = sp.render(sp.compute_layout(fp, "standard", fast_recipe()), 0.0)     # the standard one is the 2 x 2 grid
    assert np.array_equal(np.asarray(a), np.asarray(expected))


def test_adaptive_png_shows_the_centres_the_job_used(png, tmp_path):
    jg.generate_job_files(settings(png, tmp_path, method="adaptive"), fast_recipe())
    data = (tmp_path / "mask_data.gwl").read_text(encoding="utf-8")
    n = len(re.findall(r"^% FIELD", data, flags=re.M))
    a = np.asarray(Image.open(tmp_path / "mask_sliced_adaptive.png"))
    first = (a == np.array((60, 220, 90))).all(axis=2)                                # marker of the first field
    last = (a == np.array((230, 60, 230))).all(axis=2)
    assert n >= 2 and first.any() and last.any()


def test_standard_job_and_opt_out_write_no_sliced_pngs(png, tmp_path):
    jg.generate_job_files(settings(png, tmp_path / "s", method="standard"), fast_recipe())
    assert not list((tmp_path / "s").glob("*sliced*"))
    jg.generate_job_files(settings(png, tmp_path / "a", method="adaptive", export_sliced_png=False), fast_recipe())
    assert not list((tmp_path / "a").glob("*sliced*"))


def test_adaptive_stl_job_writes_the_two_sliced_pngs(tmp_path):
    pytest.importorskip("stl")
    import height_map_stl as hms
    np.save(tmp_path / "d.npy", design(60).astype(float))
    hms.height_map_to_stl(tmp_path / "d.npy", tmp_path / "d.stl", pixel_size_um=1.0, height_max_um=0.4, levels=3, base_um=0.2)
    js = jg.JobSettings(input_file=str(tmp_path / "d.stl"), output_dir=str(tmp_path / "o"), method="adaptive")
    jg.generate_job_files(js, fast_recipe())
    assert (tmp_path / "o" / "d_sliced_standard_square.png").is_file() and (tmp_path / "o" / "d_sliced_adaptive.png").is_file()


# ---------------------------------------------------------------- height-based adaptive slicing
def bump_design(n=300):
    """A flat base with two raised bumps (heights in um, base to be added)."""
    y, x = np.mgrid[0:n, 0:n]
    h = np.zeros((n, n))
    h[((x - 80) ** 2 + (y - 100) ** 2) < 30 ** 2] = 0.4
    h[((x - 210) ** 2 + (y - 190) ** 2) < 25 ** 2] = 0.8
    return h


def test_z_levels():
    assert np.allclose(jg.z_levels(0, 1.0, 0.2), np.arange(0, 1.0, 0.2))                       # off: uniform
    assert np.allclose(jg.z_levels(0, 1.0, 0.2, seg=0.4, coarse=0.0), np.arange(0, 1.0, 0.2))  # no coarse distance: uniform
    z = jg.z_levels(0, 3.0, 0.5, seg=2.0, coarse=1.0)
    assert z.tolist() == [0, 1, 2, 2.5]                                                          # coarse below 2.0, fine from it
    assert jg.z_levels(1.0, 3.0, 0.5, seg=0.5, coarse=1.0).tolist() == [1, 1.5, 2, 2.5]        # segmentation below the range
    assert np.all(np.diff(jg.z_levels(0, 50, 0.1, seg=20, coarse=2.0)) > 0)


def test_height_based_centres_follow_the_features_and_still_cover_everything():
    from slicer_preview import design_footprint
    ts = jg.tipslicer()
    h = bump_design()
    fp = design_footprint(h, 1.0, base_um=0.2, segment_height_um=0.3)                      # features: the two bumps
    rec = default_recipe()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        plain, _ = jg.height_map_layout(ts, fp.T, fp.X, fp.Y, 1.0, "adaptive", 100.0, rec.adaptive_radius, tmp, jg._Ctx())
        based, _ = jg.height_map_layout(ts, fp.T, fp.X, fp.Y, 1.0, "adaptive", 100.0, rec.adaptive_radius, tmp, jg._Ctx(), T_seg=fp.T_seg)
    assert not np.array_equal(np.sort(plain, axis=0), np.sort(based, axis=0))              # a different plan
    # every pixel of the footprint is within the write radius of a centre (the base is covered too)
    d = np.sqrt(((np.stack((fp.X, fp.Y), -1)[..., None, :] - based[None, None]) ** 2).sum(-1)).min(-1)
    assert d.max() <= rec.adaptive_radius + 1.5
    # the bump centres are inside a field centred near the bump (the features were what the voting saw)
    for bx, by in ((80 - 150, 100 - 150), (210 - 150, 190 - 150)):
        assert np.sqrt(((based - [bx, by]) ** 2).sum(1)).min() < 40


def test_height_based_needs_material_at_that_height():
    ts = jg.tipslicer()
    T = np.ones((50, 50), np.uint8)
    X, Y = np.meshgrid(np.arange(50.0), np.arange(50.0))
    with pytest.raises(ValueError, match="no material"):
        jg.height_based_centres(ts, T, np.zeros_like(T), X, Y, 1.0, 20.0)


def height_job(tmp_path, **kw):
    """Adaptive job on the bump design with a thick base: planes 0, 1 (coarse) then 2.0, 2.2 ... (fine)."""
    from state import Design
    h = bump_design(240)
    d = Design(source_path=str(tmp_path / "bumps.npy"), origin="quantizer", heights_um=h, pixel_size_um=1.0, levels=3,
               step_um=0.4, height_max_um=0.8)
    rec = fast_recipe(slicing_distance=0.2, base_slicing_distance=1.0, base_power_multiplier=3.0)
    js = jg.settings_from_design(d, str(tmp_path / "o"), method="adaptive", base_um=2.0, segment_height_um=2.0,
                                 **{k: v for k, v in kw.items() if k != "recipe"})
    return jg.generate_job_files(js, kw.get("recipe", rec)), tmp_path / "o"


def test_height_based_job_slices_coarse_below_and_fine_above(tmp_path):
    rep, out = height_job(tmp_path)
    zs = set()
    for f in (out / "bumps_files").iterdir():
        for line in read_field(f):
            zs.update(np.round(line[:, 2], 3).tolist())
    assert {0.0, 1.0, 2.0}.issubset(zs) and not zs & {0.2, 0.4, 0.6, 1.2}                  # coarse planes below z = 2
    assert zs & {2.2, 2.4, 2.6}                                                              # fine planes above it
    notes = (out / "bumps_job.gwl").read_text()
    assert "Height-based segmentation" in notes and "z = 2 um" in notes and "x3" in notes


def test_height_based_power_multiplier_in_the_field_files(tmp_path):
    rep, out = height_job(tmp_path)
    text = (out / "bumps_files" / sorted(p.name for p in (out / "bumps_files").iterdir())[0]).read_text().replace("\r", "")
    assert text.count("MultLaserPower 3.00") == 1 and text.count("MultLaserPower 1.00") == 1
    assert text.index("MultLaserPower 3.00") < text.index("MultLaserPower 1.00")            # x3 on the base, x1 from z = 2 on
    first_point = re.search(r"^[-\d.]+ [-\d.]+ [-\d.]+$", text, re.M).start()
    assert text.index("MultLaserPower 3.00") < first_point


def test_without_height_based_nothing_changes(tmp_path):
    from state import Design
    d = Design(source_path=str(tmp_path / "b.npy"), origin="quantizer", heights_um=bump_design(240), pixel_size_um=1.0, levels=3)
    rec = fast_recipe(base_slicing_distance=1.0, base_power_multiplier=3.0)                  # set, but the switch is off
    jg.generate_job_files(jg.settings_from_design(d, str(tmp_path / "o"), method="adaptive", base_um=2.0), rec)
    for f in (tmp_path / "o" / "b_files").iterdir():
        assert "MultLaserPower" not in f.read_text()
    zs = {round(z, 3) for f in (tmp_path / "o" / "b_files").iterdir() for line in read_field(f) for z in line[:, 2]}
    assert 0.2 in zs and 1.0 not in zs - {0.2 * k for k in range(60)}                        # uniform 0.2 planes


def test_height_based_validation(tmp_path):
    from state import Design
    d = Design(source_path=str(tmp_path / "b.npy"), origin="quantizer", heights_um=bump_design(100), pixel_size_um=1.0)
    with pytest.raises(ValueError, match="adaptive"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="standard", segment_height_um=0.5), fast_recipe())
    with pytest.raises(ValueError, match="higher than"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="adaptive", segment_height_um=5.0), fast_recipe())
    with pytest.raises(ValueError, match=">= 0"):
        jg.generate_job_files(jg.settings_from_design(d, str(tmp_path), method="adaptive", segment_height_um=-1.0), fast_recipe())


def test_height_based_adaptive_png_shows_the_job_centres(tmp_path):
    rep, out = height_job(tmp_path)
    assert (out / "bumps_sliced_adaptive.png").is_file() and (out / "bumps_sliced_standard_square.png").is_file()


# ---- single interface search (FindInterfaceAt only before the first field)
def non_interface(path):
    return [l for l in active(path) if not l.startswith("FindInterfaceAt")]


def test_default_keeps_the_interface_call_before_every_field(standard_run):
    out, rep, _ = standard_run
    assert active(out / "mask_data.gwl").count("FindInterfaceAt $interfacePos") == 4
    assert rep["interface_calls"] == 4 and "global_interface_found" not in (out / "mask_job.gwl").read_text()


def test_global_interface_job_has_one_call_and_same_moves(standard_run, png, tmp_path):
    out0, _, _ = standard_run
    js = settings(png, tmp_path / "one", name="mask", global_interface_found=True)
    rep = jg.generate_job_files(js, fast_recipe())
    a = active(tmp_path / "one" / "mask_data.gwl")
    assert a.count("FindInterfaceAt $interfacePos") == 1 and rep["interface_calls"] == 1
    first_field = a.index("include mask_files\\mask_0_0.gwl")
    assert a.index("FindInterfaceAt $interfacePos") < first_field                  # before the first field
    assert non_interface(tmp_path / "one" / "mask_data.gwl") == non_interface(out0 / "mask_data.gwl")   # order and moves
    assert "var $interfacePos = 0.5" in (tmp_path / "one" / "mask_job.gwl").read_text()                # the one position
    assert "global_interface_found" in (tmp_path / "one" / "mask_job.gwl").read_text()
    for f in sorted((out0 / "mask_files").iterdir()):                              # field files untouched
        assert f.read_bytes() == (tmp_path / "one" / "mask_files" / f.name).read_bytes()


def test_postprocess_matches_the_generated_global_interface_version(standard_run, png, tmp_path):
    out0, _, _ = standard_run
    jg.generate_job_files(settings(png, tmp_path / "one", name="mask", global_interface_found=True), fast_recipe())
    src = tmp_path / "mask_data.gwl"
    src.write_bytes((out0 / "mask_data.gwl").read_bytes())
    assert jg.global_interface_file(src, tmp_path / "post.gwl") == 4
    assert active(tmp_path / "post.gwl") == active(tmp_path / "one" / "mask_data.gwl")
    assert src.read_bytes() == (out0 / "mask_data.gwl").read_bytes()               # source untouched with an output path
    assert b"\r\n" in (tmp_path / "post.gwl").read_bytes() and b"\n" not in (tmp_path / "post.gwl").read_bytes().replace(b"\r\n", b"")


def test_postprocess_remove_and_different_arguments():
    text = "A\r\nFindInterfaceAt 0.5\r\nM 1\r\nFindInterfaceAt $x\r\nM 2\r\n  findinterfaceat 3\r\n"
    new, n = jg.global_interface_gwl(text, comment=False)
    assert n == 3 and new == "A\r\nFindInterfaceAt 0.5\r\nM 1\r\nM 2\r\n"
    new, n = jg.global_interface_gwl(text)
    assert [l for l in new.splitlines() if l.startswith("FindInterfaceAt")] == ["FindInterfaceAt 0.5"]
    assert new.lower().count("% findinterfaceat") == 2 and new.splitlines()[1] == "FindInterfaceAt 0.5"
    assert jg.global_interface_gwl("M 1\r\n") == ("M 1\r\n", 0)


def test_interface_off_writes_no_calls(png, tmp_path):
    rep = jg.generate_job_files(settings(png, tmp_path / "o", name="mask", global_interface_found=True),
                                fast_recipe(interface_pos=-1))
    assert "FindInterfaceAt" not in "\n".join(active(tmp_path / "o" / "mask_data.gwl")) and rep["interface_calls"] == 0


# ---- block shear angle (DeScribe: the block faces lean from the substrate normal, the seams are inclined)
def test_shear_vector_follows_angle_and_direction():
    r = default_recipe()
    assert jg.shear_vector(r) is None                                   # 0 degrees = vertical seams = today's output
    r.block_shear_angle = 30.0
    kx, ky = jg.shear_vector(r)
    assert kx == pytest.approx(np.tan(np.radians(30))) and ky == pytest.approx(0, abs=1e-12)
    r.block_shear_direction = 90.0
    kx, ky = jg.shear_vector(r)
    assert kx == pytest.approx(0, abs=1e-12) and ky == pytest.approx(np.tan(np.radians(30)))


@pytest.mark.parametrize("bad", [dict(block_shear_angle=-1), dict(block_shear_angle=61), dict(block_shear_direction=400)])
def test_shear_settings_are_validated(bad):
    r = default_recipe()
    for k, v in bad.items():
        setattr(r, k, v)
    with pytest.raises(ValueError):
        r.validate()


def test_sheared_blocks_still_tile_the_plane_and_the_seam_leans():
    centres = [(0, 0), (100, 0), (0, 100), (100, 100)]
    X, Y = np.meshgrid(np.linspace(-60, 160, 221), np.linspace(-60, 160, 221))
    k = (np.tan(np.radians(20)), 0.0)

    def mask(i, dx, dy):
        return jg.in_field(centres[i][0], centres[i][1], X - dx, Y - dy, 50.0)

    for z in (0.0, 5.0, 30.0):
        masks = [jg.sheared_mask_fn(lambda dx, dy, i=i: mask(i, dx, dy), k, 0.0)(z) for i in range(4)]
        total = sum(m.astype(int) for m in masks)
        assert total.max() <= 1                                          # no overlap
        inside = (X >= -50 + k[0] * z) & (X < 150 + k[0] * z) & (Y >= -50) & (Y < 150)
        assert np.array_equal(total.astype(bool), inside)                # no gap: the whole (sheared) area is covered
    # the seam between block 0 and block 1 sits at x = 50 + z tan(angle)
    m0 = jg.sheared_mask_fn(lambda dx, dy: mask(0, dx, dy), k, 0.0)(30.0)
    edge = X[0][m0[100]].max()
    assert edge == pytest.approx(50 + 30 * k[0], abs=1.01)


def test_shear_job_keeps_the_fields_and_order_and_notes_the_angle(png, tmp_path):
    kw = dict(height_max_um=30.0)
    rec_flat = fast_recipe(slicing_distance=2.0)
    rec_lean = fast_recipe(slicing_distance=2.0, block_shear_angle=20.0, block_shear_direction=0.0)
    a = jg.generate_job_files(settings(png, tmp_path / "flat", name="mask", **kw), rec_flat)
    b = jg.generate_job_files(settings(png, tmp_path / "lean", name="mask", **kw), rec_lean)
    assert a["fields"] == b["fields"] == 4
    assert active(tmp_path / "flat" / "mask_data.gwl") == active(tmp_path / "lean" / "mask_data.gwl")   # fields and order unchanged
    notes = (tmp_path / "lean" / "mask_job.gwl").read_text()
    assert "Block shear angle: 20 deg" in notes and "Block shear" not in (tmp_path / "flat" / "mask_job.gwl").read_text()
    # the same structure is written, only divided differently: the total hatch length agrees closely
    assert b["print_hours"] == pytest.approx(a["print_hours"], rel=0.05)
    # and the leaning seams really changed the field files
    assert any((tmp_path / "flat" / "mask_files" / f.name).read_bytes() != f.read_bytes()
               for f in (tmp_path / "lean" / "mask_files").iterdir())


def test_shear_zero_is_byte_identical(png, tmp_path):
    a = jg.generate_job_files(settings(png, tmp_path / "a", name="mask"), fast_recipe())
    b = jg.generate_job_files(settings(png, tmp_path / "b", name="mask"), fast_recipe(block_shear_angle=0.0, block_shear_direction=45.0))
    for f in sorted((tmp_path / "a").rglob("*.gwl")):
        assert f.read_bytes() == (tmp_path / "b" / f.relative_to(tmp_path / "a")).read_bytes()


def test_shear_works_on_an_stl_too(tmp_path):
    pytest.importorskip("stl")
    import height_map_stl as hms
    np.save(tmp_path / "d.npy", np.tile((np.arange(240) // 40 % 6) * 2.0, (240, 1)))              # steps up to 10 um
    hms.height_map_to_stl(tmp_path / "d.npy", tmp_path / "d.stl", pixel_size_um=1.0, height_max_um=10.0, levels=6, base_um=0.2)
    out = {}
    for name, ang in (("flat", 0.0), ("lean", 25.0)):
        rec = fast_recipe(slicing_distance=1.0, block_shear_angle=ang)
        js = jg.JobSettings(input_file=str(tmp_path / "d.stl"), output_dir=str(tmp_path / name), method="standard")
        out[name] = jg.generate_job_files(js, rec)
    assert out["flat"]["fields"] == out["lean"]["fields"] >= 4
    assert out["lean"]["print_hours"] == pytest.approx(out["flat"]["print_hours"], rel=0.06)
    assert "Block shear angle: 25 deg" in next((tmp_path / "lean").glob("*_job.gwl")).read_text()
    assert any(a.read_bytes() != b.read_bytes() for a, b in zip(sorted((tmp_path / "flat").rglob("*_files/*.gwl")),
                                                                 sorted((tmp_path / "lean").rglob("*_files/*.gwl"))))


def test_stl_fields_of_a_stepped_solid_are_not_empty(tmp_path):
    """A thin plateau-like solid used to give empty slices (TipSlicer's in-fill eroded its seeds away)."""
    pytest.importorskip("stl")
    import height_map_stl as hms
    np.save(tmp_path / "d.npy", np.tile((np.arange(240) // 40 % 6) * 2.0, (240, 1)))
    hms.height_map_to_stl(tmp_path / "d.npy", tmp_path / "d.stl", pixel_size_um=1.0, height_max_um=10.0, levels=6, base_um=0.2)
    rep = jg.generate_job_files(jg.JobSettings(input_file=str(tmp_path / "d.stl"), output_dir=str(tmp_path / "o"), method="standard"),
                                fast_recipe(slicing_distance=1.0))
    sizes = [f.stat().st_size for f in (tmp_path / "o").glob("*_files/*.gwl")]
    assert len(sizes) == rep["fields"] and min(sizes) > 1000, sizes          # every field has hatch lines
    assert rep["print_hours"] > 0


def test_cli_block_shear_flags(png, tmp_path):
    import nanoslicer as ns
    ns.cli(["job", str(png), str(tmp_path / "o"), "--pixel-size", "1", "--height-max", "30", "--levels", "3", "--name", "mask",
            "--set", "hatching_distance=0.5", "--set", "slicing_distance=2", "--block-shear-angle", "20", "--block-shear-direction", "90"])
    note = (tmp_path / "o" / "mask_job.gwl").read_text()
    assert "Block shear angle: 20 deg" in note and "towards 90 deg" in note
    with pytest.raises(SystemExit):
        ns.cli(["job", str(png), str(tmp_path / "bad"), "--pixel-size", "1", "--height-max", "1", "--levels", "3",
                "--block-shear-angle", "75"])
