"""GUI tests; skipped where there is no display / tkinter."""
import time

import numpy as np
import pytest
from PIL import Image

tk = pytest.importorskip("tkinter")
pytest.importorskip("matplotlib")

import height_map_stl as hms  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSLICER_HOME", str(tmp_path / "home"))
    import nanoslicer as ns
    try:
        a = ns.NanoSlicerApp()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    a.update()
    yield a
    a.destroy()


def pump(app, cond, timeout=60):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.update()
        time.sleep(0.02)
    return cond()


def load_example(app, tmp_path, n=120):
    """Put a small phase image into the Quantizer tab (pixel size 1 um) and wait for its design."""
    y, x = np.mgrid[0:n, 0:n]
    phase = (((x - n / 2) ** 2 + (y - n / 2) ** 2) / 200.0) % 1.0                 # radial ramp, 0 ... 1 of 2*pi
    p = tmp_path / "phase.png"
    Image.fromarray(np.round(phase * 255).astype(np.uint8)).save(p)
    q = app.quantizer
    q.v["pitch"].set("1"); q.v["input"].set(str(p)); app.update()
    assert pump(app, lambda: q.after_id is None and q.design is not None)        # the debounced preview has run: the design is settled
    return q


def test_name_and_two_tabs(app):
    assert app.title() == "NanoSlicer"
    assert [app.tabs.tab(t, "text").strip() for t in app.tabs.tabs()] == ["Quantizer", "Job File Generator"]


def test_quantizer_tab_is_the_original_ui(app):
    q = app.quantizer
    assert q.v["wavelength"].get() == "532" and q.v["step_nm"].get() == "200" and q.v["output"].get() == "both"


def test_recipe_panel(app):
    j = app.jobs
    assert j.recipe_box.get() == "63x - DIP" and j.v["r_laser_power"].get() == "57.5"
    j.v["r_laser_power"].set("60"); app.update()
    assert "modified" in j.recipe_state.cget("text")
    rec = j._fields_to_recipe(); rec.name = "mine"
    j.store.save(rec); j.select_recipe("mine"); app.update()
    assert j.recipe_box.get() == "mine" and j.v["r_laser_power"].get() == "60"
    assert "modified" not in j.recipe_state.cget("text")
    j.store.delete("mine"); j.select_recipe("63x - DIP"); app.update()
    assert j.recipe_box["values"] == ("63x - DIP",)
    j.v["r_scan_speed"].set("fast")
    with pytest.raises(ValueError):
        j._fields_to_recipe()


def test_job_tab_starts_empty_and_asks_for_a_structure(app):
    j = app.jobs
    assert j.view.img is None and "Send to Job File Generator" in j.view.message
    assert str(j.rb_quant.cget("state")) == "disabled"                      # nothing from the Quantizer yet


def test_quantizer_only_workflow_needs_no_job_tab(app, tmp_path):
    q = load_example(app, tmp_path)
    assert q.design["q"].max() >= 1 and app.state.design is None             # nothing was sent
    q.update_preview(); app.update()
    assert q.panels[1].arr is not None                                       # quantized preview is shown


def test_send_to_job_generator_keeps_everything(app, tmp_path):
    q = load_example(app, tmp_path)
    q.send_to_job(); app.update()
    d, j = app.state.design, app.jobs
    assert app.tabs.select() == str(j)                                       # the tab is shown
    assert d.origin == "quantizer" and d.source_path == str(tmp_path / "phase.png")
    assert np.allclose(d.heights_um, q.design["heights"] * 1e6)              # the quantized height map
    assert d.pixel_size_um == 1.0 and d.footprint_um == (120.0, 120.0)       # pixel size, physical dimensions
    assert d.levels == int(q.design["q"].max()) + 1 and d.step_um == pytest.approx(q.design["step"] * 1e6)   # levels, height scale
    assert d.height_max_um == pytest.approx(q.design["heights"].max() * 1e6)
    assert j.v["source"].get() == "quantizer" and str(j.rb_quant.cget("state")) == "normal"
    assert j.v["pitch"].get() == "1" and j.v["levels"].get() == str(d.levels)          # fields only mirror the state
    assert "120 × 120 µm" in j.quant_info.cget("text") and "not exported yet" in j.quant_info.cget("text")
    assert j.current_design() is d
    js = j._job_settings()
    assert js.heights_um is d.heights_um and js.pixel_size_um == 1.0 and js.levels == d.levels


def test_send_needs_a_pixel_size(app, tmp_path, monkeypatch):
    q = load_example(app, tmp_path)
    q.v["pitch"].set("")
    import quantizer as pq
    shown = []
    monkeypatch.setattr(pq.messagebox, "showerror", lambda *a, **k: shown.append(a))
    q.send_to_job()
    assert shown and app.state.design is None


def test_exported_files_follow_the_design(app, tmp_path):
    q = load_example(app, tmp_path)
    q.send_to_job(); app.update()
    d = app.state.design
    rep = {"files": [str(tmp_path / "a.png"), str(tmp_path / "a_continuous.png"), str(tmp_path / "a.stl")]}
    app._on_export(rep, q.design_serial)                                     # export of the same design
    assert d.png_path == str(tmp_path / "a.png") and d.stl_path == str(tmp_path / "a.stl")
    assert "a.stl" in app.jobs.quant_info.cget("text")
    app._on_export({"files": [str(tmp_path / "old.stl")]}, q.design_serial - 1)   # a stale export is ignored
    assert d.stl_path == str(tmp_path / "a.stl")


def test_standard_preview_after_sending(app, tmp_path):
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    assert pump(app, lambda: j.view.img is not None)
    assert "standard: 4 rectangular fields of 100 µm" in j.view.caption.cget("text")
    assert j.view.img.size == (120, 120)


def test_preview_follows_method_field_size_and_overlap(app, tmp_path):
    cv2 = pytest.importorskip("cv2"); pytest.importorskip("tables")
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    assert pump(app, lambda: "standard: 4" in j.view.caption.cget("text"))
    j.v["r_field_size"].set("40")                                            # a recipe parameter that changes the layout
    assert pump(app, lambda: "standard: 9 rectangular fields of 40 µm" in j.view.caption.cget("text"))
    j.v["r_overlap"].set("3")
    assert pump(app, lambda: "overlap 3 µm" in j.view.caption.cget("text"))
    j.v["method"].set("adaptive"); j._method_changed()
    assert pump(app, lambda: "adaptive:" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    j.v["r_adaptive_radius"].set("50")
    assert pump(app, lambda: "write radius 50 µm" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    assert cv2 is not None


def test_job_tab_alone_with_an_stl(app, tmp_path):
    a = np.zeros((60, 100)); a[:, 50:] = 0.2
    stl = hms.heights_to_stl(a, tmp_path / "solid.stl", pixel_size_um=2.0, base_um=0.2)["output"]
    j = app.jobs
    app.tabs.select(j)
    j.v["source"].set("file"); j.v["input"].set(stl); j._source_changed()
    assert pump(app, lambda: j.view.img is not None)
    assert "STL" in j.view.caption.cget("text") and "200 × 120 µm" in j.view.caption.cget("text")
    assert str(j.design_rows[0][2].cget("state")) == "disabled"             # the solid defines pixel size / height
    j.v["method"].set("standard"); j._method_changed()
    j.v["r_field_size"].set("60")
    assert pump(app, lambda: "fields of 60 µm" in j.view.caption.cget("text"))


def test_job_tab_alone_with_a_height_map_file_and_generation(app, tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("tables")
    np.save(tmp_path / "h.npy", np.tile(np.repeat(np.arange(3.0), 20), (60, 1)))      # 3 plateaus
    j = app.jobs
    j.v["input"].set(str(tmp_path / "h.npy")); j.v["pitch"].set("1"); j.v["height_max"].set("0.4"); j.v["levels"].set("3")
    j.v["output_dir"].set(str(tmp_path / "job")); j.v["r_hatching_distance"].set("0.5"); app.update()
    assert pump(app, lambda: j.view.img is not None)
    assert "standard: 1 rectangular fields" in j.view.caption.cget("text") or "fields of 100" in j.view.caption.cget("text")
    j.generate()
    assert pump(app, lambda: not j.busy and "Wrote" in j.status.get(), timeout=120), j.status.get()
    assert (tmp_path / "job" / "h_job.gwl").is_file()


def test_convert_stl_from_the_quantizer_attaches_it(app, tmp_path, monkeypatch):
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    out = tmp_path / "from_quantizer.stl"
    import nanoslicer as ns
    monkeypatch.setattr(ns.filedialog, "asksaveasfilename", lambda **k: str(out))
    j.convert_stl()
    assert pump(app, lambda: not j.busy and out.exists())
    assert app.state.design.stl_path == str(out)
    assert hms.read_stl_triangles(out).shape[1:] == (3, 3)


def test_job_settings_from_file_fields(app, tmp_path):
    j = app.jobs
    f = tmp_path / "m.npy"; np.save(f, np.arange(16.0).reshape(4, 4))
    j.v["input"].set(str(f)); j.v["pitch"].set("2"); j.v["height_max"].set("1"); j.v["levels"].set("6")
    j.v["method"].set("adaptive")
    js = j._job_settings()
    assert (js.method, js.pixel_size_um, js.height_max_um, js.levels) == ("adaptive", 2.0, 1.0, 6)
    assert js.output_dir == str(tmp_path / "m_job") and js.heights_um.shape == (4, 4)
    j.v["levels"].set("2.5")
    with pytest.raises(ValueError):
        j._job_settings()


def test_preview_toggle(app, tmp_path):
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    assert j.v["show_preview"].get() is True
    assert pump(app, lambda: j.view.img is not None and "4 rectangular" in j.view.caption.cget("text"))
    j.v["show_preview"].set(False); j._toggle_preview(); app.update()
    assert j.view.img is None and "off" in j.view.message                         # nothing is shown ...
    j.v["r_field_size"].set("40")                                                  # ... and changes do not compute a preview
    app.update(); time.sleep(0.6); app.update()
    assert j.view.img is None
    j.v["show_preview"].set(True); j._toggle_preview()
    assert pump(app, lambda: j.view.img is not None and "9 rectangular fields of 40" in j.view.caption.cget("text"))


def test_segmentation_controls(app, tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("tables")
    import nanoslicer as ns
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    assert [str(c.cget("state")) for c in j.seg_row0[1].winfo_children()] == ["disabled"]           # adaptive only
    assert j._segmentation_mode() == "footprint_edges" and j._segment_height() is None
    j.v["method"].set("adaptive"); j._method_changed(); app.update()
    assert [str(c.cget("state")) for c in j.seg_row0[1].winfo_children()] == ["readonly"]
    assert all(str(c.cget("state")) == "disabled" for c in j.seg_row[1].winfo_children())            # z only for the threshold mode
    j.v["seg_mode"].set(ns.SEG_LABELS["height_threshold"]); app.update()
    assert j.v["seg_h"].get() == "0.2" and j._segment_height() == pytest.approx(0.2)                 # defaults to the base thickness
    assert pump(app, lambda: "height-based" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    js = j._job_settings()
    assert js.segment_height_um == pytest.approx(0.2) and js.segmentation_mode == "height_threshold"
    j.v["seg_h"].set("x")
    with pytest.raises(ValueError):
        j._segment_height()
    j.v["seg_h"].set("0.4")
    assert pump(app, lambda: "z = 0.4" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    j.v["seg_mode"].set(ns.SEG_LABELS["height_gradient_guided"]); app.update()                       # the new mode
    assert j._segmentation_mode() == "height_gradient_guided" and j._segment_height() is None
    assert all(str(c.cget("state")) == "disabled" for c in j.seg_row[1].winfo_children())
    assert pump(app, lambda: "height-gradient-guided" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    js = j._job_settings()
    assert js.segmentation_mode == "height_gradient_guided" and js.segment_height_um is None
    j.v["method"].set("standard"); j._method_changed(); app.update()                                  # back to standard: ignored
    assert j._segmentation_mode() == "footprint_edges" and j._job_settings().segmentation_mode == "footprint_edges"
    assert "base_slicing_distance" in [k for k, _, _ in ns.RECIPE_FIELDS]


def test_phase_quality_option_in_the_selector(app, tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("tables")
    import nanoslicer as ns
    assert "phase_quality_guided" in ns.SEG_LABELS and "phase_quality_guided" in ns.SEG_NOTES
    q = load_example(app, tmp_path)
    q.send_to_job()
    j = app.jobs
    j.v["method"].set("adaptive"); j._method_changed()
    j.v["seg_mode"].set(ns.SEG_LABELS["phase_quality_guided"]); app.update()
    assert j._segmentation_mode() == "phase_quality_guided" and j._segment_height() is None
    assert all(str(c.cget("state")) == "disabled" for c in j.seg_row[1].winfo_children())          # no height needed
    assert pump(app, lambda: "phase-quality-guided" in j.view.caption.cget("text") and not j.preview_state.cget("text"))
    assert j._job_settings().segmentation_mode == "phase_quality_guided"
    assert "minimum-cost" in j.seg_note.cget("text")


def test_preview_follows_pixel_size_and_resampling(app, tmp_path):
    q = load_example(app, tmp_path)
    q.v["resample"].set("up"); q.v["factor"].set("2"); q.v["pitch"].set("2")
    assert pump(app, lambda: q.after_id is None and q.stat["pixels"].cget("text").startswith("240 × 240"))
    assert q.stat["pitch"].cget("text") == "2 µm" and q.stat["footprint"].cget("text").startswith("480.0 µm")
    q.v["resample"].set("down"); q.v["factor"].set("4")
    assert pump(app, lambda: q.after_id is None and q.stat["pixels"].cget("text").startswith("30 × 30"))
    assert q.stat["footprint"].cget("text").startswith("60.0 µm")


def test_invalid_settings_clear_the_stale_preview(app, tmp_path):
    q = load_example(app, tmp_path)
    q.v["material"].set("air")                                    # no index step against air
    assert pump(app, lambda: q.design is None)
    assert q.stat["pixels"].cget("text") == "—" and q.status.get().startswith("Cannot update")
    assert str(q.status_label.cget("foreground")) == u_err()
    q.v["material"].set("fused_silica")
    assert pump(app, lambda: q.design is not None)                # recovers once the setting is valid


def test_update_button_rebuilds_the_preview(app, tmp_path):
    q = load_example(app, tmp_path)
    q.v["pitch"].trace_remove("write", q.v["pitch"].trace_info()[0][1])   # simulate a missed change event
    q.v["pitch"].set("3")
    app.update()
    assert q.stat["pitch"].cget("text") == "1 µm"                 # stale until updated
    q.refresh()
    assert q.stat["pitch"].cget("text") == "3 µm"


def u_err():
    import quantizer
    return quantizer.STATUS_ERR


def test_objective_and_mode_are_dropdowns_not_typed(app):
    from recipes import MODES, OBJECTIVES, RESINS
    j = app.jobs
    for key, choices in (("objective", OBJECTIVES), ("mode", MODES), ("resin", RESINS)):
        box = j.choice_boxes[key]
        assert str(box.cget("state")) == "readonly" and tuple(box.cget("values")) == choices
    assert j.v["r_objective"].get() == "63x" and j.v["r_mode"].get() == "DiLL" and j.v["r_resin"].get() == "IP-Dip2"
    j.v["r_objective"].set("25x")                                                    # picking marks the recipe modified
    assert "modified" in j.recipe_state.cget("text")
    assert j._fields_to_recipe().objective == "25x"


def test_saved_label_outside_the_list_stays_selectable(app):
    j = app.jobs
    from recipes import Recipe
    rec = Recipe(name="custom label", objective="40x", mode="my mode", resin="my resin")
    j.store.save(rec)
    j.select_recipe("custom label")
    assert j.v["r_objective"].get() == "40x" and "40x" in j.choice_boxes["objective"].cget("values")
    assert "my mode" in j.choice_boxes["mode"].cget("values") and "my resin" in j.choice_boxes["resin"].cget("values")


@pytest.mark.parametrize("tab", ["quantizer", "jobs"])
def test_advanced_section_opens_in_a_short_window(app, tab):
    """The Advanced options used to be squeezed to zero height when the column was full; now they open and scroll into view."""
    app.geometry("1180x600")
    frame = getattr(app, tab)
    app.tabs.select(frame)
    pump(app, lambda: False, timeout=0.5)
    frame._toggle_adv()
    scroll = frame.side_scroll if tab == "quantizer" else frame.left_scroll
    assert pump(app, lambda: frame.adv.winfo_height() == frame.adv.winfo_reqheight() > 20)
    pump(app, lambda: False, timeout=0.3)
    top = frame.adv.winfo_rooty() - scroll.canvas.winfo_rooty()
    assert top >= 0 and top + frame.adv.winfo_height() <= scroll.canvas.winfo_height() + 2    # fully in view
    frame._toggle_adv()
    assert pump(app, lambda: not frame.adv.winfo_viewable())
