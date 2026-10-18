"""Writing-path solvers: pyconcorde as an optional exact backend, 2-opt as the fallback. pyconcorde itself is not needed:
a stand-in `concorde.tsp` module (same interface) plays it, and can be told to fail in every way."""
import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")
import phase_quality as pqg  # noqa: E402
import route_solvers as rs  # noqa: E402


# ---------------------------------------------------------------- the stand-in for pyconcorde
class FakeSolution:
    def __init__(self, tour, success=True, hit_timebound=False):
        self.tour, self.success, self.hit_timebound, self.optimal_value = np.asarray(tour), success, hit_timebound, 0


def parse_tsplib(path):
    lines = Path(path).read_text().split("\n")
    n = int(next(l for l in lines if l.startswith("DIMENSION")).split(":")[1])
    nums = " ".join(lines[lines.index("EDGE_WEIGHT_SECTION") + 1:]).split()
    D = np.zeros((n, n), dtype=np.int64)
    it = iter(int(v) for v in nums)
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = next(it)
    return n, D


def brute_force_tour(D):
    n = len(D)
    best = min(itertools.permutations(range(1, n)), key=lambda p: D[0, p[0]] + sum(D[a, b] for a, b in zip(p, p[1:])) + D[p[-1], 0])
    return [0, *best]


def heuristic_tour(D):                                         # large instances: plumbing only, the stand-in just claims optimal
    left, tour = set(range(1, len(D))), [0]
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt); left.discard(nxt)
    return tour


class FakeSolver:
    mode = "ok"
    calls = []

    def __init__(self, path):
        self.path = path

    @classmethod
    def from_tspfile(cls, path):
        cls.calls.append(Path(path).read_text())
        return cls(path)

    def solve(self, time_bound=-1, verbose=True, **kw):
        if FakeSolver.mode == "raise":
            raise RuntimeError("concorde crashed")
        n, D = parse_tsplib(self.path)
        if FakeSolver.mode == "fail":
            return FakeSolution([], success=False)
        if FakeSolver.mode == "timeout":
            return FakeSolution(heuristic_tour(D), success=True, hit_timebound=True)
        if FakeSolver.mode == "badtour":
            return FakeSolution([0, 1, 1, 2, 3][:n] if n > 4 else [0, 0, 1], success=True)
        return FakeSolution(brute_force_tour(D) if n <= 9 else heuristic_tour(D))


@pytest.fixture
def fake_concorde(monkeypatch):
    FakeSolver.mode, FakeSolver.calls = "ok", []
    pkg, mod = types.ModuleType("concorde"), types.ModuleType("concorde.tsp")
    mod.TSPSolver = FakeSolver
    pkg.tsp = mod
    monkeypatch.setitem(sys.modules, "concorde", pkg)
    monkeypatch.setitem(sys.modules, "concorde.tsp", mod)
    return FakeSolver


def points(n=7, seed=3):
    return np.random.default_rng(seed).random((n, 2)) * 500


def optimal_open_path_length(pts):
    return min(rs.path_length(pts, p) for p in itertools.permutations(range(len(pts))))


# ---------------------------------------------------------------- the instance
def test_open_path_instance_has_a_dummy_node_and_integer_distances(tmp_path):
    pts = points(5)
    f = rs.write_open_path_tsplib(tmp_path / "x.tsp", pts)
    n, D = parse_tsplib(f)
    assert n == 6 and (D[0] == 0).all() and (D[:, 0] == 0).all()                     # node 0: distance 0 to everything
    assert np.array_equal(D, D.T)
    expect = np.rint(np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) * rs.DISTANCE_SCALE).astype(int)
    assert np.array_equal(D[1:, 1:], expect)
    text = Path(f).read_text()
    assert "EDGE_WEIGHT_TYPE: EXPLICIT" in text and "DIMENSION: 6" in text


# ---------------------------------------------------------------- pyconcorde present
def test_pyconcorde_route_is_the_optimal_open_path(fake_concorde):
    pts = points(7)
    res = rs.solve_route(pts, np.arange(7))
    assert res.solver == "pyconcorde" and res.optimal
    assert rs.valid_order(res.order, 7)
    assert res.length_um == pytest.approx(optimal_open_path_length(pts), abs=0.02)   # integer rounding of 0.01 um
    assert "optimal open path" in res.describe() and res.info()["optimal"] is True
    assert fake_concorde.calls and "DIMENSION: 8" in fake_concorde.calls[0]          # the dummy node was sent


def test_pyconcorde_is_not_longer_than_the_two_opt_route(fake_concorde):
    for seed in range(5):
        pts = points(8, seed)
        res = rs.solve_route(pts, np.random.default_rng(seed).permutation(8))
        assert res.length_um <= rs.solver_two_opt(pts, np.random.default_rng(seed).permutation(8)).length_um + 1e-9


# ---------------------------------------------------------------- fallbacks
def no_pyconcorde(monkeypatch):
    """Make `import concorde.tsp` fail, whether or not pyconcorde is installed here."""
    monkeypatch.setitem(sys.modules, "concorde", None)
    monkeypatch.setitem(sys.modules, "concorde.tsp", None)


def test_works_without_pyconcorde_installed(monkeypatch):
    no_pyconcorde(monkeypatch)
    pts = points(30)
    res = rs.solve_route(pts, np.arange(30))                                          # the real environment: not installed
    assert res.solver == "2-opt" and not res.optimal and rs.valid_order(res.order, 30)
    assert "pyconcorde" in res.detail and "not available" in res.detail


def test_fallback_when_the_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "concorde", None)                               # `import concorde...` raises ImportError
    monkeypatch.setitem(sys.modules, "concorde.tsp", None)
    res = rs.solve_route(points(10), np.arange(10))
    assert res.solver == "2-opt" and not res.optimal and "not available" in res.detail


@pytest.mark.parametrize("mode, text", [("raise", "failed"), ("fail", "did not report"), ("timeout", "time limit"),
                                        ("badtour", "invalid")])
def test_fallback_when_concorde_fails(fake_concorde, mode, text):
    fake_concorde.mode = mode
    msgs = []
    pts = points(8)
    res = rs.solve_route(pts, np.arange(8), log=msgs.append)
    assert res.solver == "2-opt" and not res.optimal and rs.valid_order(res.order, 8)
    assert text in res.detail and msgs                                               # reported, not hidden
    assert "optimal" not in res.describe().replace("not proven optimal", "")


def test_a_time_limited_run_is_never_called_optimal(fake_concorde):
    fake_concorde.mode = "timeout"
    res = rs.solve_route(points(8), np.arange(8))
    assert res.solver == "2-opt" and res.optimal is False


def test_unsuitable_instances_use_two_opt(fake_concorde):
    assert rs.solve_route(points(3), np.arange(3)).solver == "2-opt"                 # nothing to optimise
    res = rs.solve_route(points(12), np.arange(12), max_nodes=10)                    # too big for the exact solver
    assert res.solver == "2-opt" and "more than" in res.detail and not fake_concorde.calls
    assert rs.solve_route(points(1), np.arange(1)).order.tolist() == [0]


def test_fast_mode_never_calls_the_exact_solver(fake_concorde):
    res = rs.solve_route(points(8), np.arange(8), solvers=rs.ROUTE_FAST)
    assert res.solver == "2-opt" and not fake_concorde.calls


def test_the_fallback_is_exactly_the_old_two_opt_route(monkeypatch):
    no_pyconcorde(monkeypatch)
    pts = points(40)
    init = np.random.default_rng(0).permutation(40)
    res = rs.solve_route(pts, init)
    assert np.array_equal(res.order, pqg.two_opt(pts, init))                          # unchanged behaviour without pyconcorde
    assert res.length_um == pytest.approx(pqg.travel_distance(pts, res.order))


@pytest.mark.parametrize("mode", ["ok", "raise", "fail", "timeout", "badtour"])
def test_the_route_holds_every_field_exactly_once(fake_concorde, mode):
    fake_concorde.mode = mode
    for n in (4, 7, 9):
        res = rs.solve_route(points(n, n), np.random.default_rng(n).permutation(n))
        assert sorted(res.order.tolist()) == list(range(n))


def test_start_order_may_be_a_function_and_must_be_valid():
    calls = []
    res = rs.solve_route(points(6), lambda: calls.append(1) or np.arange(6)[::-1])
    assert calls == [1] and rs.valid_order(res.order, 6)
    with pytest.raises(ValueError, match="exactly once"):
        rs.solve_route(points(6), [0, 0, 1, 2, 3, 4])


def test_solvers_are_modular(monkeypatch):
    with pytest.raises(ValueError, match="unknown route solver"):
        rs.solve_route(points(6), np.arange(6), solvers=("nope",))
    monkeypatch.setitem(rs.SOLVERS, "reverse", lambda pts, init, **kw: rs.RouteResult(np.arange(len(pts)), "reverse", False,
                                                                                      rs.path_length(pts, np.arange(len(pts)))))
    rs.register_solver("mine", lambda pts, init, **kw: (_ for _ in ()).throw(rs.SolverUnavailable("no")))
    res = rs.solve_route(points(6), np.arange(6), solvers=("mine", "2-opt"))
    assert res.solver == "2-opt" and "mine: no" in res.detail
    monkeypatch.delitem(rs.SOLVERS, "mine")


# ---------------------------------------------------------------- segmentation untouched
cv2 = pytest.importorskip("cv2"); pytest.importorskip("tables"); pytest.importorskip("matplotlib")
import job_generator as jg  # noqa: E402
import slicer_preview as sp  # noqa: E402
from recipes import default_recipe  # noqa: E402
from state import Design  # noqa: E402


def stairs(n=200, width=34, levels=6):
    return np.tile((np.arange(n) // width % levels) * 0.2, (n, 1))


def design():
    return Design(source_path=str(Path("x") / "steps.npy"), origin="quantizer", heights_um=stairs(), pixel_size_um=1.0,
                  levels=6, step_um=0.2, height_max_um=1.0)


def rec():
    r = default_recipe(); r.hatching_distance = 0.5
    return r


def test_segmentation_and_masks_do_not_depend_on_the_route_solver(fake_concorde):
    fp = sp.design_footprint(stairs() , 1.0, 0.2)
    a = sp.compute_layout(fp, "adaptive", rec(), segmentation_mode="phase_quality_guided", route_solvers=rs.ROUTE_FAST)
    b = sp.compute_layout(fp, "adaptive", rec(), segmentation_mode="phase_quality_guided", route_solvers=rs.ROUTE_AUTO)
    assert a.route["solver"] == "2-opt" and b.route["solver"] in ("2-opt", "pyconcorde")
    assert np.array_equal(a.centres, b.centres) and np.array_equal(a.labels, b.labels)           # same blocks and masks
    assert sorted(a.order.tolist()) == sorted(b.order.tolist()) == list(range(len(a.centres)))   # same fields, maybe other order
    ma, mb = sp.layout_metrics({"phase": a})["phase"], sp.layout_metrics({"phase": b})["phase"]
    for key in ("blocks", "seam_length_um", "seam_cost_mean", "seam_cost_total_um", "seam_on_step", "extra_pieces",
                "tiny_islands", "disconnected_blocks"):
        assert ma[key] == mb[key], key                                                            # only the travel can differ
    assert ma["route_solver"] == "2-opt" and mb["route_solver"] in ("2-opt", "pyconcorde")


def test_the_default_route_without_pyconcorde_is_the_legacy_one(monkeypatch):
    no_pyconcorde(monkeypatch)
    fp = sp.design_footprint(stairs(), 1.0, 0.2)
    new = sp.compute_layout(fp, "adaptive", rec(), segmentation_mode="phase_quality_guided")
    ts, tmp = jg.tipslicer(), "."
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        legacy = pqg.two_opt(new.centres, jg.tsp_order(ts, new.centres, tmp, jg._Ctx()))
    assert np.array_equal(new.order, legacy)


def job(tmp_path, **kw):
    js = jg.settings_from_design(design(), str(tmp_path), method="adaptive", base_um=0.2,
                                 segmentation_mode="phase_quality_guided", **kw)
    return jg.generate_job_files(js, rec())


def test_job_reports_the_solver(tmp_path, fake_concorde):
    rep = job(tmp_path / "exact")
    assert rep["route"]["solver"] == "pyconcorde" and rep["route"]["optimal"] is True and rep["route"]["length_um"] > 0
    note = (tmp_path / "exact" / "steps_job.gwl").read_text()
    assert "Writing order: pyconcorde (proven optimal open path)" in note
    data = (tmp_path / "exact" / "steps_data.gwl").read_text(encoding="utf-8")
    assert data.count("% FIELD") == rep["fields"]                                              # every field once
    fake_concorde.mode = "fail"
    rep2 = job(tmp_path / "fallback")
    assert rep2["route"]["solver"] == "2-opt" and rep2["route"]["optimal"] is False
    assert "Writing order: 2-opt (improved, not proven optimal)" in (tmp_path / "fallback" / "steps_job.gwl").read_text()
    assert rep2["fields"] == rep["fields"]                                                     # the segmentation is the same
    labels = [np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(tmp_path / d / "steps_segmentation_labels.png")) for d in ("exact", "fallback")]
    assert np.array_equal(*labels)
    rep3 = jg.generate_job_files(jg.settings_from_design(design(), str(tmp_path / "fast"), method="adaptive", base_um=0.2,
                                                         segmentation_mode="phase_quality_guided", route_solvers=rs.ROUTE_FAST), rec())
    assert rep3["route"]["solver"] == "2-opt"


def test_every_adaptive_mode_reports_its_route_and_standard_none(tmp_path, monkeypatch):
    no_pyconcorde(monkeypatch)
    rep = jg.generate_job_files(jg.settings_from_design(design(), str(tmp_path / "a"), method="adaptive", base_um=0.2), rec())
    assert rep["route"]["solver"] == "2-opt" and "pyconcorde" in rep["route"]["detail"]          # the reason is kept
    assert "Writing order: 2-opt" in next((tmp_path / "a").glob("*_job.gwl")).read_text()
    rep = jg.generate_job_files(jg.settings_from_design(design(), str(tmp_path / "s"), method="standard", base_um=0.2), rec())
    assert rep["route"] is None                                                                  # the grid keeps its row order


def test_pyconcorde_orders_the_other_adaptive_modes_too(tmp_path, fake_concorde):
    for mode in ("footprint_edges", "height_gradient_guided"):
        rep = jg.generate_job_files(jg.settings_from_design(design(), str(tmp_path / mode), method="adaptive", base_um=0.2,
                                                            segmentation_mode=mode), rec())
        r = rep["route"]
        assert r["solver"] in ("pyconcorde", "2-opt") and (r["solver"] == "pyconcorde" and r["optimal"] or r["detail"]), mode
        assert rep["fields"] >= 4


def test_comparison_metrics_and_cli_report_the_solver(tmp_path, fake_concorde, capsys):
    import nanoslicer as ns
    np.save(tmp_path / "h.npy", stairs() + 0.0)
    common = [str(tmp_path / "h.npy"), str(tmp_path / "c.png"), "--pixel-size", "1", "--base", "0.2", "--max-side", "200",
              "--metrics", str(tmp_path / "m.json")]
    ns.cli(["compare", *common])
    out = capsys.readouterr().out
    header = next(l for l in out.splitlines() if l.startswith("mode"))
    assert header.rstrip().endswith("route") and "pyconcorde (optimal)" in out
    met = json.loads((tmp_path / "m.json").read_text())
    assert met["phase"]["route_solver"] == "pyconcorde" and met["phase"]["route_optimal"] is True
    assert met["standard"]["route_solver"] is None                                             # the grid keeps its row order
    assert all(met[k]["route_solver"] in ("pyconcorde", "2-opt") for k in met if k != "standard")   # every adaptive order is optimised
    assert met["gradient"]["route_solver"] == "pyconcorde"
    ns.cli(["compare", *common, "--route-solver", "2-opt"])
    out = capsys.readouterr().out
    assert "pyconcorde" not in out and "2-opt" in out
    assert json.loads((tmp_path / "m.json").read_text())["phase"]["route_solver"] == "2-opt"
    # the comparison image of a job shows the route the job wrote
    rep = job(tmp_path / "j")
    assert rep["route"]["solver"] == "pyconcorde" and (tmp_path / "j" / "steps_sliced_comparison.png").is_file()


def test_job_command_prints_the_writing_order(tmp_path, fake_concorde, capsys):
    import nanoslicer as ns
    np.save(tmp_path / "h.npy", stairs())
    base = ["job", str(tmp_path / "h.npy"), str(tmp_path / "o"), "--method", "adaptive", "--segmentation", "phase_quality_guided",
            "--pixel-size", "1", "--height-max", "1", "--set", "hatching_distance=0.5"]
    ns.cli(base)
    assert "writing order: pyconcorde (proven optimal open path)" in capsys.readouterr().out
    ns.cli(base[:2] + [str(tmp_path / "o2")] + base[3:] + ["--route-solver", "2-opt"])
    assert "writing order: 2-opt" in capsys.readouterr().out


def test_the_live_preview_uses_the_same_solver_as_the_job(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    monkeypatch.setenv("NANOSLICER_HOME", str(tmp_path / "home"))
    import nanoslicer as ns
    try:
        app = ns.NanoSlicerApp()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    seen = []
    real = sp.compute_layout
    monkeypatch.setattr(sp, "compute_layout", lambda *a, **k: (seen.append(k.get("route_solvers")), real(*a, **k))[1])
    try:
        app.update()
        j = app.jobs
        np.save(tmp_path / "h.npy", stairs())
        j.v["input"].set(str(tmp_path / "h.npy")); j.v["pitch"].set("1"); j.v["height_max"].set("1")
        j.v["method"].set("adaptive"); j._method_changed()
        j.v["seg_mode"].set(ns.SEG_LABELS["phase_quality_guided"])
        j.update_preview()
        import time
        t0 = time.time()
        while not seen and time.time() - t0 < 60:
            app.update(); time.sleep(0.05)
        assert seen and all(s == rs.ROUTE_AUTO for s in seen)
    finally:
        app.destroy()


# ---------------------------------------------------------------- STL input uses the same solver
def stl_file(tmp_path, n=260):
    pytest.importorskip("stl")
    import height_map_stl as hms
    np.save(tmp_path / "d.npy", stairs(n=n, width=40))
    hms.height_map_to_stl(tmp_path / "d.npy", tmp_path / "d.stl", pixel_size_um=1.0, height_max_um=1.0, levels=6, base_um=0.2)
    return tmp_path / "d.stl"


def written_travel(data_file):
    """Stage travel (um) of a data file: the MoveStage lines replayed, one stop per include."""
    import re
    x = y = 0.0
    pts = []
    for l in Path(data_file).read_text().replace("\r", "").split("\n"):
        m = re.match(r"MoveStage([XY]) (\S+)", l)
        if m:
            if m.group(1) == "X":
                x += float(m.group(2))
            else:
                y += float(m.group(2))
        if l.startswith("include"):
            pts.append((x, y))
    p = np.array(pts)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@pytest.mark.parametrize("method", ["adaptive", "standard"])
def test_stl_job_uses_pyconcorde_and_reports_the_written_travel(tmp_path, fake_concorde, method):
    stl = stl_file(tmp_path)
    rep = jg.generate_job_files(jg.JobSettings(input_file=str(stl), output_dir=str(tmp_path / "o"), method=method), rec())
    assert rep["fields"] >= 4
    r = rep["route"]
    assert r["solver"] == "pyconcorde" and r["optimal"] and r["length_um"] > 0, r
    assert FakeSolver.calls                                                       # the exact solver was really asked
    if method == "adaptive":                                                      # the standard grid uses relative moves per row
        assert written_travel(next((tmp_path / "o").glob("*_data.gwl"))) == pytest.approx(r["length_um"], rel=1e-3)
    assert "Writing order: pyconcorde" in next((tmp_path / "o").glob("*_job.gwl")).read_text()


def test_stl_job_falls_back_to_2opt_with_the_reason(tmp_path, monkeypatch):
    no_pyconcorde(monkeypatch)
    stl = stl_file(tmp_path)
    rep = jg.generate_job_files(jg.JobSettings(input_file=str(stl), output_dir=str(tmp_path / "o"), method="adaptive"), rec())
    assert rep["route"]["solver"] == "2-opt" and "pyconcorde" in rep["route"]["detail"]


def test_stl_preview_matches_the_job_order_for_both_methods(tmp_path, fake_concorde):
    stl = stl_file(tmp_path)
    fp = sp.stl_footprint(str(stl))
    for method in ("standard", "adaptive"):
        lay = sp.compute_layout(fp, method, rec())
        assert lay.route and lay.route["solver"] in ("pyconcorde", "2-opt"), method          # same solver call as the job
        assert rs.valid_order(lay.order, len(lay.centres))
