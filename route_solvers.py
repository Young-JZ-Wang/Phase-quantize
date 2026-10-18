#!/usr/bin/env python3
"""
route_solvers.py
----------------
Writing-path (field order) optimisation, with interchangeable backends.

The stage visits every field once; the order sets the stage travel. That is a travelling-salesman problem on the field
centres, but an OPEN one: the path starts and ends anywhere.

Backends (tried in the order given; "2-opt" is the dependency-free fallback and is always computed)
    "pyconcorde"  exact open path via Concorde (optional, `pip install` of the pyconcorde project, see its README). The
                  open path is solved as a closed tour with one extra "dummy" node at distance 0 from everything (the
                  tour is then cut at the dummy), so the result is the optimal OPEN path, not a closed tour with its
                  longest edge removed. Distances are rounded to integers (Concorde works on integers): `scale` = 100
                  gives 0.01 um, so "optimal" means optimal for those rounded distances.
    "2-opt"       the starting order (the TipSlicer TSP order) improved by 2-opt moves; good, not guaranteed optimal.

A route is called optimal only when Concorde proves it (success, no time limit hit, a valid tour). In every other case
(not installed, import or solver error, time limit reached, tour not valid, instance too small or too large) the 2-opt
route is used and reported as "2-opt".

More backends (e.g. ortools) can be added with register_solver(name, function).
"""
import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

import phase_quality as pqg

ROUTE_AUTO = ("pyconcorde", "2-opt")        # exact if possible, else 2-opt
ROUTE_FAST = ("2-opt",)                     # never calls an exact solver (used by the live preview)
DEFAULT_TIME_LIMIT = 60.0                   # seconds Concorde may take
DEFAULT_MAX_NODES = 1500                    # larger instances are not sent to the exact solver
DISTANCE_SCALE = 100                        # um -> integer units for Concorde


class SolverUnavailable(Exception):
    """A backend that cannot run here (not installed, instance unsuitable)."""


@dataclass
class RouteResult:
    order: np.ndarray                        # visiting order: indices into the points, every field exactly once
    solver: str                              # "pyconcorde" or "2-opt"
    optimal: bool                            # True only when Concorde proved the route optimal
    length_um: float                         # stage travel along the route (Euclidean)
    seconds: float = 0.0
    detail: str = ""                         # what happened, e.g. why the fallback was used
    attempts: list = field(default_factory=list)

    def info(self) -> dict:
        """The route metadata (everything but the order) for reports, notes and metrics."""
        return {"solver": self.solver, "optimal": self.optimal, "length_um": self.length_um, "seconds": self.seconds,
                "detail": self.detail}

    def describe(self) -> str:
        what = "optimal open path (Concorde)" if self.optimal else ("2-opt improved route" if self.solver == "2-opt" else self.solver)
        return f"{self.solver}: {what}, {self.length_um / 1000:.1f} mm of stage travel"


def path_length(points, order) -> float:
    return pqg.travel_distance(points, order)


def valid_order(order, n) -> bool:
    o = np.asarray(order)
    return o.ndim == 1 and len(o) == n and sorted(o.tolist()) == list(range(n))


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------
def solver_two_opt(points, init_order, **_):
    t0 = time.time()
    order = pqg.two_opt(points, init_order)
    return RouteResult(order, "2-opt", False, path_length(points, order), time.time() - t0, "2-opt improvement of the start order")


def write_open_path_tsplib(path, points, scale=DISTANCE_SCALE):
    """TSPLIB file of the open-path problem: node 0 is the dummy at distance 0 from every field, nodes 1..n the fields;
    distances are Euclidean in units of 1/scale um, rounded to integers (EXPLICIT / UPPER_ROW)."""
    p = np.asarray(points, dtype=float)
    n = len(p)
    d = np.rint(np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)) * scale).astype(np.int64)
    with open(path, "w") as f:
        f.write(f"NAME: fields\nTYPE: TSP\nCOMMENT: open writing path as a tour with a dummy node\nDIMENSION: {n + 1}\n"
                "EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: UPPER_ROW\nDISPLAY_DATA_TYPE: NO_DISPLAY\nEDGE_WEIGHT_SECTION\n")
        f.write(" ".join(["0"] * n) + "\n")                                                     # the dummy's row
        for i in range(n - 1):
            f.write(" ".join(str(int(v)) for v in d[i, i + 1:]) + "\n")
    return path


def solver_pyconcorde(points, init_order=None, time_limit=DEFAULT_TIME_LIMIT, max_nodes=DEFAULT_MAX_NODES, **_):
    """Exact open path through Concorde. Raises SolverUnavailable when it cannot give a proven optimal route."""
    n = len(points)
    if n < 4:
        raise SolverUnavailable(f"only {n} fields: nothing to optimise")
    if n > max_nodes:
        raise SolverUnavailable(f"{n} fields is more than the {max_nodes} sent to the exact solver")
    try:
        from concorde.tsp import TSPSolver
    except Exception as e:                                              # not installed, or its import fails
        raise SolverUnavailable(f"pyconcorde is not available ({type(e).__name__}: {e})") from e
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="nanoslicer_tsp_") as tmp:
        tsp = write_open_path_tsplib(os.path.join(tmp, "fields.tsp"), points)
        solution = TSPSolver.from_tspfile(tsp).solve(time_bound=float(time_limit), verbose=False)
    if not getattr(solution, "success", False):
        raise SolverUnavailable("Concorde did not report a solution")
    if getattr(solution, "hit_timebound", True):
        raise SolverUnavailable(f"Concorde hit the {time_limit:g} s time limit before proving optimality")
    tour = np.asarray(getattr(solution, "tour", []), dtype=int)
    if len(tour) != n + 1 or sorted(tour.tolist()) != list(range(n + 1)):
        raise SolverUnavailable("Concorde returned an invalid tour")
    k = int(np.where(tour == 0)[0][0])
    order = np.roll(tour, -k)[1:] - 1                                   # cut the tour at the dummy node
    if not valid_order(order, n):
        raise SolverUnavailable("the route cut from the Concorde tour is not valid")
    return RouteResult(order, "pyconcorde", True, path_length(points, order), time.time() - t0,
                       f"optimal open path for distances rounded to {1.0 / DISTANCE_SCALE:g} um")


SOLVERS = {"2-opt": solver_two_opt, "pyconcorde": solver_pyconcorde}


def register_solver(name, fn):
    """Add a backend: fn(points, init_order, time_limit=..., max_nodes=...) -> RouteResult, raising SolverUnavailable
    when it cannot run. Name it in the `solvers` list of solve_route to use it."""
    SOLVERS[name] = fn


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def solve_route(points, init_order, solvers=ROUTE_AUTO, time_limit=DEFAULT_TIME_LIMIT, max_nodes=DEFAULT_MAX_NODES,
                log=None) -> RouteResult:
    """The writing order of the field centres `points`, from the first backend in `solvers` that succeeds, with the
    2-opt route as the fallback (and as a check: a backend's route is only used when it is not longer than it).

    init_order may be an order or a function returning one (called once, only if needed). Returns a RouteResult whose
    order holds every field exactly once. `solver` is "pyconcorde" or "2-opt"; `optimal` is True only for a route
    Concorde proved optimal."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    unknown = [s for s in solvers if s not in SOLVERS]
    if unknown:
        raise ValueError(f"unknown route solver(s) {unknown}; available: {sorted(SOLVERS)}")
    start = init_order() if callable(init_order) else init_order
    start = np.arange(n) if start is None else np.asarray(start, dtype=int)
    if not valid_order(start, n):
        raise ValueError("the start order must contain every field exactly once")
    best = solver_two_opt(pts, start)
    attempts = []
    for name in solvers:
        if name == "2-opt":
            continue
        try:
            res = SOLVERS[name](pts, start, time_limit=time_limit, max_nodes=max_nodes)
        except SolverUnavailable as e:
            attempts.append(f"{name}: {e}")
            if log:
                log(f"route solver {name} not used: {e}")
            continue
        except Exception as e:                                          # a backend must never break the job
            attempts.append(f"{name}: failed ({type(e).__name__}: {e})")
            if log:
                log(f"route solver {name} failed ({type(e).__name__}: {e}); using the 2-opt route")
            continue
        if not valid_order(res.order, n):
            attempts.append(f"{name}: returned an invalid order")
            continue
        if res.length_um <= best.length_um + 1e-9:                       # never accept a longer route than the fallback
            best = res
            break
        attempts.append(f"{name}: route not shorter than the 2-opt route")
    best.attempts = attempts
    if attempts and best.solver == "2-opt":
        best.detail = "; ".join(attempts)
    return best
