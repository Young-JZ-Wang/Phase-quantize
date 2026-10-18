#!/usr/bin/env python3
"""
recipes.py
----------
Writing recipes for the Nanoscribe job files: the parameters that belong to the objective / resin and the exposure
(speed, power, slicing and hatching distance, ...), stored by name so they can be loaded, customised, saved as new
recipes and deleted.

The default recipe `63x - DIP` is built in: it can be selected and customised, but not overwritten or deleted
(save a customised copy under a new name). User recipes live in one JSON file:
    $NANOSLICER_HOME/recipes.json     (default: ~/.nanoslicer/recipes.json)
"""
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_NAME = "63x - DIP"

# Choices offered by the recipe's Objective / Mode / Resin drop-downs (labels only; edit the lists to add yours).
# Nanoscribe: DiLL = dip-in laser lithography (objective dipped into the resin), Oil = oil immersion.
# A recipe saved with some other label still loads and shows it: it is added to the list while selected.
OBJECTIVES = ("10x", "20x", "25x", "63x")
MODES = ("DiLL", "Oil")
RESINS = ("IP-Dip", "IP-Dip2", "IP-L", "IP-S", "IP-Q", "IP-n162", "IP-Visio")
RECIPE_CHOICES = {"objective": OBJECTIVES, "mode": MODES, "resin": RESINS}


@dataclass
class Recipe:
    name: str = DEFAULT_NAME
    objective: str = "63x"                  # objective (label; the scan field below is what is used)
    mode: str = "DiLL"                      # writing mode: DiLL or Oil (label)
    resin: str = "IP-Dip2"                  # photoresist (label)
    scan_speed: float = 40000.0             # um/s            -> $solidScanSpeed
    laser_power: float = 57.5               # power setting   -> $solidLaserPower
    slicing_distance: float = 0.2           # um, between layers (z)
    hatching_distance: float = 0.2          # um, between hatch lines
    shells: int = 0                         # contour shells (0 = solid hatch lines only)
    hatch_angle: float = 0.0                # deg; 0 = lines along x like DeScribe, -1 = random per layer
    field_size: float = 100.0               # um, side of a field (standard slicing)
    adaptive_radius: float = 70.71          # um, vote + write radius of a field (adaptive slicing)
    scan_field_diameter: float = 200.0      # um, scan field of the objective; fields must fit inside
    overlap: float = 0.0                    # um, overlap between neighbouring fields
    interface_pos: float = 0.5              # um, FindInterfaceAt for every field (-1 = off)
    piezo_settling_time: float = 10.0       # ms
    galvo_acceleration: float = 2.0
    stage_velocity: float = 200.0           # um/s
    # height-based slicing (adaptive method, "Height-based segmentation" switched on); the defaults change nothing
    base_slicing_distance: float = 0.0      # um, coarser z slicing below the segmentation height (0 = same as slicing_distance)
    base_power_multiplier: float = 1.0      # laser power factor below the segmentation height (1 = same power)
    # block shear angle (DeScribe "Block shear angle"): the block faces lean by this angle from the substrate normal, so the
    # seams between blocks are inclined; 0 = vertical seams (the default, output unchanged)
    block_shear_angle: float = 0.0          # deg from the substrate normal
    block_shear_direction: float = 0.0      # deg in the xy plane the blocks lean towards (0 = +x, 90 = +y)

    def validate(self):
        """Raises ValueError with a readable message for a recipe that cannot be written."""
        if not str(self.name).strip():
            raise ValueError("the recipe needs a name")
        positive = ("scan_speed", "laser_power", "slicing_distance", "hatching_distance", "field_size",
                    "adaptive_radius", "scan_field_diameter", "galvo_acceleration", "stage_velocity")
        for k in positive:
            if not float(getattr(self, k)) > 0:
                raise ValueError(f"{k.replace('_', ' ')} must be > 0")
        if self.shells < 0 or int(self.shells) != self.shells:
            raise ValueError("shells must be a whole number >= 0")
        if self.base_slicing_distance < 0:
            raise ValueError("base slicing distance must be >= 0 (0 = same as the slicing distance)")
        if not self.base_power_multiplier > 0:
            raise ValueError("base power multiplier must be > 0")
        if not 0 <= self.block_shear_angle <= 60:
            raise ValueError("block shear angle must be between 0 and 60 degrees (DeScribe recommends 15-30 for stitching)")
        if not -360 <= self.block_shear_direction <= 360:
            raise ValueError("block shear direction must be between -360 and 360 degrees")
        if self.overlap < 0 or self.piezo_settling_time < 0:
            raise ValueError("overlap and piezo settling time must be >= 0")
        if self.hatch_angle != -1 and not -360 <= self.hatch_angle <= 360:
            raise ValueError("hatch angle must be -1 (random) or between -360 and 360")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Unknown keys are ignored and missing keys take the default, so older / newer files still load."""
        known = {f.name: f.type for f in fields(cls)}
        d = dict(d)
        if "resin" not in d and d.get("mode") not in (None, "DiLL", "Oil"):
            # older recipes kept the resin in "mode" (e.g. "DIP" = IP-Dip in DiLL): split it into the two fields
            d["resin"] = {"DIP": "IP-Dip"}.get(str(d["mode"]), str(d["mode"]))
            d["mode"] = "DiLL"
        kw = {}
        for k, v in d.items():
            if k in known:
                kw[k] = str(v) if known[k] is str else int(v) if known[k] is int else float(v)
        return cls(**kw)


def default_recipe() -> Recipe:
    return Recipe()


def home_dir() -> Path:
    return Path(os.environ.get("NANOSLICER_HOME") or Path.home() / ".nanoslicer")


class RecipeStore:
    """Named recipes: the built-in default plus the user's own, kept in a JSON file."""

    def __init__(self, path=None):
        self.path = Path(path) if path else home_dir() / "recipes.json"
        self._user: dict[str, Recipe] = {}
        self.load()

    # ---- persistence
    def load(self):
        self._user = {}
        if not self.path.is_file():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw.get("recipes", []):
                r = Recipe.from_dict(item).validate()
                if r.name != DEFAULT_NAME:
                    self._user[r.name] = r
        except (OSError, ValueError, TypeError, AttributeError):
            # unreadable file: keep a copy, start empty rather than losing or crashing on it
            try:
                os.replace(self.path, self.path.with_suffix(".json.bak"))
            except OSError:
                pass
            self._user = {}

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "recipes": [r.to_dict() for r in self._user.values()]}, f, indent=2)
        os.replace(tmp, self.path)                      # atomic: never leaves a half-written file

    # ---- access
    def names(self) -> list[str]:
        return [DEFAULT_NAME] + sorted(self._user, key=str.lower)

    def get(self, name: str) -> Recipe:
        if name == DEFAULT_NAME:
            return default_recipe()
        if name not in self._user:
            raise KeyError(f"no recipe named {name!r}")
        return Recipe.from_dict(self._user[name].to_dict())      # a copy, callers may edit it

    def is_builtin(self, name: str) -> bool:
        return name == DEFAULT_NAME

    # ---- changes
    def save(self, recipe: Recipe, overwrite=False) -> Recipe:
        """Store `recipe` under recipe.name. The built-in default is never overwritten; an existing user recipe only
        with overwrite=True."""
        recipe = Recipe.from_dict(recipe.to_dict()).validate()
        recipe.name = recipe.name.strip()
        if recipe.name == DEFAULT_NAME:
            raise ValueError(f"{DEFAULT_NAME!r} is the built-in default; save your changes under a new name")
        if recipe.name in self._user and not overwrite:
            raise ValueError(f"a recipe named {recipe.name!r} already exists")
        self._user[recipe.name] = recipe
        self._write()
        return recipe

    def delete(self, name: str):
        if name == DEFAULT_NAME:
            raise ValueError(f"{DEFAULT_NAME!r} is the built-in default and cannot be deleted")
        if name not in self._user:
            raise KeyError(f"no recipe named {name!r}")
        del self._user[name]
        self._write()
