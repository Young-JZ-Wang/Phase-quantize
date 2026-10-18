import json

import pytest

from recipes import DEFAULT_NAME, Recipe, RecipeStore, default_recipe


def test_default_recipe_is_63x_dip():
    r = default_recipe()
    assert r.name == DEFAULT_NAME == "63x - DIP"
    assert (r.objective, r.mode, r.resin) == ("63x", "DiLL", "IP-Dip2")
    assert r.validate() is r


def test_store_lists_default_first_and_it_always_exists(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    assert st.names() == [DEFAULT_NAME]
    assert st.is_builtin(DEFAULT_NAME)
    assert st.get(DEFAULT_NAME).to_dict() == default_recipe().to_dict()


def test_save_load_delete_roundtrip(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    r = default_recipe(); r.name = "25x - test"; r.objective = "25x"; r.laser_power = 40; r.shells = 2
    st.save(r)
    assert st.names() == [DEFAULT_NAME, "25x - test"]
    again = RecipeStore(tmp_path / "r.json")                            # a fresh process reads it back
    got = again.get("25x - test")
    assert (got.objective, got.laser_power, got.shells) == ("25x", 40.0, 2)
    again.delete("25x - test")
    assert RecipeStore(tmp_path / "r.json").names() == [DEFAULT_NAME]


def test_get_returns_a_copy(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    st.get(DEFAULT_NAME).laser_power = 1
    assert st.get(DEFAULT_NAME).laser_power == default_recipe().laser_power


def test_default_cannot_be_overwritten_or_deleted(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    r = default_recipe(); r.laser_power = 99
    with pytest.raises(ValueError):
        st.save(r, overwrite=True)
    with pytest.raises(ValueError):
        st.delete(DEFAULT_NAME)
    assert st.get(DEFAULT_NAME).laser_power == 57.5


def test_save_needs_overwrite_for_an_existing_name(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    r = default_recipe(); r.name = "mine"
    st.save(r)
    with pytest.raises(ValueError):
        st.save(r)
    r.laser_power = 50
    st.save(r, overwrite=True)
    assert st.get("mine").laser_power == 50


def test_delete_missing_and_bad_values(tmp_path):
    st = RecipeStore(tmp_path / "r.json")
    with pytest.raises(KeyError):
        st.delete("nope")
    with pytest.raises(KeyError):
        st.get("nope")
    bad = default_recipe(); bad.name = "bad"; bad.scan_speed = 0
    with pytest.raises(ValueError):
        st.save(bad)
    for field, value in (("shells", -1), ("overlap", -1), ("hatch_angle", 1000), ("name", "  ")):
        r = default_recipe(); setattr(r, field, value)
        with pytest.raises(ValueError):
            r.validate()


def test_unknown_keys_ignored_and_missing_keys_defaulted(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"version": 7, "recipes": [{"name": "old", "laser_power": 33, "future_field": 1}]}))
    got = RecipeStore(p).get("old")
    assert got.laser_power == 33 and got.scan_speed == default_recipe().scan_speed


def test_corrupt_file_is_kept_as_backup_and_store_starts_empty(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{ not json")
    st = RecipeStore(p)
    assert st.names() == [DEFAULT_NAME]
    assert (tmp_path / "r.json.bak").read_text() == "{ not json"
    r = default_recipe(); r.name = "after"
    st.save(r)
    assert RecipeStore(p).names() == [DEFAULT_NAME, "after"]


def test_home_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOSLICER_HOME", str(tmp_path / "home"))
    st = RecipeStore()
    r = default_recipe(); r.name = "x"
    st.save(r)
    assert (tmp_path / "home" / "recipes.json").is_file()


def test_recipe_from_dict_types():
    r = Recipe.from_dict({"shells": "3", "scan_speed": "1000", "objective": 25})
    assert r.shells == 3 and r.scan_speed == 1000.0 and r.objective == "25"



def test_height_based_fields_default_to_off_and_are_validated():
    r = default_recipe()
    assert r.base_slicing_distance == 0.0 and r.base_power_multiplier == 1.0          # the default recipe is unchanged in effect
    for field, value in (("base_slicing_distance", -1), ("base_power_multiplier", 0)):
        bad = default_recipe(); setattr(bad, field, value)
        with pytest.raises(ValueError):
            bad.validate()
    old = Recipe.from_dict({"name": "old", "laser_power": 40})                         # a recipe saved before these fields existed
    assert old.base_slicing_distance == 0.0 and old.base_power_multiplier == 1.0


def test_old_recipe_with_resin_in_mode_is_split():
    from recipes import Recipe
    old = Recipe.from_dict({"name": "old", "objective": "25x", "mode": "DIP", "laser_power": 50})
    assert (old.mode, old.resin) == ("DiLL", "IP-Dip")
    other = Recipe.from_dict({"name": "o2", "mode": "IP-S"})
    assert (other.mode, other.resin) == ("DiLL", "IP-S")
    new = Recipe.from_dict({"name": "n", "mode": "Oil", "resin": "IP-L"})
    assert (new.mode, new.resin) == ("Oil", "IP-L")
    assert Recipe.from_dict(Recipe().to_dict()) == Recipe()


def test_choices_are_the_nanoscribe_modes_and_resins():
    from recipes import MODES, RESINS
    assert MODES == ("DiLL", "Oil")
    assert {"IP-Dip2", "IP-L", "IP-S", "IP-Q", "IP-n162", "IP-Visio"} <= set(RESINS)
