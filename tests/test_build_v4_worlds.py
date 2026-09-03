"""The world-set builder: graded seeds come from a sealed file and never reach a log.

A world's whole configuration is a function of its seed, so a graded seed written into
the repository is the graded configuration written into the repository, and a graded seed
printed by the build loop is the same thing in a terminal scrollback. Development seeds
are the opposite case: a method may tune on those worlds, so their seeds are committed
and printed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_v4_worlds.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_v4_worlds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_graded_seed_is_written_into_the_repository():
    module = _module()
    assert not hasattr(module, "GRADED_SEEDS")
    text = SCRIPT.read_text()
    for name in ("DEVELOPMENT_SEEDS", "QUALIFICATION_SEEDS"):
        assert name in text
    # The graded family names a file, not a tuple of integers.
    assert "GRADED_SEED_FILE" in text
    assert str(module.GRADED_SEED_FILE) not in (str(SCRIPT.parent), str(SCRIPT))
    assert SCRIPT.parents[1] not in module.GRADED_SEED_FILE.parents


def test_graded_seeds_are_read_from_the_sealed_file(tmp_path):
    module = _module()
    sealed = tmp_path / "v4_graded_seeds.json"
    sealed.write_text(json.dumps([4241, 4242, 4243]))
    assert module.graded_seeds(sealed) == (4241, 4242, 4243)

    plan = module.family_plan("graded", sealed)
    assert [entry["name"] for entry in plan] == ["graded-0", "graded-1", "graded-2"]
    assert all(entry["params"].regime == "hidden" for entry in plan)
    assert all(entry["development"] is False for entry in plan)

    with pytest.raises(FileNotFoundError, match="graded seed file"):
        module.graded_seeds(tmp_path / "absent.json")
    (tmp_path / "repeats.json").write_text(json.dumps([7, 7]))
    with pytest.raises(ValueError, match="repeats"):
        module.graded_seeds(tmp_path / "repeats.json")
    (tmp_path / "wrong.json").write_text(json.dumps({"seeds": [7]}))
    with pytest.raises(ValueError, match="list of integers"):
        module.graded_seeds(tmp_path / "wrong.json")


def test_only_development_seeds_reach_the_build_log(tmp_path):
    module = _module()
    sealed = tmp_path / "v4_graded_seeds.json"
    sealed.write_text(json.dumps([4241]))
    lines = []
    for family, plan in (("development", module.family_plan("development")),
                         ("qualification", module.family_plan("qualification")),
                         ("graded", module.family_plan("graded", sealed))):
        for entry in plan:
            line = module.progress_line(family, entry, 22, 6.0)
            if family == "development":
                assert f"seed {entry['seed']}" in line
            else:
                assert str(entry["seed"]) not in line
                assert "seed" not in line
            lines.append(line)
    assert len(lines) == 12 + 6 + 1


def test_the_three_families_share_one_committed_world_size():
    module = _module()
    sealed_free = ("development", "qualification")
    sizes = {family: {(entry["params"].grid, entry["params"].total,
                       entry["params"].observed_months, entry["params"].horizon_months,
                       entry["params"].ensemble_members)
                      for entry in module.family_plan(family)}
             for family in sealed_free}
    assert len(sizes["development"] | sizes["qualification"]) == 1
    with pytest.raises(ValueError, match="unknown world family"):
        module.family_plan("something-else")
