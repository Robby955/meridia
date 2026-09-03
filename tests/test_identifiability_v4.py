"""The identifiability reading is a file, and the worlds it was taken on are named in it.

Six pooled correlations decide which axes an evaluation world may push past the
development band. A correlation over thirty worlds is a claim about thirty definite
worlds: every figure moves with which hidden worlds were drawn, so a reading whose seed
set lives only in one run is a reading no reader can rebuild and no test can check. These
tests hold the record, the seeds committed in the builder, and the figures printed in the
decisions record to each other.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RECORD = ROOT / "measurements" / "identifiability_v4.json"
DECISIONS = ROOT / "docs" / "V4_DECISIONS.md"
FIGURE = re.compile(
    r"^ {4}(\w+) +([+-]\d\.\d{3}) +\[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]$", re.MULTILINE)


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record() -> dict:
    return json.loads(RECORD.read_text())


def test_the_measurement_names_the_worlds_it_was_taken_on():
    record = _record()
    build = _script("build_v4_worlds")
    worlds = record["worlds"]

    assert record["schema"] == _script("identifiability_v4").RECORD_SCHEMA
    assert worlds["development_seeds"] == list(build.DEVELOPMENT_SEEDS)
    assert worlds["hidden_seeds"] == list(build.IDENTIFIABILITY_HIDDEN_SEEDS)
    assert len(worlds["development_seeds"]) + len(worlds["hidden_seeds"]) == 30
    # The measurement worlds are their own set. Nothing is graded on them and no bar is
    # frozen on them, so they do not reach into the qualification set either.
    assert set(worlds["hidden_seeds"]).isdisjoint(build.QUALIFICATION_SEEDS)
    assert set(worlds["hidden_seeds"]).isdisjoint(worlds["development_seeds"])

    plan = build.family_plan("identifiability")
    assert [entry["seed"] for entry in plan] == \
        worlds["development_seeds"] + worlds["hidden_seeds"]
    assert [entry["params"].regime for entry in plan] == \
        ["development"] * 12 + ["hidden"] * 18
    assert {entry["params"].ensemble_members for entry in plan} == \
        {worlds["ensemble_members"]}

    committed = build.WORLD
    assert worlds["grid"] == list(committed.grid)
    assert worlds["total"] == committed.total
    assert worlds["n_states"] == committed.n_states
    assert worlds["observed_months"] == committed.observed_months
    assert worlds["horizon_months"] == committed.horizon_months
    assert "--family identifiability" in worlds["build"]


def test_every_axis_carries_the_spread_of_its_own_reading():
    record = _record()
    script = _script("identifiability_v4")
    assert set(record["axes"]) == set(script.AXES)
    assert record["resample"]["draws"] == script.RESAMPLE_DRAWS
    assert record["resample"]["ends"] == list(script.RESAMPLE_ENDS)
    for axis, entry in record["axes"].items():
        low, high = entry["pooled_interval"]
        assert low < entry["pooled"] < high
        hidden_low, hidden_high = entry["hidden_interval"]
        assert hidden_low < entry["within_hidden"] < hidden_high
        assert entry["expected_sign"] == script.EXPECTED_SIGN[axis]
        assert entry["statistic"] == script.STATISTIC[axis]


def test_the_decisions_record_prints_the_reading_the_file_holds():
    record = _record()
    printed = FIGURE.findall(DECISIONS.read_text())
    assert len(printed) == len(record["axes"])
    for axis, pooled, low, high in printed:
        entry = record["axes"][axis]
        assert float(pooled) == entry["pooled"]
        assert [float(low), float(high)] == entry["pooled_interval"]


def test_the_record_is_written_from_the_worlds_and_not_from_the_run():
    """A statistic that orders the worlds exactly reads +1, and the sign is the
    mechanism's own."""
    script = _script("identifiability_v4")
    rng = np.random.default_rng(11)
    truth = rng.normal(size=30)
    rows = {"regime": ["development"] * 12 + ["hidden"] * 18}
    for axis in script.AXES:
        rows[f"true_{axis}"] = truth
        rows[f"read_{axis}"] = truth * script.EXPECTED_SIGN[axis]
    frame = pd.DataFrame(rows)
    record = script.measurement_record(frame)
    for axis in script.AXES:
        entry = record["axes"][axis]
        assert entry["pooled"] == 1.0
        # A resample repeats worlds, so its ranks carry ties the original does not and
        # the interval sits a little under the reading it brackets.
        assert 0.95 < entry["pooled_interval"][0] <= entry["pooled_interval"][1] <= 1.0
    assert record["worlds"]["hidden_seeds"] == \
        list(_script("build_v4_worlds").IDENTIFIABILITY_HIDDEN_SEEDS)
