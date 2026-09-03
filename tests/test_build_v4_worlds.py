"""The world-set builder opens graded seeds only after freeze authorization.

A world's whole configuration is a function of its seed, so a graded seed written into
the participant packet is the graded configuration written into the participant packet,
and a graded seed printed by the build loop is the same thing in a terminal scrollback.
The seed remains only in sealed retained metadata. Development seeds are the opposite
case: a method may tune on those worlds, so their seeds are committed and printed.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.graded_readiness import GradedReadiness
import meridia.sealing as sealing

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_v4_worlds.py"


def test_build_v4_worlds_cli_help_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/build_v4_worlds.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--world-workers" in result.stdout


def _module():
    spec = importlib.util.spec_from_file_location("build_v4_worlds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _readiness(module):
    return GradedReadiness(
        bars_sha256="a" * 64,
        reserve_calibration_sha256="b" * 64,
        reserve_rate_per_person_year=module.WORLD.reserve_rate_per_person_year,
        graded_world_count=3,
    )


def _write_v4_seal(module, path, count=3):
    del module, count
    path.write_text("{}")
    return path


def _graded_paths(module, tmp_path, count=3):
    seal = _write_v4_seal(module, tmp_path / "v4-seal.json", count)
    key = tmp_path / "v4.key"
    key.write_bytes(b"k" * 32)
    return {
        "bars_path": tmp_path / "bars.json",
        "reserve_calibration_path": tmp_path / "reserve.json",
        "seal_manifest_path": seal,
        "key_path": key,
    }


def _graded_argv(module, tmp_path):
    paths = _graded_paths(module, tmp_path)
    return [
        str(SCRIPT),
        "--out", str(tmp_path / "worlds"),
        "--family", "graded",
        "--bars", str(paths["bars_path"]),
        "--reserve-calibration-audit", str(paths["reserve_calibration_path"]),
        "--seal-manifest", str(paths["seal_manifest_path"]),
        "--key", str(paths["key_path"]),
    ]


def _job(module, tmp_path, *, family="qualification", workers=1, world_workers=1):
    entry = module.family_plan(family)[0]
    return {
        "family": family,
        "entry": entry,
        "out": str(tmp_path),
        "workers": workers,
        "world_workers": world_workers,
        "cache": None,
    }


def _graded_entry(module, index=0):
    return {
        "name": f"graded-{index}",
        "index": index,
        "authorization_binding_sha256": "e" * 64,
        "params": module.PacketParams(**{**module.WORLD.__dict__, "regime": "hidden"}),
        "development": False,
        "public_seed": False,
    }


def _graded_job(module, tmp_path, *, index=0):
    paths = _graded_paths(module, tmp_path)
    return {
        "family": "graded",
        "entry": _graded_entry(module, index),
        "out": str(tmp_path / "worlds"),
        "workers": 1,
        "world_workers": 1,
        "cache": None,
        "graded_authorization": {
            name: str(path) for name, path in paths.items()
        },
    }


def test_graded_family_has_no_raw_seed_file_path():
    module = _module()
    assert not hasattr(module, "GRADED_SEEDS")
    assert not hasattr(module, "GRADED_SEED_FILE")
    assert not hasattr(module, "graded_seeds")
    text = SCRIPT.read_text()
    for name in ("DEVELOPMENT_SEEDS", "QUALIFICATION_SEEDS"):
        assert name in text
    assert "--seal-manifest" in text
    assert "verify_and_derive_v4_seed" in text


def test_graded_plan_verifies_the_v4_seal_without_returning_seeds(tmp_path, monkeypatch):
    module = _module()
    paths = _graded_paths(module, tmp_path)
    readiness = _readiness(module)
    calls = []

    def validate(*args, **kwargs):
        calls.append("readiness")
        assert kwargs["expected_rate_per_person_year"] \
            == module.WORLD.reserve_rate_per_person_year
        return readiness

    def authorize(index, manifest, key, **kwargs):
        calls.append(f"verify-{index}")
        assert manifest == paths["seal_manifest_path"]
        assert key == paths["key_path"]
        assert kwargs["params"].regime == "hidden"
        assert kwargs["params"].design_cell is None
        assert kwargs["readiness"] is readiness
        return module.V4WorldAuthorization(
            seed=9000 + index,
            binding_sha256="d" * 64,
        )

    monkeypatch.setattr(module, "validate_graded_readiness", validate)
    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    plan = module.family_plan("graded", **paths)

    assert all("seed" not in entry for entry in plan)
    assert [entry["index"] for entry in plan] == [0, 1, 2]
    assert {entry["authorization_binding_sha256"] for entry in plan} == {"d" * 64}
    assert calls == ["readiness", "verify-0", "verify-1", "verify-2"]
    assert [entry["name"] for entry in plan] == ["graded-0", "graded-1", "graded-2"]
    assert all(entry["params"].regime == "hidden" for entry in plan)
    assert all(entry["params"].design_cell is None for entry in plan)
    assert all(entry["development"] is False for entry in plan)


def test_graded_family_validates_receipts_before_any_seal_or_key_read(
    tmp_path, monkeypatch
):
    module = _module()
    calls = []

    def refuse(*args, **kwargs):
        calls.append("readiness")
        raise ValueError("freeze incomplete")

    monkeypatch.setattr(module, "validate_graded_readiness", refuse)
    monkeypatch.setattr(
        module,
        "verify_and_derive_v4_seed",
        lambda *args, **kwargs: pytest.fail("seal was read before readiness"),
    )
    with pytest.raises(ValueError, match="freeze incomplete"):
        module.family_plan(
            "graded",
            bars_path=tmp_path / "bars.json",
            reserve_calibration_path=tmp_path / "reserve.json",
            seal_manifest_path=tmp_path / "seal.json",
            key_path=tmp_path / "key",
        )
    assert calls == ["readiness"]


def test_graded_family_rejects_missing_authorization_and_forged_token(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="require bars"):
        module.family_plan("graded")
    with pytest.raises(TypeError, match="graded_readiness"):
        module.family_plan(
            "graded",
            **_graded_paths(module, tmp_path),
            graded_readiness=_readiness(module),
        )


def test_seal_rejection_stops_the_graded_plan(tmp_path, monkeypatch):
    module = _module()
    paths = _graded_paths(module, tmp_path, count=2)
    monkeypatch.setattr(
        module, "validate_graded_readiness", lambda *args, **kwargs: _readiness(module)
    )
    monkeypatch.setattr(
        module,
        "verify_and_derive_v4_seed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("V4 seal world count or requested index is invalid")
        ),
    )
    with pytest.raises(ValueError, match="world count"):
        module.family_plan("graded", **paths)


def test_only_development_seeds_reach_the_build_log(tmp_path):
    module = _module()
    lines = []
    for family, plan in (("development", module.family_plan("development")),
                         ("qualification", module.family_plan("qualification")),
                         ("graded", [{"name": "graded-0", "index": 0,
                                      "public_seed": False}])):
        for entry in plan:
            line = module.progress_line(family, entry, 22, 6.0)
            if family == "development":
                assert f"seed {entry['seed']}" in line
            else:
                assert "seed" not in line
            lines.append(line)
    assert len(lines) == 12 + 6 + 1

    malformed = dict(module.family_plan("qualification")[0], public_seed=True)
    hidden_line = module.progress_line("qualification", malformed, 22, 6.0)
    assert str(malformed["seed"]) not in hidden_line


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


def test_the_ensemble_step_is_divided_apart_from_the_baseline_step():
    """Two flags, two axes of parallelism, and a cache between rebuilds.

    The twenty-one world set took about three and a half hours on six processes because
    the only division was inside one world's ensemble, and every world waited for the one
    before it. Whole worlds share nothing, so they go in parallel; the ensemble is the
    part that costs, and it depends on nothing a verifier or a bar reads, so it is cached
    on the digest of the baseline ledger that produced it.
    """
    module = _module()
    text = SCRIPT.read_text()
    assert "--world-workers" in text
    assert "--cache" in text
    assert callable(module.build_one)
    for flag in ("--out", "--family", "--workers", "--world-workers", "--cache",
                 "--bars", "--reserve-calibration-audit", "--seal-manifest",
                 "--key"):
        assert flag in text


def test_cli_requires_one_explicit_family(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--out", str(tmp_path), "--family", "all"],
    )
    with pytest.raises(SystemExit):
        module.main()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_cli_rejects_invalid_worker_counts(value):
    module = _module()
    with pytest.raises(module.argparse.ArgumentTypeError, match="positive integer"):
        module.positive_integer(value)


def test_graded_cli_validates_receipts_before_seal_or_key_access(monkeypatch, tmp_path):
    module = _module()
    sentinel = RuntimeError("receipt rejected before seed access")

    def reject_receipts(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(module, "validate_graded_readiness", reject_receipts)
    monkeypatch.setattr(
        module,
        "verify_and_derive_v4_seed",
        lambda *args, **kwargs: pytest.fail("seal was read before readiness"),
    )
    monkeypatch.setattr(sys, "argv", _graded_argv(module, tmp_path))
    with pytest.raises(RuntimeError, match="receipt rejected"):
        module.main()
    assert not (tmp_path / "worlds").exists()


def test_graded_cli_requires_every_authorization_input(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(
        module,
        "validate_graded_readiness",
        lambda *args, **kwargs: pytest.fail("incomplete authorization was validated"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--out", str(tmp_path / "worlds"), "--family", "graded",
         "--bars", str(tmp_path / "bars.json"),
         "--reserve-calibration-audit", str(tmp_path / "reserve.json"),
         "--seal-manifest", str(tmp_path / "seal.json")],
    )
    with pytest.raises(SystemExit):
        module.main()


def test_params_or_source_mismatch_stops_before_any_packet_build(monkeypatch, tmp_path):
    module = _module()
    readiness = _readiness(module)
    verification_calls = []

    monkeypatch.setattr(
        module, "validate_graded_readiness", lambda *args, **kwargs: readiness
    )

    def reject_mismatch(index, manifest, key, **kwargs):
        verification_calls.append(index)
        assert kwargs["params"].regime == "hidden"
        assert kwargs["params"].design_cell is None
        assert kwargs["readiness"] is readiness
        raise ValueError("V4 seal law or freeze receipts have drifted")

    monkeypatch.setattr(module, "verify_and_derive_v4_seed", reject_mismatch)
    monkeypatch.setattr(
        module,
        "build_one",
        lambda *args, **kwargs: pytest.fail("unverified world reached packet build"),
    )
    monkeypatch.setattr(sys, "argv", _graded_argv(module, tmp_path))
    with pytest.raises(ValueError, match="law or freeze receipts"):
        module.main()
    assert verification_calls == [0]
    assert not (tmp_path / "worlds").exists()


def test_direct_graded_build_cannot_bypass_freeze_and_seal(monkeypatch, tmp_path):
    module = _module()
    job = _graded_job(module, tmp_path)
    job.pop("graded_authorization")
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("unauthorized graded packet was built"),
    )
    with pytest.raises(ValueError, match="lacks complete"):
        module.build_one(job)

    job = _graded_job(module, tmp_path)
    job["entry"]["seed"] = 12345
    with pytest.raises(ValueError, match="canonical sealed-world entry"):
        module.build_one(job)


def test_graded_worker_revalidates_binding_before_and_after_build(
    monkeypatch, tmp_path
):
    module = _module()
    job = _graded_job(module, tmp_path)
    seed = 987_654_321
    calls = []

    def validate_readiness(*args, **kwargs):
        calls.append("readiness")
        return _readiness(module)

    def authorize(index, manifest, key, **kwargs):
        calls.append("authorize")
        return module.V4WorldAuthorization(seed=seed, binding_sha256="e" * 64)

    def build(actual_seed, path, params, **kwargs):
        assert actual_seed == seed
        calls.append("build")
        authority = kwargs["graded_authorization"]
        assert isinstance(authority, module.V4PublicationAuthorization)
        authority.confirm(seed=actual_seed, params=params)
        path.mkdir(parents=True)

    def validate_packet(path, **kwargs):
        assert kwargs["expected_seed"] == seed
        calls.append("packet")
        return {
            "development": False,
            "participant": {"contract.json": {}},
            "retained": {"world.json": {}},
        }

    monkeypatch.setattr(module, "validate_graded_readiness", validate_readiness)
    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(sealing, "validate_graded_readiness", validate_readiness)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(module, "build_packet", build)
    monkeypatch.setattr(module, "validate_packet_directory", validate_packet)
    line = module.build_one(job)
    assert calls == [
        "readiness", "authorize", "build", "readiness", "authorize", "packet"
    ]
    assert str(seed) not in line
    directory = Path(job["out"]) / "graded" / "graded-0"
    assert directory.is_dir()


def test_graded_worker_rejects_authorization_drift_after_build(monkeypatch, tmp_path):
    module = _module()
    job = _graded_job(module, tmp_path)
    job["entry"]["authorization_binding_sha256"] = "a" * 64
    calls = 0

    monkeypatch.setattr(
        module, "validate_graded_readiness", lambda *args, **kwargs: _readiness(module)
    )

    def authorize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return module.V4WorldAuthorization(
            seed=111,
            binding_sha256=("a" if calls == 1 else "b") * 64,
        )

    def build(seed, path, params, **kwargs):
        del path
        kwargs["graded_authorization"].confirm(seed=seed, params=params)

    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(sealing, "validate_graded_readiness", module.validate_graded_readiness)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(module, "build_packet", build)
    monkeypatch.setattr(
        module,
        "validate_packet_directory",
        lambda *args, **kwargs: {
            "development": False,
            "participant": {"contract.json": {}},
            "retained": {"world.json": {}},
        },
    )
    with pytest.raises(RuntimeError, match="authorization"):
        module.build_one(job)
    assert calls == 2
    directory = Path(job["out"]) / "graded" / "graded-0"
    assert not directory.exists()


@pytest.mark.parametrize("workers", [0, -1, True, False, 1.5, "2", None])
def test_python_api_rejects_invalid_worker_counts(monkeypatch, tmp_path, workers):
    module = _module()
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("invalid job reached packet build"),
    )
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        module.build_one(_job(module, tmp_path, workers=workers))


@pytest.mark.parametrize("world_workers", [0, -1, True, False, 1.5, "2", None])
def test_python_api_rejects_invalid_world_worker_counts(
    monkeypatch, tmp_path, world_workers
):
    module = _module()
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("invalid job reached packet build"),
    )
    with pytest.raises(ValueError, match="world_workers must be a positive integer"):
        module.build_one(_job(module, tmp_path, world_workers=world_workers))


def test_python_api_rejects_nested_parallelism(monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("invalid job reached packet build"),
    )
    with pytest.raises(ValueError, match="cannot both exceed one"):
        module.build_one(_job(module, tmp_path, workers=2, world_workers=2))


def test_invalid_existing_packet_fails_without_deleting_it(monkeypatch, tmp_path):
    module = _module()
    job = _job(module, tmp_path)
    directory = tmp_path / job["family"] / job["entry"]["name"]
    directory.mkdir(parents=True)
    marker = directory / "interrupted.txt"
    marker.write_text("keep for diagnosis")

    def reject_packet(*args, **kwargs):
        raise ValueError("packet manifest is incomplete")

    monkeypatch.setattr(module, "validate_packet_directory", reject_packet)
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("existing packet was overwritten"),
    )
    with pytest.raises(ValueError, match="manifest is incomplete"):
        module.build_one(job)
    assert marker.read_text() == "keep for diagnosis"


def test_valid_existing_packet_is_resumed_only_after_validation(monkeypatch, tmp_path):
    module = _module()
    job = _job(module, tmp_path)
    directory = tmp_path / job["family"] / job["entry"]["name"]
    directory.mkdir(parents=True)
    calls = []
    finalized = []

    def validate(path, **kwargs):
        calls.append((path, kwargs))
        return {"development": False, "participant": {}, "retained": {}}

    monkeypatch.setattr(module, "validate_packet_directory", validate)
    monkeypatch.setattr(
        module,
        "_finalize_packet_build_intent",
        lambda path, **kwargs: finalized.append((path, kwargs)),
    )
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("valid packet was rebuilt"),
    )
    line = module.build_one(job)
    assert line == "qualification/qual-0: already built"
    assert calls == [(
        directory,
        {
            "expected_packet_class": "qualification",
            "expected_params": job["entry"]["params"],
            "expected_seed": job["entry"]["seed"],
        },
    )]
    assert finalized == [(
        directory,
        {
            "seed": job["entry"]["seed"],
            "params": job["entry"]["params"],
            "packet_class": "qualification",
            "development": False,
            "graded_authorization": None,
        },
    )]
    assert str(job["entry"]["seed"]) not in line


def test_existing_graded_packet_is_reused_only_after_two_authorizations(
    monkeypatch, tmp_path
):
    module = _module()
    job = _graded_job(module, tmp_path)
    directory = Path(job["out"]) / "graded" / "graded-0"
    directory.mkdir(parents=True)
    calls = []

    def readiness(*args, **kwargs):
        calls.append("readiness")
        return _readiness(module)

    def authorize(*args, **kwargs):
        calls.append("authorize")
        return module.V4WorldAuthorization(seed=111, binding_sha256="e" * 64)

    def validate(*args, **kwargs):
        calls.append("packet")
        return {"development": False, "participant": {}, "retained": {}}

    def finalize(*args, **kwargs):
        del args, kwargs
        calls.append("intent")

    monkeypatch.setattr(module, "validate_graded_readiness", readiness)
    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(sealing, "validate_graded_readiness", readiness)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(module, "validate_packet_directory", validate)
    monkeypatch.setattr(module, "_finalize_packet_build_intent", finalize)
    monkeypatch.setattr(
        module,
        "build_packet",
        lambda *args, **kwargs: pytest.fail("existing graded packet was rebuilt"),
    )
    assert module.build_one(job) == "graded/graded-0: already built"
    assert calls == [
        "readiness", "authorize", "packet", "readiness", "authorize", "intent"
    ]


def test_new_packet_is_classified_and_validated_after_build(monkeypatch, tmp_path):
    module = _module()
    job = _job(module, tmp_path)
    directory = tmp_path / job["family"] / job["entry"]["name"]
    build_calls = []
    validation_calls = []

    def build(seed, path, params, **kwargs):
        build_calls.append((seed, path, params, kwargs))
        path.mkdir(parents=True)

    def validate(path, **kwargs):
        validation_calls.append((path, kwargs))
        return {
            "development": False,
            "participant": {"contract.json": {}},
            "retained": {"world.json": {}},
        }

    monkeypatch.setattr(module, "build_packet", build)
    monkeypatch.setattr(module, "validate_packet_directory", validate)
    line = module.build_one(job)

    assert build_calls[0][1] == directory
    assert build_calls[0][3]["packet_class"] == "qualification"
    assert build_calls[0][3]["graded_authorization"] is None
    assert validation_calls[0][1]["expected_packet_class"] == "qualification"
    assert line.startswith("qualification/qual-0: files 2 ")
    assert str(job["entry"]["seed"]) not in line
