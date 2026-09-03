"""The sealed builder authorizes grading before it opens any seed material."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from meridia.sealing import V4WorldAuthorization
import meridia.sealing as sealing


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_sealed_reconstruction_packet.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("sealed_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _argv(tmp_path: Path) -> list[str]:
    return [
        "--seal-manifest",
        str(tmp_path / "seal.json"),
        "--index",
        "0",
        "--out",
        str(tmp_path / "packet"),
        "--key",
        str(tmp_path / "key"),
        "--bars",
        str(tmp_path / "bars.json"),
        "--reserve-calibration-audit",
        str(tmp_path / "reserve.json"),
    ]


def test_freeze_failure_happens_before_seal_or_key_reads(tmp_path, monkeypatch):
    module = _module()
    calls = []

    def refuse(*args, **kwargs):
        calls.append("readiness")
        raise ValueError("freeze incomplete")

    monkeypatch.setattr(module, "validate_graded_readiness", refuse)
    with pytest.raises(ValueError, match="freeze incomplete"):
        module.main(_argv(tmp_path))
    assert calls == ["readiness"]


@pytest.mark.parametrize("workers", ["0", "-1", "not-an-integer"])
def test_worker_count_must_be_positive(tmp_path, workers):
    module = _module()
    with pytest.raises(SystemExit):
        module.main([*_argv(tmp_path), "--workers", workers])


def test_builder_marks_and_validates_a_graded_packet(tmp_path, monkeypatch, capsys):
    module = _module()
    seed = 987_654_321
    seal = tmp_path / "seal.json"
    key = tmp_path / "key"
    seal.write_text(json.dumps({"worlds": [{"index": 0}]}))
    key.write_bytes(b"fake key bytes")
    calls = []

    monkeypatch.setattr(
        module,
        "validate_graded_readiness",
        lambda *args, **kwargs: calls.append("readiness") or object(),
    )

    def authorize(*args, **kwargs):
        calls.append("authorize")
        return V4WorldAuthorization(seed=seed, binding_sha256="a" * 64)

    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(sealing, "validate_graded_readiness", module.validate_graded_readiness)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", authorize)

    def fake_build(actual_seed, out, params, **kwargs):
        assert actual_seed == seed
        assert kwargs["packet_class"] == "graded"
        assert kwargs["development"] is False
        authority = kwargs["graded_authorization"]
        assert isinstance(authority, module.V4PublicationAuthorization)
        calls.append("build")
        authority.confirm(seed=actual_seed, params=params)
        out.mkdir(parents=True)
        (out / "manifest.json").write_text("{}\n")

    monkeypatch.setattr(module, "build_packet", fake_build)

    def fake_validate(path, **kwargs):
        assert kwargs["expected_packet_class"] == "graded"
        assert kwargs["expected_seed"] == seed
        calls.append("validate")
        return {"development": False, "packet_class": "graded"}

    monkeypatch.setattr(module, "validate_packet_directory", fake_validate)
    assert module.main(_argv(tmp_path)) == 0
    output = capsys.readouterr().out
    assert str(seed) not in output
    assert calls == [
        "readiness", "authorize", "build", "readiness", "authorize", "validate"
    ]


def test_authorization_drift_never_publishes_the_graded_packet(tmp_path, monkeypatch):
    module = _module()
    calls = 0
    monkeypatch.setattr(
        module,
        "validate_graded_readiness",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        sealing,
        "validate_graded_readiness",
        module.validate_graded_readiness,
    )

    def authorize(*args, **kwargs):
        nonlocal calls
        calls += 1
        return V4WorldAuthorization(
            seed=987_654_321,
            binding_sha256=("a" if calls == 1 else "b") * 64,
        )

    def fake_build(seed, out, params, **kwargs):
        kwargs["graded_authorization"].confirm(seed=seed, params=params)
        out.mkdir(parents=True)

    monkeypatch.setattr(module, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", authorize)
    monkeypatch.setattr(module, "build_packet", fake_build)
    with pytest.raises(RuntimeError, match="authorization changed"):
        module.main(_argv(tmp_path))
    assert calls == 2
    assert not (tmp_path / "packet").exists()
