"""Sealing: no seed leakage, reproducible digests, tamper detection."""

import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.graded_readiness import GradedReadiness
from meridia.packet import GRADING_WORLD
from meridia.sealing import (V4PublicationAuthorization, V4WorldAuthorization,
                             create_master_key, generate_and_digest, seal_v4_worlds,
                             seal_worlds, sealed_seed, v4_sealed_seed,
                             verify_and_derive_v4_seed, verify_sealed_world,
                             verify_v4_sealed_world)
import meridia.sealing as sealing


def _small_grid(monkeypatch):
    monkeypatch.setattr(sealing, "GRID", (48, 64))


def test_seal_and_verify_roundtrip(tmp_path, monkeypatch):
    _small_grid(monkeypatch)
    key = create_master_key(tmp_path / "master.key")
    manifest_path = tmp_path / "manifest.json"
    manifest = seal_worlds(2, manifest_path, key_path=key)
    assert manifest["n_worlds"] == 2
    assert verify_sealed_world(0, manifest_path, key_path=key)
    assert verify_sealed_world(1, manifest_path, key_path=key)


def test_manifest_contains_no_seed(tmp_path, monkeypatch):
    _small_grid(monkeypatch)
    key = create_master_key(tmp_path / "master.key")
    manifest_path = tmp_path / "manifest.json"
    seal_worlds(1, manifest_path, key_path=key)
    text = manifest_path.read_text()
    seed = sealed_seed(key.read_bytes(), 0)
    assert str(seed) not in text
    assert "seed" not in json.loads(text)["worlds"][0]


def test_tampered_manifest_fails_verification(tmp_path, monkeypatch):
    _small_grid(monkeypatch)
    key = create_master_key(tmp_path / "master.key")
    manifest_path = tmp_path / "manifest.json"
    seal_worlds(1, manifest_path, key_path=key)
    manifest = json.loads(manifest_path.read_text())
    manifest["worlds"][0]["digests"]["elevation"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    assert not verify_sealed_world(0, manifest_path, key_path=key)


def test_digests_deterministic(monkeypatch):
    _small_grid(monkeypatch)
    assert generate_and_digest(12345) == generate_and_digest(12345)


def test_master_key_never_overwritten(tmp_path):
    key = create_master_key(tmp_path / "master.key")
    try:
        create_master_key(key)
        raised = False
    except FileExistsError:
        raised = True
    assert raised


def test_master_key_is_exclusive_nofollow_and_mode_600_at_creation(
    tmp_path, monkeypatch
):
    real_open = os.open
    created = []

    def observed_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "master.key":
            created.append((flags, mode, stat.S_IMODE(os.fstat(descriptor).st_mode)))
        return descriptor

    monkeypatch.setattr(sealing.os, "open", observed_open)
    key = create_master_key(tmp_path / "master.key")
    assert key.read_bytes() and len(key.read_bytes()) == 32
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert len(created) == 1
    flags, requested_mode, creation_mode = created[0]
    assert flags & os.O_EXCL
    assert flags & os.O_NOFOLLOW
    assert requested_mode == 0o600
    assert creation_mode == 0o600


def test_master_key_rejects_file_and_parent_symlinks(tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"do not overwrite")
    link = tmp_path / "master.key"
    link.symlink_to(victim)
    with pytest.raises(FileExistsError, match="already exists"):
        create_master_key(link)
    assert victim.read_bytes() == b"do not overwrite"
    assert link.is_symlink()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="parent must be a real directory"):
        create_master_key(linked_parent / "other.key")
    assert not (real_parent / "other.key").exists()

    real_ancestor = tmp_path / "real-ancestor"
    real_ancestor.mkdir()
    linked_ancestor = tmp_path / "linked-ancestor"
    linked_ancestor.symlink_to(real_ancestor, target_is_directory=True)
    with pytest.raises(ValueError, match="parent must be a real directory"):
        create_master_key(linked_ancestor / "nested" / "other.key")
    assert not (real_ancestor / "nested").exists()


def test_publication_authority_rejects_world_authorization_subclasses(tmp_path):
    class ForgedWorldAuthorization(V4WorldAuthorization):
        def __eq__(self, other):
            del other
            return True

    with pytest.raises(TypeError, match="requires a V4 world authorization"):
        V4PublicationAuthorization(
            before=ForgedWorldAuthorization(seed=7, binding_sha256="a" * 64),
            index=0,
            seal_manifest_path=tmp_path / "seal.json",
            key_path=tmp_path / "key",
            bars_path=tmp_path / "bars.json",
            reserve_calibration_path=tmp_path / "reserve.json",
        )


def test_failed_master_key_write_removes_the_exclusive_partial_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sealing.os,
        "write",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    key = tmp_path / "master.key"
    with pytest.raises(OSError, match="write failed"):
        create_master_key(key)
    assert not key.exists() and not key.is_symlink()


def _v4_readiness() -> GradedReadiness:
    return GradedReadiness(
        bars_sha256="a" * 64,
        reserve_calibration_sha256="b" * 64,
        reserve_rate_per_person_year=GRADING_WORLD.reserve_rate_per_person_year,
        graded_world_count=3,
    )


def test_publication_authority_binds_seed_and_post_build_authorization(
    tmp_path, monkeypatch
):
    before = V4WorldAuthorization(seed=123456, binding_sha256="a" * 64)
    authority = V4PublicationAuthorization(
        before=before,
        index=0,
        seal_manifest_path=tmp_path / "seal.json",
        key_path=tmp_path / "key",
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
    )
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    monkeypatch.setattr(sealing, "verify_and_derive_v4_seed", lambda *a, **k: before)
    assert authority.confirm(seed=before.seed, params=GRADING_WORLD) == before

    with pytest.raises(RuntimeError, match="does not match the packet"):
        authority.confirm(seed=before.seed + 1, params=GRADING_WORLD)
    monkeypatch.setattr(
        sealing,
        "verify_and_derive_v4_seed",
        lambda *a, **k: V4WorldAuthorization(seed=before.seed, binding_sha256="b" * 64),
    )
    with pytest.raises(RuntimeError, match="authorization changed"):
        authority.confirm(seed=before.seed, params=GRADING_WORLD)


def test_v4_seal_binds_params_source_law_and_freeze_receipts(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    key = create_master_key(tmp_path / "v4.key")
    manifest_path = tmp_path / "v4-seal.json"
    manifest = seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    assert manifest["schema"] == sealing.V4_SEAL_SCHEMA
    assert manifest["packet_class"] == "graded"
    assert manifest["freeze_receipts"] == {
        "bars_sha256": readiness.bars_sha256,
        "reserve_calibration_sha256": readiness.reserve_calibration_sha256,
    }
    assert verify_v4_sealed_world(
        2, manifest_path, key, readiness=readiness
    )
    assert "seed" not in manifest_path.read_text()

    with pytest.raises(ValueError, match="graded world count"):
        seal_v4_worlds(
            2,
            tmp_path / "wrong-count.json",
            bars_path=tmp_path / "bars.json",
            reserve_calibration_path=tmp_path / "reserve.json",
            key_path=key,
        )


def test_v4_seal_rejects_parameter_source_and_receipt_drift(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    key = create_master_key(tmp_path / "v4.key")
    manifest_path = tmp_path / "v4-seal.json"
    seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    changed_params = replace(GRADING_WORLD, total=GRADING_WORLD.total + 1)
    assert not verify_v4_sealed_world(
        0, manifest_path, key, params=changed_params, readiness=readiness
    )
    changed_receipts = replace(readiness, bars_sha256="c" * 64)
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=changed_receipts
    )
    original = sealing.v4_generator_source_law_digest
    monkeypatch.setattr(
        sealing,
        "v4_generator_source_law_digest",
        lambda: "d" * 64,
    )
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=readiness
    )
    monkeypatch.setattr(sealing, "v4_generator_source_law_digest", original)


def test_v4_seed_domain_is_distinct_and_bound_into_the_commitment(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    key = create_master_key(tmp_path / "v4.key")
    master = key.read_bytes()
    assert v4_sealed_seed(master, 0, "c" * 64) != sealed_seed(master, 0)

    manifest_path = tmp_path / "v4-seal.json"
    seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    original = sealing.v4_sealed_seed
    monkeypatch.setattr(
        sealing,
        "v4_sealed_seed",
        lambda actual_master, index: original(actual_master, index) ^ 1,
    )
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=readiness
    )


def test_v4_manifest_schema_rejects_seed_bearing_or_other_extra_fields(
    tmp_path, monkeypatch
):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    key = create_master_key(tmp_path / "v4.key")
    manifest_path = tmp_path / "v4-seal.json"
    original = seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )

    top_level = dict(original, seed=123)
    manifest_path.write_text(json.dumps(top_level))
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=readiness
    )

    world_level = json.loads(json.dumps(original))
    world_level["worlds"][0]["seed"] = 123
    manifest_path.write_text(json.dumps(world_level))
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=readiness
    )


def test_v4_runtime_law_is_deterministic_and_enforced(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    assert sealing.v4_runtime_law_digest() == sealing.v4_runtime_law_digest()
    key = create_master_key(tmp_path / "v4.key")
    manifest_path = tmp_path / "v4-seal.json"
    seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    runtime = sealing.v4_runtime_law()
    runtime["dependencies"]["numpy"] = "different"
    monkeypatch.setattr(sealing, "v4_runtime_law", lambda: runtime)
    assert not verify_v4_sealed_world(
        0, manifest_path, key, readiness=readiness
    )


def test_one_shot_authorization_reads_the_key_once(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    key = create_master_key(tmp_path / "v4.key")
    manifest_path = tmp_path / "v4-seal.json"
    seal_v4_worlds(
        3,
        manifest_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    original = sealing._master_key
    calls = []

    def read_once(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(sealing, "_master_key", read_once)
    authorization = verify_and_derive_v4_seed(
        1, manifest_path, key, readiness=readiness
    )
    assert calls == [key]
    kdf_context = json.loads(manifest_path.read_text())["kdf_context_sha256"]
    assert authorization.seed == v4_sealed_seed(key.read_bytes(), 1, kdf_context)
    assert str(authorization.seed) not in repr(authorization)


def test_resealing_the_same_index_mints_a_new_v4_world(tmp_path, monkeypatch):
    readiness = _v4_readiness()
    monkeypatch.setattr(sealing, "validate_graded_readiness", lambda *a, **k: readiness)
    nonces = iter(("1" * 64, "2" * 64))
    monkeypatch.setattr(sealing.secrets, "token_hex", lambda size: next(nonces))
    key = create_master_key(tmp_path / "v4.key")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = seal_v4_worlds(
        3,
        first_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    second = seal_v4_worlds(
        3,
        second_path,
        bars_path=tmp_path / "bars.json",
        reserve_calibration_path=tmp_path / "reserve.json",
        key_path=key,
    )
    assert first["seal_nonce"] != second["seal_nonce"]
    first_authorization = verify_and_derive_v4_seed(
        0, first_path, key, readiness=readiness
    )
    second_authorization = verify_and_derive_v4_seed(
        0, second_path, key, readiness=readiness
    )
    assert first_authorization.seed != second_authorization.seed
    assert first_authorization.binding_sha256 != second_authorization.binding_sha256


def test_v4_seal_checks_freeze_before_opening_the_master_key(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("freeze incomplete")

    monkeypatch.setattr(sealing, "validate_graded_readiness", refuse)
    with pytest.raises(ValueError, match="freeze incomplete"):
        seal_v4_worlds(
            1,
            tmp_path / "v4-seal.json",
            bars_path=tmp_path / "bars.json",
            reserve_calibration_path=tmp_path / "reserve.json",
            key_path=tmp_path / "missing.key",
        )
