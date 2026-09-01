"""Sealing: no seed leakage, reproducible digests, tamper detection."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.sealing import (create_master_key, generate_and_digest, seal_worlds,
                             sealed_seed, verify_sealed_world)
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
