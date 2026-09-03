"""Sealing protocol: evaluation worlds nobody has seen, provably unchanged.

A sealed world's seed is derived from a master secret that never enters the repository:
seed_i = SHA-256(master_secret || index). The committed manifest records, per index,
only the digests of the generated layers. Nothing else is retained: generation runs
headless, arrays are hashed and discarded, and no render, summary, or statistic beyond
the digests exists anywhere. Anyone holding the manifest can later confirm that a world
used for grading is byte-identical to the one sealed on registration day; nobody without
the master secret can regenerate it, and nobody with it has looked.

The no-inspection rule is enforced by shape: `generate_and_digest` returns digests only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import secrets
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .character import draw_world_character
from .microdata import build_microdata
from .population import build_population, draw_national_total
from .terrain import generate_elevation
from .graded_readiness import GradedReadiness, validate_graded_readiness
from .packet import GRADING_WORLD, PacketParams, continuation_source_law_digest

DEFAULT_KEY_PATH = Path.home() / ".meridia" / "sealed_master.key"
GRID = (288, 384)
V4_SEAL_SCHEMA = "meridia.sealed-packet.v4"
V4_KDF_SCHEMA = "meridia.v4.kdf.hmac-sha256.v1"
V4_RUNTIME_LAW_SCHEMA = "meridia.v4.runtime-law.v1"
V4_PACKET_SOURCE_FILES = (
    "actuarial.py",
    "admin.py",
    "businesses.py",
    "character.py",
    "demography.py",
    "dwellings.py",
    "events.py",
    "hospitals.py",
    "hydrology.py",
    "identities.py",
    "mechanisms.py",
    "microdata.py",
    "packet.py",
    "population.py",
    "projection.py",
    "release.py",
    "sources.py",
    "survey.py",
    "terrain.py",
)
V4_MANIFEST_FIELDS = {
    "schema",
    "packet_class",
    "n_worlds",
    "packet_params",
    "packet_params_sha256",
    "generator_source_law_sha256",
    "continuation_source_law_sha256",
    "runtime_law",
    "runtime_law_sha256",
    "kdf_schema",
    "seal_nonce",
    "kdf_context_sha256",
    "freeze_receipts",
    "worlds",
}
V4_WORLD_FIELDS = {"index", "commitment"}
V4_FREEZE_RECEIPT_FIELDS = {"bars_sha256", "reserve_calibration_sha256"}


@dataclass(frozen=True)
class V4WorldAuthorization:
    """A verified seed plus the public binding that authorized its use.

    The seed is deliberately excluded from ``repr`` so an exception or debug rendering of
    the authorization object cannot put it in a terminal transcript.
    """

    seed: int = field(repr=False)
    binding_sha256: str


@dataclass(frozen=True)
class V4PublicationAuthorization:
    """Concrete authority that can re-open one registered world before publication.

    Packet construction accepts this object rather than an arbitrary callback. Its
    confirmation method receives no staging path, and it succeeds only when the current
    freeze, seal, key, parameters, seed, and initial binding reproduce the authorization
    obtained before construction.
    """

    before: V4WorldAuthorization = field(repr=False)
    index: int
    seal_manifest_path: Path
    key_path: Path = field(repr=False)
    bars_path: Path
    reserve_calibration_path: Path

    def __post_init__(self) -> None:
        if type(self.before) is not V4WorldAuthorization:
            raise TypeError("publication authority requires a V4 world authorization")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("publication authority index must be nonnegative")
        if not _is_sha256(self.before.binding_sha256):
            raise ValueError("publication authority binding must be canonical SHA-256")
        for name in (
            "seal_manifest_path",
            "key_path",
            "bars_path",
            "reserve_calibration_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))

    def assert_initial(self, *, seed: int) -> None:
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or seed != self.before.seed):
            raise RuntimeError("graded publication authority does not match the packet")

    def confirm(self, *, seed: int, params: PacketParams) -> V4WorldAuthorization:
        """Reauthorize current files and bind the result to the pre-build authority."""
        self.assert_initial(seed=seed)
        readiness = validate_graded_readiness(
            self.bars_path,
            self.reserve_calibration_path,
            expected_rate_per_person_year=params.reserve_rate_per_person_year,
        )
        after = verify_and_derive_v4_seed(
            self.index,
            self.seal_manifest_path,
            self.key_path,
            params=params,
            readiness=readiness,
        )
        if after != self.before:
            raise RuntimeError("graded authorization changed while the packet was built")
        return after


def _open_key_parent(path: Path, flags: int) -> int:
    """Open/create every parent component without following a symbolic link."""
    descriptor = os.open("/" if path.is_absolute() else ".", flags)
    parts = path.parent.parts[1:] if path.is_absolute() else path.parent.parts
    try:
        for part in parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise ValueError("master key path cannot traverse a parent directory")
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except OSError as error:
                    raise ValueError(
                        "master key parent must contain only real directories"
                    ) from error
            except OSError as error:
                raise ValueError(
                    "master key parent must contain only real directories"
                ) from error
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def create_master_key(path: Path = DEFAULT_KEY_PATH) -> Path:
    """Create a complete mode-600 key once without following or replacing a path."""
    path = Path(path)
    if not path.name or path.name in {".", ".."}:
        raise ValueError("master key path must name a file")
    if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise RuntimeError("this platform cannot create a symlink-safe master key")

    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        parent_descriptor = _open_key_parent(path, parent_flags)
    except (OSError, ValueError) as error:
        raise ValueError("master key parent must be a real directory") from error

    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            raise FileExistsError(f"master key already exists at {path}") from None
        opened = os.fstat(descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        os.fchmod(descriptor, 0o600)
        secret = secrets.token_bytes(32)
        view = memoryview(secret)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("master key write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise OSError("master key mode is not 0600")
    except BaseException:
        if created_identity is not None:
            try:
                current = os.stat(path.name, dir_fd=parent_descriptor,
                                  follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(path.name, dir_fd=parent_descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    return path


def sealed_seed(master: bytes, index: int) -> int:
    """Legacy V1 seed derivation. V4 must use :func:`v4_sealed_seed`."""
    digest = hashlib.sha256(master + index.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def v4_sealed_seed(master: bytes, index: int, kdf_context_sha256: str) -> int:
    """Derive one V4 seed in a domain disjoint from every legacy seal."""
    if not isinstance(master, bytes) or len(master) != 32:
        raise ValueError("V4 seed derivation requires a 32-byte master key")
    if isinstance(index, bool) or not isinstance(index, int) \
            or not 0 <= index < 2**64:
        raise ValueError("V4 world index must be an unsigned 64-bit integer")
    if not _is_sha256(kdf_context_sha256):
        raise ValueError("V4 KDF context digest must be canonical SHA-256")
    payload = b"\0".join((
        V4_KDF_SCHEMA.encode("ascii"),
        bytes.fromhex(kdf_context_sha256),
        index.to_bytes(8, "big"),
    ))
    digest = hmac.new(master, payload, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def generate_and_digest(seed: int) -> dict:
    """Generate a full world and return layer digests only; arrays are discarded."""
    height, width = GRID
    character = draw_world_character(seed)
    world = generate_elevation(seed, height, width)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    total = draw_national_total(seed, int(world["land"].sum()))
    people = build_population(world, accumulation, total, 24,
                              params=character["population"], seed=seed)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    digests = {
        "elevation": _digest(world["elevation"]),
        "flow_direction": _digest(direction),
        "population_grid": _digest(people["population"]),
        "person_table": hashlib.sha256(
            b"".join(np.ascontiguousarray(v).tobytes()
                     for v in micro["person"].values())).hexdigest(),
        "household_cells": _digest(micro["household_cell"]),
    }
    return digests


def seal_worlds(n_worlds: int, manifest_path: Path,
                key_path: Path = DEFAULT_KEY_PATH) -> dict:
    """Register and seal n worlds; writes the public manifest, returns it."""
    master = key_path.read_bytes()
    worlds = []
    for index in range(n_worlds):
        seed = sealed_seed(master, index)
        digests = generate_and_digest(seed)
        commitment = hashlib.sha256(
            master + index.to_bytes(8, "big") + b"commit").hexdigest()
        worlds.append({"index": index, "commitment": commitment, "digests": digests})
    manifest = {"schema": "meridia.sealed.v1", "grid": list(GRID),
                "n_worlds": n_worlds, "worlds": worlds}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_sealed_world(index: int, manifest_path: Path,
                        key_path: Path = DEFAULT_KEY_PATH) -> bool:
    """Regenerate world `index` from the secret and check every digest matches."""
    master = key_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    entry = next(w for w in manifest["worlds"] if w["index"] == index)
    digests = generate_and_digest(sealed_seed(master, index))
    return digests == entry["digests"]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _v4_params(params: PacketParams) -> tuple[PacketParams, dict]:
    hidden = PacketParams(**{**asdict(params), "regime": "hidden", "design_cell": None})
    return hidden, json.loads(_canonical_json(asdict(hidden)))


def v4_runtime_law() -> dict:
    """Return the runtime facts on which deterministic packet bytes may depend."""
    return {
        "schema": V4_RUNTIME_LAW_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def v4_runtime_law_digest() -> str:
    """Digest the exact interpreter and numerical-library law used by V4."""
    return hashlib.sha256(_canonical_json(v4_runtime_law())).hexdigest()


def _v4_kdf_context_digest(
    *,
    seal_nonce: str,
    params_sha256: str,
    generator_source_sha256: str,
    continuation_source_sha256: str,
    runtime_law_sha256: str,
    readiness: GradedReadiness,
) -> str:
    """Bind seed derivation to this freeze and this unique seal registration."""
    if not _is_sha256(seal_nonce):
        raise ValueError("V4 seal nonce must be 32 canonical hexadecimal bytes")
    record = {
        "kdf_schema": V4_KDF_SCHEMA,
        "seal_nonce": seal_nonce,
        "params_sha256": params_sha256,
        "generator_source_sha256": generator_source_sha256,
        "continuation_source_sha256": continuation_source_sha256,
        "runtime_law_sha256": runtime_law_sha256,
        "bars_sha256": readiness.bars_sha256,
        "reserve_calibration_sha256": readiness.reserve_calibration_sha256,
        "graded_world_count": readiness.graded_world_count,
    }
    return hashlib.sha256(_canonical_json(record)).hexdigest()


def v4_generator_source_law_digest() -> str:
    """Bind every source module that can change a deterministic V4 packet."""
    digest = hashlib.sha256(b"meridia.v4.packet-source-law.v1")
    continuation = continuation_source_law_digest()
    digest.update(bytes.fromhex(continuation))
    source_root = Path(__file__).resolve().parent
    for name in V4_PACKET_SOURCE_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256((source_root / name).read_bytes()).digest())
    return digest.hexdigest()


def _v4_commitment(
    master: bytes,
    index: int,
    seed: int,
    params_sha256: str,
    generator_source_sha256: str,
    runtime_law_sha256: str,
    kdf_context_sha256: str,
    readiness: GradedReadiness,
) -> str:
    payload = _canonical_json(
        {
            "schema": V4_SEAL_SCHEMA,
            "kdf_schema": V4_KDF_SCHEMA,
            "index": index,
            "derived_seed": seed,
            "params_sha256": params_sha256,
            "generator_source_sha256": generator_source_sha256,
            "runtime_law_sha256": runtime_law_sha256,
            "kdf_context_sha256": kdf_context_sha256,
            "bars_sha256": readiness.bars_sha256,
            "reserve_calibration_sha256": readiness.reserve_calibration_sha256,
            "graded_world_count": readiness.graded_world_count,
        }
    )
    return hmac.new(master, payload, hashlib.sha256).hexdigest()


def _master_key(path: Path) -> bytes:
    source = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError("sealed master key must be a regular file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("sealed master key must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            master = handle.read()
    finally:
        os.close(descriptor)
    if len(master) != 32:
        raise ValueError("sealed master key must contain exactly 32 bytes")
    return master


def _read_v4_manifest(path: Path) -> tuple[dict, str]:
    source = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ValueError("V4 seal manifest must be a regular JSON file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("V4 seal manifest must be a regular JSON file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("V4 seal manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("V4 seal manifest must be a JSON object")
    return manifest, hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(character in "0123456789abcdef" for character in value)


def seal_v4_worlds(
    n_worlds: int,
    manifest_path: Path,
    *,
    bars_path: Path,
    reserve_calibration_path: Path,
    key_path: Path = DEFAULT_KEY_PATH,
    params: PacketParams = GRADING_WORLD,
) -> dict:
    """Commit unopened V4 graded seeds to the frozen contract and generator law.

    Receipt validation precedes the master-key read. The public manifest contains only
    keyed commitments, canonical packet parameters, source digests, and receipt digests;
    it never contains a derived seed.
    """
    if isinstance(n_worlds, bool) or not isinstance(n_worlds, int) or n_worlds < 1:
        raise ValueError("n_worlds must be a positive integer")
    destination = Path(manifest_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"seal manifest already exists: {destination}")
    hidden, params_record = _v4_params(params)
    readiness = validate_graded_readiness(
        bars_path,
        reserve_calibration_path,
        expected_rate_per_person_year=hidden.reserve_rate_per_person_year,
    )
    if n_worlds != readiness.graded_world_count:
        raise ValueError("seal count differs from the frozen graded world count")
    master = _master_key(key_path)
    source_digest = v4_generator_source_law_digest()
    continuation_digest = continuation_source_law_digest()
    runtime_record = v4_runtime_law()
    runtime_digest = hashlib.sha256(_canonical_json(runtime_record)).hexdigest()
    params_digest = hashlib.sha256(_canonical_json(params_record)).hexdigest()
    seal_nonce = secrets.token_hex(32)
    kdf_context_digest = _v4_kdf_context_digest(
        seal_nonce=seal_nonce,
        params_sha256=params_digest,
        generator_source_sha256=source_digest,
        continuation_source_sha256=continuation_digest,
        runtime_law_sha256=runtime_digest,
        readiness=readiness,
    )
    manifest = {
        "schema": V4_SEAL_SCHEMA,
        "packet_class": "graded",
        "n_worlds": n_worlds,
        "packet_params": params_record,
        "packet_params_sha256": params_digest,
        "generator_source_law_sha256": source_digest,
        "continuation_source_law_sha256": continuation_digest,
        "runtime_law": runtime_record,
        "runtime_law_sha256": runtime_digest,
        "kdf_schema": V4_KDF_SCHEMA,
        "seal_nonce": seal_nonce,
        "kdf_context_sha256": kdf_context_digest,
        "freeze_receipts": {
            "bars_sha256": readiness.bars_sha256,
            "reserve_calibration_sha256": readiness.reserve_calibration_sha256,
        },
        "worlds": [
            {
                "index": index,
                "commitment": _v4_commitment(
                    master,
                    index,
                    v4_sealed_seed(master, index, kdf_context_digest),
                    params_digest,
                    source_digest,
                    runtime_digest,
                    kdf_context_digest,
                    readiness,
                ),
            }
            for index in range(n_worlds)
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{destination.name}.",
            suffix=".partial", dir=destination.parent, delete=False
        ) as handle:
            scratch = Path(handle.name)
            handle.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(scratch, destination)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return manifest


def verify_and_derive_v4_seed(
    index: int,
    manifest_path: Path,
    key_path: Path = DEFAULT_KEY_PATH,
    *,
    params: PacketParams = GRADING_WORLD,
    readiness: GradedReadiness,
) -> V4WorldAuthorization:
    """Verify a V4 seal and derive its seed from the same opened key bytes.

    The returned binding hashes the exact public manifest bytes. Callers that validate
    before and after a build compare the complete authorization objects, so a receipt,
    source, runtime, seal, key, or seed change fails closed.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("V4 world index must be a non-negative integer")
    manifest, manifest_digest = _read_v4_manifest(manifest_path)
    if set(manifest) != V4_MANIFEST_FIELDS \
            or manifest.get("schema") != V4_SEAL_SCHEMA \
            or manifest.get("packet_class") != "graded" \
            or manifest.get("kdf_schema") != V4_KDF_SCHEMA:
        raise ValueError("V4 seal manifest schema is not exact")

    hidden, expected_params = _v4_params(params)
    if readiness.reserve_rate_per_person_year != hidden.reserve_rate_per_person_year:
        raise ValueError("V4 readiness rate differs from the packet parameters")
    if not _is_sha256(readiness.bars_sha256) \
            or not _is_sha256(readiness.reserve_calibration_sha256):
        raise ValueError("V4 readiness receipt digests are malformed")
    if manifest.get("packet_params") != expected_params:
        raise ValueError("V4 seal packet parameters have drifted")
    params_digest = hashlib.sha256(_canonical_json(expected_params)).hexdigest()
    source_digest = v4_generator_source_law_digest()
    continuation_digest = continuation_source_law_digest()
    runtime_record = v4_runtime_law()
    runtime_digest = hashlib.sha256(_canonical_json(runtime_record)).hexdigest()
    seal_nonce = manifest.get("seal_nonce")
    kdf_context_digest = _v4_kdf_context_digest(
        seal_nonce=seal_nonce,
        params_sha256=params_digest,
        generator_source_sha256=source_digest,
        continuation_source_sha256=continuation_digest,
        runtime_law_sha256=runtime_digest,
        readiness=readiness,
    )
    freeze_receipts = manifest.get("freeze_receipts")
    if not isinstance(freeze_receipts, dict) \
            or set(freeze_receipts) != V4_FREEZE_RECEIPT_FIELDS \
            or freeze_receipts != {
                "bars_sha256": readiness.bars_sha256,
                "reserve_calibration_sha256": readiness.reserve_calibration_sha256,
            } \
            or manifest.get("packet_params_sha256") != params_digest \
            or manifest.get("generator_source_law_sha256") != source_digest \
            or manifest.get("continuation_source_law_sha256") != continuation_digest \
            or manifest.get("runtime_law") != runtime_record \
            or manifest.get("runtime_law_sha256") != runtime_digest \
            or manifest.get("kdf_context_sha256") != kdf_context_digest:
        raise ValueError("V4 seal law or freeze receipts have drifted")

    count = manifest.get("n_worlds")
    worlds = manifest.get("worlds")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 \
            or count != readiness.graded_world_count \
            or not isinstance(worlds, list) or len(worlds) != count \
            or not 0 <= index < count:
        raise ValueError("V4 seal world count or requested index is invalid")
    for expected_index, world in enumerate(worlds):
        if not isinstance(world, dict) or set(world) != V4_WORLD_FIELDS \
                or world.get("index") != expected_index \
                or not _is_sha256(world.get("commitment")):
            raise ValueError("V4 seal world entry schema is not exact")

    # This is the only master-key read in the operation. The seed and commitment are both
    # computed from this immutable byte string, closing the verify-then-reread race.
    master = _master_key(key_path)
    seed = v4_sealed_seed(master, index, kdf_context_digest)
    expected_commitment = _v4_commitment(
        master,
        index,
        seed,
        params_digest,
        source_digest,
        runtime_digest,
        kdf_context_digest,
        readiness,
    )
    if not hmac.compare_digest(worlds[index]["commitment"], expected_commitment):
        raise ValueError("V4 world commitment replay failed")
    return V4WorldAuthorization(seed=seed, binding_sha256=manifest_digest)


def verify_v4_sealed_world(
    index: int,
    manifest_path: Path,
    key_path: Path = DEFAULT_KEY_PATH,
    *,
    params: PacketParams = GRADING_WORLD,
    readiness: GradedReadiness,
) -> bool:
    """Compatibility predicate around the fail-loud one-shot V4 authorization API."""
    try:
        verify_and_derive_v4_seed(
            index,
            manifest_path,
            key_path,
            params=params,
            readiness=readiness,
        )
        return True
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return False
