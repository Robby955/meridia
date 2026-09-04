"""Build the committed version-four world set: development, qualification, and graded.

Three families, one script, so the world size is written once and never drifts between
them.

- ``development``: twelve worlds, one per row of the committed twelve-run design, under
  the development source regime. These are the worlds a method may tune on, and their
  seeds are committed here.
- ``qualification``: six worlds under the hidden source regime, minted before any graded
  world. Thresholds are frozen on these and on nothing else, so a seed written here is
  the whole world a bar was measured on written here. ``qualification_seeds`` reads them
  from a sealed JSON file outside the repository, at
  ``~/.config/meridia/v4_qualification_seeds.json`` or wherever
  ``MERIDIA_QUALIFICATION_SEED_FILE`` points, and refuses a missing, malformed, short or
  repeating file by name. The world count stays here because the freeze design publishes
  it; only the values are sealed.
- ``graded``: independent worlds under the hidden source regime, minted after the
  thresholds are frozen and never read back into them. Their seeds are derived from a
  keyed V4 seal only after the frozen-bar and reserve-rate receipts authenticate both the
  seal and the exact hidden packet law. They are never printed, written to the participant
  packet, or committed. A seed remains in retained metadata needed to authenticate the
  sealed packet. A world's whole configuration follows from its seed, so a graded seed in
  the participant tree or a build log would disclose the graded configuration.

Every packet is a deterministic function of its seed and the shared parameters. Two
flags divide the work and neither changes the output. ``--workers`` divides one world's
continuation ensemble between processes; ``--world-workers`` builds whole worlds at once.
The second is the one that scales: the ensemble step is what a packet costs, it holds no
lock and shares nothing with the baseline step, and a world's ledger is a single
process's work whatever the ensemble is doing. Use one or the other, since together they
oversubscribe the machine.

``--cache`` points at a directory of continuation ensembles keyed on the digest of the
baseline ledger that produced them. A rebuild that changes only what a verifier or a bar
reads takes the futures off the shelf instead of paying for them again.

``--ensemble-members`` builds a world at a smaller continuation ensemble than the
committed one. It exists for the identifiability preflight, which reads participant files
the ensemble does not enter, and it is refused for a graded world. Evidence a bar reads is
measured on the committed size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.graded_readiness import validate_graded_readiness
from meridia.mechanisms import DEVELOPMENT_DESIGN
from meridia.packet import (
    GRADING_WORLD,
    PacketParams,
    _finalize_packet_build_intent,
    build_packet,
    validate_packet_directory,
)
from meridia.sealing import (V4PublicationAuthorization, V4WorldAuthorization,
                             verify_and_derive_v4_seed)

WORLD = GRADING_WORLD

DEVELOPMENT_SEEDS = tuple(1101 + i for i in range(len(DEVELOPMENT_DESIGN)))
QUALIFICATION_WORLD_COUNT = 6
QUALIFICATION_SEED_FILE_ENV = "MERIDIA_QUALIFICATION_SEED_FILE"
QUALIFICATION_SEED_KEY = "qualification_seeds"
def default_qualification_seed_file() -> Path:
    """Where the sealed qualification seeds live when the environment names nothing."""
    return Path.home() / ".config" / "meridia" / "v4_qualification_seeds.json"


def qualification_seed_file() -> Path:
    """The sealed file this run reads, from the environment or the default path."""
    named = os.environ.get(QUALIFICATION_SEED_FILE_ENV)
    return Path(named) if named else default_qualification_seed_file()


def qualification_seeds(path: Path | None = None) -> tuple[int, ...]:
    """The six qualification seeds, read from a sealed file outside the repository.

    A world's whole configuration follows from its seed, so the qualification seeds in
    the tree are the six worlds every bar is frozen on written into the tree. They are
    read the way a sealed input is read: from a JSON object outside the repository, at
    the path ``MERIDIA_QUALIFICATION_SEED_FILE`` names or at
    ``~/.config/meridia/v4_qualification_seeds.json``, under the single key
    ``qualification_seeds``.

    Every refusal names the file and the fault and carries no seed value. The world count
    stays in the tree because it is published in the freeze design; only the values are
    sealed.
    """
    source = Path(path) if path is not None else qualification_seed_file()
    try:
        raw = source.read_text()
    except OSError as exc:
        raise ValueError(
            f"sealed qualification seed file {source} could not be read"
        ) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"sealed qualification seed file {source} is not valid JSON"
        ) from exc
    if not isinstance(document, dict) or set(document) != {QUALIFICATION_SEED_KEY}:
        raise ValueError(
            f"sealed qualification seed file {source} must hold one object with the "
            f"single key {QUALIFICATION_SEED_KEY!r}"
        )
    values = document[QUALIFICATION_SEED_KEY]
    if not isinstance(values, list) or len(values) != QUALIFICATION_WORLD_COUNT:
        raise ValueError(
            f"sealed qualification seed file {source} must hold "
            f"{QUALIFICATION_WORLD_COUNT} qualification seeds"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError(
            f"sealed qualification seed file {source} holds a seed that is not a "
            "positive integer"
        )
    if len(set(values)) != len(values):
        raise ValueError(
            f"sealed qualification seed file {source} repeats a seed"
        )
    return tuple(int(value) for value in values)


GRADED_AUTHORIZATION_FIELDS = {
    "bars_path",
    "reserve_calibration_path",
    "seal_manifest_path",
    "key_path",
}


def _authorization_paths(
    bars_path: Path,
    reserve_calibration_path: Path,
    seal_manifest_path: Path,
    key_path: Path,
) -> dict[str, str]:
    """Serializable paths a worker must revalidate; none contains seed material."""
    return {
        "bars_path": str(Path(bars_path)),
        "reserve_calibration_path": str(Path(reserve_calibration_path)),
        "seal_manifest_path": str(Path(seal_manifest_path)),
        "key_path": str(Path(key_path)),
    }


def family_plan(
    family: str,
    *,
    bars_path: Path | None = None,
    reserve_calibration_path: Path | None = None,
    seal_manifest_path: Path | None = None,
    key_path: Path | None = None,
    ensemble_members: int | None = None,
) -> list[dict]:
    """The worlds of one family, with the parameters each is built under."""
    size = {}
    if ensemble_members is not None:
        if family == "graded":
            raise ValueError("a graded world is built at the committed ensemble size")
        size = {"ensemble_members": _worker_count(ensemble_members, "ensemble_members")}
    if family == "development":
        return [{"name": f"dev-{cell:02d}", "seed": seed,
                 "params": PacketParams(**{**WORLD.__dict__, "design_cell": cell,
                                           **size}),
                 "development": True, "public_seed": True}
                for cell, seed in enumerate(DEVELOPMENT_SEEDS)]
    if family == "qualification":
        return [{"name": f"qual-{i}", "seed": seed,
                 "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden",
                                           **size}),
                 "development": False, "public_seed": False}
                for i, seed in enumerate(qualification_seeds())]
    if family == "graded":
        if any(path is None for path in (
            bars_path,
            reserve_calibration_path,
            seal_manifest_path,
            key_path,
        )):
            raise ValueError(
                "graded worlds require bars, reserve calibration, a V4 seal, and its key"
            )
        params = PacketParams(**{**WORLD.__dict__, "regime": "hidden"})
        # This validation is deliberately inside the public planning API, not just its
        # CLI caller. No seal or key read is permitted before it succeeds.
        readiness = validate_graded_readiness(
            Path(bars_path),
            Path(reserve_calibration_path),
            expected_rate_per_person_year=params.reserve_rate_per_person_year,
        )
        authorizations = []
        for index in range(readiness.graded_world_count):
            authorizations.append(verify_and_derive_v4_seed(
                index,
                Path(seal_manifest_path),
                Path(key_path),
                params=params,
                readiness=readiness,
            ))
        bindings = {authorization.binding_sha256 for authorization in authorizations}
        if len(bindings) != 1:
            raise RuntimeError("V4 seal changed while the graded plan was authorized")
        return [{"name": f"graded-{index}",
                 "index": index,
                 "authorization_binding_sha256": authorization.binding_sha256,
                 "params": params,
                 "development": False, "public_seed": False}
                for index, authorization in enumerate(authorizations)]
    raise ValueError(f"unknown world family {family!r}")


FAMILIES = ("development", "qualification", "graded")


def positive_integer(value: str) -> int:
    """An argparse type for a nonzero process count."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _worker_count(value: object, label: str) -> int:
    """Validate process counts supplied through the Python API as strictly as the CLI."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def progress_line(family: str, entry: dict, n_files: int, seconds: float) -> str:
    """One line of build log. A seed reaches it only for the worlds a method may tune on."""
    # The family, not caller-supplied entry metadata, is the disclosure boundary. This
    # prevents a malformed or hand-built graded job from opting its seed into a log.
    named = f"seed {entry['seed']} " if family == "development" else ""
    return f"{family}/{entry['name']}: {named}files {n_files} {seconds:.0f}s"


def _authorize_graded_job(job: dict, entry: dict) -> V4WorldAuthorization:
    """Revalidate every graded-build authority inside the worker process."""
    expected_entry_fields = {
        "name",
        "index",
        "authorization_binding_sha256",
        "params",
        "development",
        "public_seed",
    }
    if not isinstance(entry, dict) or set(entry) != expected_entry_fields \
            or isinstance(entry.get("index"), bool) \
            or not isinstance(entry.get("index"), int) \
            or entry.get("index") < 0 \
            or entry.get("name") != f"graded-{entry.get('index')}" \
            or not isinstance(entry.get("authorization_binding_sha256"), str) \
            or len(entry.get("authorization_binding_sha256")) != 64 \
            or not set(entry.get("authorization_binding_sha256")) \
            <= set("0123456789abcdef") \
            or entry.get("development") is not False \
            or entry.get("public_seed") is not False:
        raise ValueError("graded job entry is not a canonical sealed-world entry")
    paths = job.get("graded_authorization")
    if not isinstance(paths, dict) or set(paths) != GRADED_AUTHORIZATION_FIELDS:
        raise ValueError("graded job lacks complete freeze and seal authorization")
    params = entry["params"]
    if not isinstance(params, PacketParams):
        raise ValueError("graded job parameters are malformed")
    readiness = validate_graded_readiness(
        Path(paths["bars_path"]),
        Path(paths["reserve_calibration_path"]),
        expected_rate_per_person_year=params.reserve_rate_per_person_year,
    )
    authorization = verify_and_derive_v4_seed(
        entry["index"],
        Path(paths["seal_manifest_path"]),
        Path(paths["key_path"]),
        params=params,
        readiness=readiness,
    )
    if authorization.binding_sha256 != entry["authorization_binding_sha256"]:
        raise RuntimeError("graded authorization differs from the approved build plan")
    return authorization


def _graded_publication_authority(
    before: V4WorldAuthorization,
    job: dict,
    entry: dict,
) -> V4PublicationAuthorization:
    paths = job["graded_authorization"]
    return V4PublicationAuthorization(
        before=before,
        index=entry["index"],
        seal_manifest_path=Path(paths["seal_manifest_path"]),
        key_path=Path(paths["key_path"]),
        bars_path=Path(paths["bars_path"]),
        reserve_calibration_path=Path(paths["reserve_calibration_path"]),
    )


def build_one(job: dict) -> str:
    """Build one world and return its log line. Runs in this process or a worker."""
    family, entry = job["family"], job["entry"]
    workers = _worker_count(job.get("workers"), "workers")
    world_workers = _worker_count(job.get("world_workers", 1), "world_workers")
    if workers > 1 and world_workers > 1:
        raise ValueError("workers and world_workers cannot both exceed one")
    if family not in FAMILIES:
        raise ValueError(f"unknown world family {family!r}")
    authorization = _authorize_graded_job(job, entry) if family == "graded" else None
    seed = authorization.seed if authorization is not None else entry["seed"]
    publication_authority = (
        _graded_publication_authority(authorization, job, entry)
        if authorization is not None else None
    )
    directory = Path(job["out"]) / family / entry["name"]
    if directory.exists():
        validate_packet_directory(
            directory,
            expected_packet_class=family,
            expected_params=entry["params"],
            expected_seed=seed,
        )
        if publication_authority is not None:
            publication_authority.confirm(seed=seed, params=entry["params"])
        _finalize_packet_build_intent(
            directory,
            seed=seed,
            params=entry["params"],
            packet_class=family,
            development=entry["development"],
            graded_authorization=publication_authority,
        )
        return f"{family}/{entry['name']}: already built"
    start = time.time()
    build_packet(
        seed,
        directory,
        entry["params"],
        development=entry["development"],
        workers=workers,
        cache_dir=Path(job["cache"]) if job.get("cache") else None,
        packet_class=family,
        graded_authorization=publication_authority,
    )
    manifest = validate_packet_directory(
        directory,
        expected_packet_class=family,
        expected_params=entry["params"],
        expected_seed=seed,
    )
    if manifest["development"] != entry["development"]:
        raise RuntimeError(f"{family}/{entry['name']} was written on the wrong side")
    n_files = len(manifest["participant"]) + len(manifest["retained"])
    return progress_line(family, entry, n_files, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--workers", type=positive_integer, default=1,
                        help="processes inside one world's continuation ensemble")
    parser.add_argument("--world-workers", type=positive_integer, default=1,
                        help="worlds built at once, each on its own process")
    parser.add_argument("--cache", type=Path, default=None,
                        help="directory of continuation ensembles, keyed on the "
                             "baseline ledger digest")
    parser.add_argument("--ensemble-members", type=positive_integer, default=None,
                        help="continuation members per world, for a preflight that "
                             "reads participant files only; the committed size is the "
                             "default and the only size a freeze may read")
    parser.add_argument("--bars", type=Path,
                        help="frozen composite-bar receipt (required for graded)")
    parser.add_argument(
        "--reserve-calibration-audit",
        type=Path,
        help="accepted reserve-rate audit frozen into the bars (required for graded)",
    )
    parser.add_argument("--seal-manifest", type=Path,
                        help="params-aware V4 seal manifest (required for graded)")
    parser.add_argument("--key", type=Path,
                        help="master key for the V4 seal (required for graded)")
    args = parser.parse_args()
    if args.workers > 1 and args.world_workers > 1:
        parser.error("--workers and --world-workers cannot both exceed one")

    if args.family == "graded":
        if any(value is None for value in (
            args.bars,
            args.reserve_calibration_audit,
            args.seal_manifest,
            args.key,
        )):
            parser.error(
                "graded builds require --bars, --reserve-calibration-audit, "
                "--seal-manifest, and --key"
            )
    if args.family == "graded" and args.ensemble_members is not None:
        parser.error("a graded world is built at the committed ensemble size")
    plan = family_plan(
        args.family,
        bars_path=args.bars,
        reserve_calibration_path=args.reserve_calibration_audit,
        seal_manifest_path=args.seal_manifest,
        key_path=args.key,
        ensemble_members=args.ensemble_members,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    graded_authorization = None
    if args.family == "graded":
        graded_authorization = _authorization_paths(
            args.bars,
            args.reserve_calibration_audit,
            args.seal_manifest,
            args.key,
        )
    jobs = [{"family": args.family, "entry": entry, "out": str(args.out),
             "workers": args.workers,
             "world_workers": args.world_workers,
             "cache": str(args.cache) if args.cache else None,
             **({"graded_authorization": graded_authorization}
                if graded_authorization is not None else {})}
            for entry in plan]
    if args.world_workers > 1:
        with ProcessPoolExecutor(max_workers=args.world_workers) as pool:
            for line in pool.map(build_one, jobs):
                print(line, flush=True)
        return
    for job in jobs:
        print(build_one(job), flush=True)


if __name__ == "__main__":
    main()
