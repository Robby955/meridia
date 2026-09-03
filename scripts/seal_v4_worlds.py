"""Commit V4 graded worlds after the bars and reserve rate are frozen."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.sealing import DEFAULT_KEY_PATH, seal_v4_worlds


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=_positive_int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--reserve-calibration-audit", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = seal_v4_worlds(
        args.count,
        args.out,
        bars_path=args.bars,
        reserve_calibration_path=args.reserve_calibration_audit,
        key_path=args.key,
    )
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"V4_WORLDS_SEALED count={manifest['n_worlds']} manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
