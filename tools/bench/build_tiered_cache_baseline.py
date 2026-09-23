#!/usr/bin/env python3
"""Build and identify the recorded tiered-cache comparison executable."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.tiered_cache_replay import BASELINE_REVISION, sha256_file, write_json


def captured(command: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    return (result.stdout + result.stderr).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args(argv)
    if args.parallel <= 0:
        parser.error("--parallel must be positive")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    revision = captured(
        ["git", "rev-parse", f"{BASELINE_REVISION}^{{commit}}"], cwd=REPO_ROOT
    )
    if revision != BASELINE_REVISION:
        raise RuntimeError(f"baseline revision resolved unexpectedly: {revision}")

    archive = output / "source.tar"
    source = output / "source"
    build = output / "build"
    source.mkdir()
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={archive}", revision],
        cwd=REPO_ROOT,
        check=True,
    )
    archive_sha256 = sha256_file(archive)
    with tarfile.open(archive, "r") as bundle:
        bundle.extractall(source, filter="data")

    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=89",
        "-DNINFER_BUILD_APPS=ON",
        "-DBUILD_TESTING=OFF",
        "-DNINFER_BUILD_BENCHMARKS=OFF",
    ]
    build_command = [
        "cmake",
        "--build",
        str(build),
        "--parallel",
        str(args.parallel),
        "--target",
        "ninfer-serve",
    ]
    subprocess.run(configure, cwd=REPO_ROOT, check=True)
    subprocess.run(build_command, cwd=REPO_ROOT, check=True)
    serve = build / "apps/ninfer-serve"
    if not serve.is_file() or not os.access(serve, os.X_OK):
        raise RuntimeError(f"baseline build produced no executable: {serve}")

    identity = {
        "artifact_type": "ninfer_tiered_cache_baseline_build",
        "schema_version": 1,
        "source_revision": revision,
        "source_archive_sha256": archive_sha256,
        "serve_path": str(serve),
        "serve_sha256": sha256_file(serve),
        "build_command": [*configure, "&&", *build_command],
        "compiler": {
            "cmake": captured(["cmake", "--version"], cwd=REPO_ROOT).splitlines()[0],
            "nvcc": captured(["nvcc", "--version"], cwd=REPO_ROOT).splitlines()[-1],
        },
    }
    manifest = output / "baseline-build.json"
    write_json(manifest, identity)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
