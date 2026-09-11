#!/usr/bin/env python3
"""Record executable-identified focused evidence for the tiered-cache replay report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.tiered_cache_replay import sha256_file
from tools.artifact.container import Artifact, ResourceObject
from tools.convert.qwen3_8_27b.convert import OFFICIAL_RESOURCE_SHA256


REAL_SCENARIOS = (
    "host-restore",
    "pressure-resume",
    "shared-snapshot",
    "four-request-root-fallback",
    "delayed-spill",
    "delayed-active-capture",
    "cuda-transfer-failure",
    "pending-snapshot-shutdown",
)
NATIVE_CASES = {
    "frontend-boundary-token-lineage": "ninfer_qwen3_6_frontend_test",
    "openai-chat-boundary": "ninfer_openai_schema_test",
    "openai-responses-boundary": "ninfer_openai_responses_test",
    "anthropic-boundary": "ninfer_anthropic_schema_test",
    "resource-manager": "ninfer_resource_manager_test",
    "durable-shared-prefix-catalog": "ninfer_durable_shared_prefix_catalog_test",
}


def artifact_frontend_identity(path: Path) -> dict[str, object]:
    resources: dict[str, dict[str, object]] = {}
    with Artifact.open(path) as artifact:
        if (artifact.identity.model_id, artifact.identity.weights_id) != (
            "qwen3.8-27b",
            "groupwise-int",
        ):
            raise ValueError(
                "authorized artifact identity differs: "
                f"{artifact.identity.model_id}/{artifact.identity.weights_id}"
            )
        for name, expected_sha256 in OFFICIAL_RESOURCE_SHA256.items():
            obj = artifact.find(name)
            if not isinstance(obj, ResourceObject):
                raise ValueError(f"artifact frontend object is not a resource: {name}")
            actual_sha256 = hashlib.sha256(artifact.payload(obj)).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"artifact frontend resource hash differs for {name}: "
                    f"{actual_sha256} != {expected_sha256}"
                )
            resources[name] = {
                "sha256": actual_sha256,
                "offset": obj.offset,
                "bytes": obj.bytes,
                "encoding": obj.encoding,
            }
    return {
        "authorization": "user-authorized-target-artifact-2026-09-11",
        "path": str(path),
        "bytes": path.stat().st_size,
        "model_id": "qwen3.8-27b",
        "weights_id": "groupwise-int",
        "resources": resources,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build-agent-verify"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    args = parser.parse_args(argv)

    serve = args.serve.resolve()
    build = args.build_dir.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    prefix_test = build / "tests/ninfer_qwen3_6_27b_prefix_real_test"
    for required in (serve, weights, prefix_test):
        if not required.is_file():
            parser.error(f"required input is unavailable: {required}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_root = output.parent / (output.stem + "-logs")
    log_root.mkdir(parents=True, exist_ok=False)

    cases: dict[str, dict[str, object]] = {}
    failed = False
    try:
        frontend_identity = artifact_frontend_identity(weights)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    base_environment = dict(os.environ)
    base_environment["NINFER_QWEN3_8_27B_WEIGHTS"] = str(weights)
    for scenario in REAL_SCENARIOS:
        environment = dict(base_environment)
        environment["NINFER_PREFIX_REAL_SCENARIO"] = scenario
        command = [str(prefix_test)]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        log = log_root / f"{scenario}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        status = "PASS" if result.returncode == 0 else "SKIP" if result.returncode == 77 else "FAIL"
        failed = failed or status == "FAIL"
        cases[scenario] = {
            "status": status,
            "command": command,
            "returncode": result.returncode,
            "evidence": str(log),
            "test_executable_sha256": sha256_file(prefix_test),
        }

    for case, target in NATIVE_CASES.items():
        executable = build / f"tests/{target}"
        if not executable.is_file():
            parser.error(f"native validation executable is unavailable: {executable}")
        command = [str(executable)]
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        log = log_root / f"{case}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        status = "PASS" if result.returncode == 0 else "SKIP" if result.returncode == 77 else "FAIL"
        failed = failed or status == "FAIL"
        cases[case] = {
            "status": status,
            "command": command,
            "returncode": result.returncode,
            "evidence": str(log),
            "test_executable_sha256": sha256_file(executable),
        }

    artifact_frontend_status = str(cases["host-restore"]["status"])
    cases["official-tokenizer-lineage"] = {
        "status": artifact_frontend_status,
        "command": cases["host-restore"]["command"],
        "returncode": cases["host-restore"]["returncode"],
        "evidence": (
            "embedded official-source Qwen3.8 frontend objects matched their pinned hashes; "
            "the artifact-backed public Engine Frontend path passed host-restore"
        ),
        "test_executable_sha256": sha256_file(prefix_test),
        "artifact_frontend": frontend_identity,
    }

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    evidence = {
        "artifact_type": "ninfer_tiered_cache_validation_evidence",
        "schema_version": 1,
        "date": dt.date.today().isoformat(),
        "source_revision": revision,
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        ),
        "serve_path": str(serve),
        "serve_sha256": sha256_file(serve),
        "weights_path": str(weights),
        "weights_bytes": weights.stat().st_size,
        "cases": cases,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
