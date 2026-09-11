#!/usr/bin/env python3
"""Record executable-identified focused evidence for the tiered-cache replay report."""

from __future__ import annotations

import argparse
import datetime as dt
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


REAL_SCENARIOS = (
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
