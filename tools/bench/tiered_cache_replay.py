"""Shared validation and reporting helpers for the tiered-cache replay campaign."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_TYPE = "ninfer_tiered_cache_fixture_manifest"
MANIFEST_VERSION = 1
EVIDENCE_TYPE = "ninfer_tiered_cache_evidence"
EVIDENCE_VERSION = 5
CALIBRATION_TYPE = "ninfer_tiered_cache_predecessor_calibration"
CALIBRATION_VERSION = 1


class ReplayError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read fixture manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReplayError("fixture manifest is not an object")
    identity = (value.get("artifact_type"), value.get("schema_version"))
    if identity != (MANIFEST_TYPE, MANIFEST_VERSION):
        raise ReplayError(f"unsupported fixture manifest identity: {identity!r}")
    return value


def validate_manifest(path: Path, *, verify_source: bool = False) -> dict[str, Any]:
    manifest = load_manifest(path)
    provenance = manifest.get("provenance")
    raw = manifest.get("raw_render_replay")
    cases = manifest.get("protocol_cases")
    measurement = manifest.get("measurement_fixture")
    if not isinstance(provenance, dict) or provenance.get("observed_txt_files") != 66:
        raise ReplayError("fixture provenance must record the bounded 66-file source corpus")
    if not isinstance(raw, dict) or raw.get("transport") != "none":
        raise ReplayError("raw-render replay must remain separate from protocol replay")
    raw_path = path.parent / str(raw.get("path", ""))
    if sha256_file(raw_path) != raw.get("sha256"):
        raise ReplayError(f"raw-render fixture hash mismatch: {raw_path}")
    if not isinstance(cases, list) or not cases:
        raise ReplayError("protocol fixture list is empty")
    required_shapes = {
        "harness-only",
        "project-envelope",
        "mixed-harness-project",
        "volatile-fields",
        "ordinary-conversation-lookalike",
    }
    actual_shapes: set[str] = set()
    protocols: set[str] = set()
    names: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ReplayError("protocol fixture is not an object")
        name = case.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ReplayError("protocol fixture names must be nonempty and unique")
        names.add(name)
        actual_shapes.add(str(case.get("shape")))
        protocols.add(str(case.get("protocol")))
        if not isinstance(case.get("stable"), str) or not isinstance(case.get("volatile"), str):
            raise ReplayError(f"protocol fixture {name} has no stable/volatile text")
    if actual_shapes != required_shapes:
        raise ReplayError(f"protocol fixture shapes differ: {sorted(actual_shapes)!r}")
    if protocols != {"openai_chat", "openai_responses", "anthropic_messages"}:
        raise ReplayError(f"protocol fixture coverage differs: {sorted(protocols)!r}")
    if not isinstance(measurement, dict) or measurement.get("case") not in names:
        raise ReplayError("measurement fixture does not select a protocol case")
    repetitions = measurement.get("stable_repetitions")
    suffixes = measurement.get("measured_suffixes")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ReplayError("measurement stable_repetitions must be positive")
    if not isinstance(suffixes, list) or len(suffixes) < 4:
        raise ReplayError("measurement fixture needs four distinct request suffixes")

    source_result = "not-requested"
    if verify_source:
        source_root = Path(str(provenance.get("source_path", "")))
        if not source_root.is_dir():
            raise ReplayError(f"authorized source corpus is unavailable: {source_root}")
        source_files = sorted(source_root.glob("*.txt"))
        if len(source_files) != provenance["observed_txt_files"]:
            raise ReplayError(
                f"source corpus has {len(source_files)} text files, expected "
                f"{provenance['observed_txt_files']}"
            )
        for selected in provenance.get("selection", []):
            candidate = source_root / str(selected.get("source_basename", ""))
            if sha256_file(candidate) != selected.get("source_sha256"):
                raise ReplayError(f"selected source identity changed: {candidate.name}")
        source_result = "verified"
    return {
        "manifest_sha256": sha256_file(path),
        "raw_render_sha256": raw["sha256"],
        "protocol_cases": len(cases),
        "source": source_result,
    }


def expanded_fixture(manifest: dict[str, Any], suffix: str) -> dict[str, Any]:
    measurement = manifest["measurement_fixture"]
    selected = next(
        case for case in manifest["protocol_cases"] if case["name"] == measurement["case"]
    )
    # Grow the stable side of the structural boundary.  Appending the corpus after the marker
    # would benchmark a private continuation whose reusable frontier happens to resemble the
    # intended shared prefix, while leaving the actual SSD-eligible checkpoint tiny.
    marker = "=== CACHE_BREAKPOINT ==="
    marker_offset = selected["stable"].find(marker)
    if marker_offset < 0:
        raise ReplayError("measurement fixture has no structural cache boundary")
    repeated = measurement["stable_unit"] * int(measurement["stable_repetitions"])
    stable = (
        selected["stable"][:marker_offset]
        + repeated
        + "\n"
        + selected["stable"][marker_offset:]
    )
    return {
        "name": selected["name"],
        "protocol": selected["protocol"],
        "measurement": True,
        "stable": stable,
        "volatile": selected["volatile"],
        "suffix": suffix,
        "payload_sha256": hashlib.sha256(
            (stable + "\n" + selected["volatile"] + "\n" + suffix).encode("utf-8")
        ).hexdigest(),
    }


def boundary_fixtures(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in manifest["protocol_cases"]:
        payload = case["stable"] + "\n" + case["volatile"] + "\nBoundary replay."
        result.append(
            {
                "name": case["name"],
                "protocol": case["protocol"],
                "measurement": False,
                "stable": case["stable"],
                "volatile": case["volatile"],
                "suffix": "Boundary replay.",
                "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
        )
    return result


def pressure_fixture(fixture: dict[str, Any], index: int) -> dict[str, Any]:
    pressured = dict(fixture)
    pressured["stable"] = f"Independent synthetic pressure lineage {index}."
    pressured["volatile"] = "No shared structural boundary."
    pressured["suffix"] = "Continue with distinct deterministic filler."
    pressured["protocol"] = "openai_chat"
    pressured["measurement"] = False
    pressured["cache_writes"] = False
    pressured["max_output_tokens"] = 512
    payload = pressured["stable"] + "\n" + pressured["volatile"] + "\n" + pressured["suffix"]
    pressured["payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return pressured


def median_mad(values: Sequence[float]) -> tuple[float, float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ReplayError("timing samples must be finite and nonnegative")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return median, mad


def load_predecessor_calibration(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read predecessor calibration {path}: {error}") from error
    if not isinstance(value, dict) or (
        value.get("artifact_type"), value.get("schema_version")
    ) != (CALIBRATION_TYPE, CALIBRATION_VERSION):
        raise ReplayError("unsupported predecessor calibration identity")
    provenance = value.get("provenance")
    target = value.get("target")
    snapshot = value.get("complete_private_host_snapshot")
    cost = value.get("h2d_cost_reference")
    if not all(isinstance(item, dict) for item in (provenance, target, snapshot, cost)):
        raise ReplayError("predecessor calibration is missing required sections")
    expected_target = {
        "compute_capability": "8.9",
        "model_id": "qwen3.8-27b",
        "weights_id": "groupwise-int",
        "kv_dtype": "rk4v4-e8",
        "mtp_draft_tokens": 3,
        "max_context": 128000,
        "max_concurrency": 4,
        "kv_capacity_mode": "auto",
    }
    mismatched = {
        name: (target.get(name), expected)
        for name, expected in expected_target.items()
        if target.get(name) != expected
    }
    if mismatched:
        raise ReplayError(f"predecessor calibration target differs: {mismatched}")
    try:
        components = [
            int(snapshot.get(name, -1))
            for name in ("state_bytes", "main_kv_bytes", "mtp_kv_bytes")
        ]
        payload_bytes = int(snapshot.get("physical_h2d_payload_bytes", -1))
        copy_operations = int(snapshot.get("copy_operations", -1))
    except (TypeError, ValueError) as error:
        raise ReplayError("predecessor calibration private Host payload is malformed") from error
    if any(component <= 0 for component in components) or sum(components) != payload_bytes:
        raise ReplayError("predecessor calibration private Host payload is incomplete")
    if copy_operations != len(components):
        raise ReplayError("predecessor calibration copy-operation count is inconsistent")
    try:
        batch_ns = int(cost["batch_ns"])
        operation_ns = int(cost["operation_ns"])
        ns_per_byte_q32 = int(cost["ns_per_byte_q32"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("predecessor calibration H2D cost reference is malformed") from error
    if min(batch_ns, operation_ns, ns_per_byte_q32) < 0:
        raise ReplayError("predecessor calibration H2D costs must be nonnegative")
    cost_source = cost.get("source_path")
    cost_source_sha256 = cost.get("source_sha256")
    if not isinstance(cost_source, str) or not isinstance(cost_source_sha256, str):
        raise ReplayError("predecessor calibration has no H2D cost-source identity")
    cost_source_path = Path(__file__).resolve().parents[2] / cost_source
    if not cost_source_path.is_file() or sha256_file(cost_source_path) != cost_source_sha256:
        raise ReplayError("predecessor calibration H2D cost-source identity changed")
    source_report = provenance.get("source_report")
    source_report_sha256 = provenance.get("source_report_sha256")
    if not isinstance(source_report, str) or not isinstance(source_report_sha256, str):
        raise ReplayError("predecessor calibration has no source-report identity")
    report_path = Path(__file__).resolve().parents[2] / source_report
    if not report_path.is_file() or sha256_file(report_path) != source_report_sha256:
        raise ReplayError("predecessor calibration source-report identity changed")
    bandwidth_ns = (ns_per_byte_q32 * payload_bytes + (1 << 32) - 1) >> 32
    operation_limited_ns = batch_ns + copy_operations * operation_ns
    reference_ns = max(bandwidth_ns, operation_limited_ns)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "source_commit": provenance.get("source_commit"),
        "source_report": source_report,
        "source_report_sha256": source_report_sha256,
        "target": target,
        "complete_private_host_snapshot": snapshot,
        "h2d_cost_reference": cost,
        "complete_private_host_h2d_reference_ms": reference_ns / 1.0e6,
        "threshold_role": value.get("threshold_role"),
    }


def freeze_material_improvement_threshold(
    baseline_groups: Iterable[Sequence[float]],
    predecessor_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Freeze a conservative reduction threshold before optimized trials are inspected.

    Three relative MADs covers ordinary run-to-run noise.  The independently recorded predecessor
    calibration supplies a dimensionally compatible minimum effect size: the net TTFT saving must
    exceed one estimated complete private Host H2D materialization.  That estimate is not promoted
    to a measured latency result.
    """

    groups = [list(group) for group in baseline_groups]
    if not groups:
        raise ReplayError("at least one baseline group is required")
    details: list[dict[str, float]] = []
    noise = 0.0
    for group in groups:
        median, mad = median_mad(group)
        relative_mad = mad / median if median > 0 else 0.0
        noise = max(noise, 3.0 * relative_mad)
        details.append({"median_ms": median, "mad_ms": mad, "relative_mad": relative_mad})
    reference_ms = float(predecessor_calibration["complete_private_host_h2d_reference_ms"])
    comparison_baseline_ms = details[-1]["median_ms"]
    if not math.isfinite(reference_ms) or reference_ms <= 0 or comparison_baseline_ms <= 0:
        raise ReplayError("predecessor calibration has no usable materiality reference")
    calibration_fraction = reference_ms / comparison_baseline_ms
    threshold = max(calibration_fraction, noise)
    return {
        "method": (
            "max(predecessor complete-private Host H2D effect size divided by the "
            "existing-cache median, three times the largest baseline relative MAD)"
        ),
        "predecessor_calibration": predecessor_calibration,
        "calibration_effect_size_ms": reference_ms,
        "calibration_fraction": calibration_fraction,
        "noise_fraction": noise,
        "required_reduction_fraction": threshold,
        "baseline_groups": details,
    }


def require_unchanged_frozen_threshold(
    recorded: dict[str, Any], recomputed: dict[str, Any]
) -> None:
    try:
        recorded_fraction = float(recorded["required_reduction_fraction"])
        recomputed_fraction = float(recomputed["required_reduction_fraction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError("frozen material-improvement threshold is malformed") from error
    if not math.isclose(
        recorded_fraction, recomputed_fraction, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ReplayError(
            "predecessor calibration changes the pre-optimized numeric threshold; rerun the "
            "campaign instead of applying post-hoc comparisons"
        )


def request_measurement(
    label: str,
    external_ttft_ms: float,
    event: dict[str, Any],
    *,
    payload_sha256: str,
) -> dict[str, Any]:
    if event.get("event") != "request_done":
        raise ReplayError(f"{label} has no request_done event")
    result = event.get("result", {})
    timing = event.get("timings_seconds", {})
    engine = event.get("engine_timing", {})
    speculative = event.get("speculative", {})
    durable = result.get("durable_restore", {})
    materialization = event.get("materialization", {})
    request = event.get("request", {})
    try:
        prompt_tokens = int(result["prompt_tokens"])
        reused_tokens = int(result["prefix_cache_hit_tokens"])
        completion_tokens = int(result["completion_tokens"])
        total_seconds = float(timing["total"])
        server_ttft_ms = float(timing["ttft"]) * 1000.0
        queue_delay_ms = float(engine["queue_wait_seconds"]) * 1000.0
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError(f"{label} request log is missing required metrics: {error}") from error
    generated_token_ids = result.get("generated_token_ids")
    if generated_token_ids is not None and (
        not isinstance(generated_token_ids, list)
        or any(not isinstance(token, int) for token in generated_token_ids)
    ):
        raise ReplayError(f"{label} generated token IDs are malformed")
    if generated_token_ids is not None and len(generated_token_ids) != completion_tokens:
        raise ReplayError(
            f"{label} logged {len(generated_token_ids)} generated token IDs for "
            f"{completion_tokens} completion tokens"
        )
    speculative_values = {
        "rounds": speculative.get("rounds"),
        "drafted_tokens": speculative.get("drafted_tokens"),
        "accepted_tokens": speculative.get("accepted_tokens"),
        "fallback_steps": speculative.get("fallback_steps"),
    }
    if any(not isinstance(value, int) or value < 0 for value in speculative_values.values()):
        raise ReplayError(f"{label} speculative counters are missing or malformed")
    draft_window = speculative.get("draft_window")
    accepted_per_position = speculative.get("accepted_per_position")
    if (
        not isinstance(draft_window, int)
        or draft_window < 0
        or not isinstance(accepted_per_position, list)
        or any(not isinstance(value, int) or value < 0 for value in accepted_per_position)
        or len(accepted_per_position) != draft_window
        or sum(accepted_per_position) != speculative_values["accepted_tokens"]
    ):
        raise ReplayError(f"{label} per-position speculative counters are missing or malformed")
    return {
        "label": label,
        "request_id": request.get("request_id"),
        "response_id": request.get("response_id"),
        "payload_sha256": payload_sha256,
        "external_ttft_ms": external_ttft_ms,
        "server_ttft_ms": server_ttft_ms,
        "queue_delay_ms": queue_delay_ms,
        "total_ms": total_seconds * 1000.0,
        "prompt_tokens": prompt_tokens,
        "evaluated_tokens": max(prompt_tokens - reused_tokens, 0),
        "reused_tokens": reused_tokens,
        "completion_tokens": completion_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_token_ids_sha256": (
            hashlib.sha256(
                b"".join(
                    int(token).to_bytes(4, "little", signed=True)
                    for token in generated_token_ids
                )
            ).hexdigest()
            if generated_token_ids is not None
            else None
        ),
        "selected_tier": (
            "ssd"
            if durable.get("ssd_loaded")
            else "memory" if reused_tokens else "root"
        ),
        "selected_frontier": int(durable.get("frontier_tokens", 0) or reused_tokens),
        "prefix_reuse_path": result.get("prefix_reuse_path"),
        "durable_restore": durable,
        "materialization": materialization,
        "speculative": {
            "backend": speculative.get("backend"),
            "draft_window": draft_window,
            **speculative_values,
            "accepted_per_position": accepted_per_position,
        },
    }


def compare_profile(
    baseline: Sequence[dict[str, Any]],
    subject: Sequence[dict[str, Any]],
    threshold: dict[str, Any],
) -> dict[str, Any]:
    baseline_median, baseline_mad = median_mad(
        [float(row["external_ttft_ms"]) for row in baseline]
    )
    subject_median, subject_mad = median_mad(
        [float(row["external_ttft_ms"]) for row in subject]
    )
    reduction = (baseline_median - subject_median) / baseline_median
    required = float(threshold["required_reduction_fraction"])
    return {
        "baseline_median_ttft_ms": baseline_median,
        "baseline_mad_ttft_ms": baseline_mad,
        "subject_median_ttft_ms": subject_median,
        "subject_mad_ttft_ms": subject_mad,
        "reduction_fraction": reduction,
        "required_reduction_fraction": required,
        "material_improvement": reduction >= required,
        "trials": min(len(baseline), len(subject)),
    }
