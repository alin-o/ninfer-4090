from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest

from tools.bench.run_serve_corpus import (
    CampaignError,
    Fixture,
    RunSpec,
    build_result_record,
    require_server_log_identity,
    summary_row,
)
from tools.bench.run_tiered_cache_replay import (
    RunningServe,
    boundary_replay_status,
    build_regression_matrix,
    build_supplemental_evidence,
    campaign_exit_status,
    campaign_verdict,
    configuration_decision,
    current_build_control_identity,
    done_by_response_id,
    load_baseline_identity,
    load_validation_cases,
    require_profile_evidence,
    summarize_profile,
    validate_measured_continuations,
)
from tools.bench.tiered_cache_replay import (
    ReplayError,
    boundary_fixtures,
    compare_profile,
    expanded_fixture,
    freeze_material_improvement_threshold,
    load_predecessor_calibration,
    pressure_fixture,
    request_measurement,
    require_unchanged_frozen_threshold,
    sha256_file,
    validate_manifest,
)


TIERED_FIXTURES = Path("bench/fixtures/tiered_cache/manifest.json")
TIERED_PREDECESSOR_CALIBRATION = Path(
    "bench/fixtures/tiered_cache/predecessor-calibration.json"
)


def test_request_log_v20_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 20,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=19)
    with pytest.raises(CampaignError):
        require_server_log_identity(stale, "server_start")


def test_result_record_parses_request_host_exposure() -> None:
    fixture = Fixture(
        name="fixture",
        messages=[],
        thinking=True,
        max_new=8,
        suite="test",
    )
    spec = RunSpec(
        target="qwen3_6_27b",
        model_id="qwen3.6-27b",
        artifact=Path("/tmp/model.ninfer"),
        speculative_mode="mtp3",
        speculative_backend="mtp",
        draft_tokens=3,
        sampling_mode="greedy",
        fixture=fixture,
        seed=7,
    )
    payload = {"model": spec.model_id}
    response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    event = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 20,
        "event": "request_done",
        "request": {
            "model": spec.model_id,
            "requested_output_tokens": 8,
            "enable_thinking": True,
            "sampling": {"seed": 7},
        },
        "result": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "finish_reason": "output_limit",
        },
        "timings_seconds": {
            "prepare": 0.1,
            "vision": 0.0,
            "prefill": 0.2,
            "decode": 0.4,
            "total": 0.7,
        },
        "speculative": {
            "backend": "mtp",
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 3,
            "fallback_steps": 0,
        },
        "engine_timing": {
            "queue_wait_seconds": 0.001,
            "host_exposed_seconds": {
                "engine_boundary": 0.001,
                "program_submit": 0.002,
                "program_post": 0.003,
                "engine_commit_output": 0.004,
                "engine_maintenance": 0.005,
                "total": 0.015,
            },
            "device_wait_exposed_seconds": 0.3,
            "decode": {
                "host_exposed_seconds": 0.01,
                "device_wait_exposed_seconds": 0.2,
                "rounds": 2,
            },
        },
    }

    record = build_result_record(spec, "groupwise-int", payload, response, event)
    assert record["schema_version"] == 6
    assert record["metrics"]["engine_host_exposed_ms"] == pytest.approx(15.0)
    assert record["metrics"]["decode_host_us_per_round"] == pytest.approx(5000.0)
    assert record["metrics"]["decode_device_wait_us_per_round"] == pytest.approx(100000.0)


def test_summary_retains_one_canonical_weights_id() -> None:
    records = [{"weights_id": "nvfp4", "metrics": {}}]
    row = summary_row(
        "context_profile",
        "qwen3_6_27b",
        "fixture",
        "fixture",
        "mtp0",
        "greedy",
        records,
    )
    assert row["weights_id"] == "nvfp4"

    with pytest.raises(CampaignError):
        summary_row(
            "context_profile",
            "qwen3_6_27b",
            "fixture",
            "fixture",
            "mtp0",
            "greedy",
            [*records, {"weights_id": "groupwise-int", "metrics": {}}],
        )


def test_tiered_cache_fixture_provenance_and_protocol_split() -> None:
    checked = validate_manifest(TIERED_FIXTURES)
    assert checked["protocol_cases"] == 5
    assert checked["source"] == "not-requested"

    import json

    manifest = json.loads(TIERED_FIXTURES.read_text(encoding="utf-8"))
    fixture = expanded_fixture(manifest, "branch")
    assert fixture["protocol"] == "anthropic_messages"
    assert fixture["measurement"] is True
    assert fixture["stable"].count(manifest["measurement_fixture"]["stable_unit"]) == 384
    marker = fixture["stable"].index("=== CACHE_BREAKPOINT ===")
    assert fixture["stable"].index(manifest["measurement_fixture"]["stable_unit"]) < marker
    boundaries = boundary_fixtures(manifest)
    assert {item["protocol"] for item in boundaries} == {
        "openai_chat",
        "openai_responses",
        "anthropic_messages",
    }
    assert len({item["payload_sha256"] for item in boundaries}) == len(boundaries)
    pressured = [pressure_fixture(fixture, index) for index in range(4)]
    assert len({item["payload_sha256"] for item in pressured}) == 4
    assert all(item["cache_writes"] is False for item in pressured)
    assert all(item["max_output_tokens"] == 512 for item in pressured)


def test_tiered_cache_threshold_consumes_predecessor_calibration_and_baseline_noise() -> None:
    calibration = load_predecessor_calibration(TIERED_PREDECESSOR_CALIBRATION)
    assert calibration["source_commit"] == "69ef567748cc6b79604caeef64e4bb22309afdab"
    assert calibration["complete_private_host_h2d_reference_ms"] == pytest.approx(7.809127)

    threshold = freeze_material_improvement_threshold(
        [[100.0, 101.0, 99.0], [80.0, 82.0, 78.0]], calibration
    )
    assert threshold["calibration_fraction"] == pytest.approx(7.809127 / 80.0)
    assert threshold["required_reduction_fraction"] == pytest.approx(7.809127 / 80.0)
    assert threshold["predecessor_calibration"]["sha256"] == calibration["sha256"]

    noisy = freeze_material_improvement_threshold([[100.0, 120.0, 80.0]], calibration)
    assert noisy["required_reduction_fraction"] == pytest.approx(0.60)

    require_unchanged_frozen_threshold(threshold, dict(threshold))
    with pytest.raises(ReplayError, match="post-hoc comparisons"):
        require_unchanged_frozen_threshold(
            threshold, dict(threshold, required_reduction_fraction=0.5)
        )


def test_tiered_cache_measurement_preserves_tier_and_speculative_counters() -> None:
    event = {
        "event": "request_done",
        "result": {
            "prompt_tokens": 100,
            "completion_tokens": 4,
            "generated_token_ids": [11, 12, 13, 14],
            "prefix_cache_hit_tokens": 80,
            "prefix_reuse_path": "shared_stable_prefix",
            "durable_restore": {
                "frontier_tokens": 80,
                "ssd_loaded": True,
                "warm_available": False,
                "fallback_reason": "",
            },
        },
        "timings_seconds": {"ttft": 0.2, "total": 0.3},
        "engine_timing": {"queue_wait_seconds": 0.01},
        "materialization": {"reclaimed_device_main_kv_pages": 2},
        "speculative": {
            "backend": "mtp",
            "draft_window": 3,
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 0,
            "fallback_steps": 2,
            "accepted_per_position": [0, 0, 0],
        },
    }
    row = request_measurement("ssd", 210.0, event, payload_sha256="a" * 64)
    assert row["selected_tier"] == "ssd"
    assert row["evaluated_tokens"] == 20
    assert row["generated_token_ids"] == [11, 12, 13, 14]
    assert row["speculative"]["accepted_tokens"] == 0

    malformed = dict(event, result=dict(event["result"], completion_tokens=3))
    with pytest.raises(ReplayError, match="generated token IDs"):
        request_measurement("bad", 210.0, malformed, payload_sha256="b" * 64)

    comparison = compare_profile(
        [dict(row, external_ttft_ms=value) for value in (200.0, 202.0, 198.0)],
        [dict(row, external_ttft_ms=value) for value in (150.0, 151.0, 149.0)],
        freeze_material_improvement_threshold(
            [[200.0, 202.0, 198.0]],
            load_predecessor_calibration(TIERED_PREDECESSOR_CALIBRATION),
        ),
    )
    assert comparison["material_improvement"] is True


def test_tiered_cache_server_start_failure_reaps_owned_process(tmp_path: Path) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    running = RunningServe(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path / "startup.log",
        port,
        0.05,
    )
    with pytest.raises(RuntimeError, match="timed out"):
        running.__enter__()
    assert running.process is not None
    assert running.process.poll() is not None
    assert running.output is None


def test_tiered_cache_server_early_exit_closes_log(tmp_path: Path) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    running = RunningServe(
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path / "startup-exit.log",
        port,
        1.0,
    )
    with pytest.raises(RuntimeError, match="exited during startup"):
        running.__enter__()
    assert running.process is not None
    assert running.process.returncode == 7
    assert running.output is None


def test_tiered_cache_baseline_identity_pins_revision_and_executable(tmp_path: Path) -> None:
    serve = tmp_path / "ninfer-serve"
    serve.write_bytes(b"recorded baseline executable")
    identity = {
        "artifact_type": "ninfer_tiered_cache_baseline_build",
        "schema_version": 1,
        "source_revision": "c7dca9acfef663acd10d4ec2817f184bc50547fe",
        "source_archive_sha256": "a" * 64,
        "serve_sha256": sha256_file(serve),
    }
    manifest = tmp_path / "baseline-build.json"
    manifest.write_text(json.dumps(identity), encoding="utf-8")
    assert load_baseline_identity(manifest, serve)["serve_sha256"] == sha256_file(serve)

    manifest.write_text(
        json.dumps(dict(identity, source_revision="current-worktree")), encoding="utf-8"
    )
    with pytest.raises(ReplayError, match="baseline build revision"):
        load_baseline_identity(manifest, serve)


def test_tiered_cache_validation_pass_requires_executable_identity(tmp_path: Path) -> None:
    evidence = {
        "artifact_type": "ninfer_tiered_cache_validation_evidence",
        "schema_version": 1,
        "serve_sha256": "b" * 64,
        "cases": {"pressure-resume": {"status": "PASS"}},
    }
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ReplayError, match="test executable SHA-256"):
        load_validation_cases(path, "b" * 64)


def test_tiered_cache_concurrent_events_join_by_response_identity(tmp_path: Path) -> None:
    path = tmp_path / "requests.jsonl"
    events = [
        {
            "schema_version": 20,
            "event": "request_done",
            "request": {"request_id": request_id, "response_id": response_id},
        }
        for request_id, response_id in ((2, "response-b"), (1, "response-a"))
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    joined = done_by_response_id(
        path, 0, ["response-a", "response-b"], 0.1, allowed_schemas=(20,)
    )
    assert joined["response-a"]["request"]["request_id"] == 1
    assert joined["response-b"]["request"]["request_id"] == 2


def test_tiered_cache_summary_sums_interval_deltas_and_peaks_gauges() -> None:
    row = {
        "queue_delay_ms": 1.0,
        "external_ttft_ms": 2.0,
        "total_ms": 3.0,
        "reused_tokens": 4,
        "evaluated_tokens": 5,
        "selected_tier": "host",
        "speculative": {"drafted_tokens": 6, "accepted_tokens": 3},
    }
    intervals = []
    for count, occupied in ((2, 7), (3, 5)):
        transfer = {
            direction: {"count": 0, "pages": 0, "bytes": 0, "seconds": 0.0}
            for direction in ("d2d", "d2h", "h2d")
        }
        transfer["h2d"] = {
            "count": count,
            "pages": count,
            "bytes": count * 100,
            "seconds": count / 10,
        }
        intervals.append(
            {
                "interval_seconds": 1.0,
                "tokens": {"computed_prefill": 1, "committed_decode": 2},
                "context_cache": {
                    "actual_transfer_seconds": count / 10,
                    "state_transfers": transfer,
                    "main_kv_transfers": transfer,
                    "backend_kv_transfers": transfer,
                    "occupancy": {"device_state_slots": occupied},
                    "pressure": {
                        "spill_pages": count,
                        "private_owners_evicted": count,
                    },
                },
            }
        )
    final_metrics = {
        'ninfer:context_cache_reclaimed_capacity_total{tier="device",resource="main_kv_pages"}': 4
    }
    summary = summarize_profile(
        [row], {"throughput_events": intervals, "final_metrics": final_metrics}
    )
    assert summary["transfer"]["actual_seconds"] == pytest.approx(0.5)
    assert summary["transfer"]["state_transfers"]["h2d"]["count"] == 5
    assert summary["transfer"]["state_transfers"]["h2d"]["bytes"] == 500
    assert summary["transfer"]["state_transfers"]["h2d"]["seconds"] == pytest.approx(0.5)
    assert summary["transfer"]["main_kv_transfers"]["h2d"]["pages"] == 5
    assert summary["transfer"]["backend_kv_transfers"]["h2d"]["pages"] == 5
    assert summary["pressure"]["spill_pages"] == 5
    assert summary["pressure"]["private_owners_evicted"] == 5
    assert summary["peak_occupancy"]["device_state_slots"] == 7
    assert summary["reclaimed_capacity"]["device_main_kv_pages"] == 4


def test_tiered_cache_profile_evidence_sums_host_restore_intervals() -> None:
    measurements = {
        "device": [{"reused_tokens": 10, "selected_tier": "memory"}],
        "host": [{"reused_tokens": 10, "selected_tier": "memory"}],
        "ssd": [{"reused_tokens": 10, "selected_tier": "ssd"}],
        "overlap": [{"prefix_reuse_path": "shared_stable_prefix"}],
    }
    profiles = {
        "host-trial-0": {
            "throughput_events": [
                {
                    "context_cache": {
                        "state_transfers": {"h2d": {"count": count}}
                    }
                }
                for count in (0, 1)
            ],
            "final_metrics": {},
        },
        "ssd-seed": {
            "final_metrics": {
                'ninfer:shared_ssd_writes_total{result="completed"}': 1,
                "ninfer:shared_ssd_manifest_records": 1,
            }
        },
    }
    require_profile_evidence(measurements, profiles)  # type: ignore[arg-type]
    assert measurements["host"][0]["selected_tier"] == "host"


def test_tiered_cache_verdicts_are_derived_and_missing_validation_is_unverified() -> None:
    def measured(label: str, profile: str, drafted: int) -> dict[str, object]:
        return {
            "label": label,
            "payload_sha256": "f" * 64,
            "generated_token_ids": [11, 12],
            "generated_token_ids_sha256": "a" * 64,
            "speculative": {
                "draft_window": 3,
                "rounds": 2,
                "drafted_tokens": drafted,
                "accepted_tokens": 1,
                "fallback_steps": 0,
                "accepted_per_position": [1, 0, 0],
            },
            "selected_tier": profile,
        }

    measurements = {
        "cold": [measured("cold-0", "root", 4)],
        "device": [measured("device-0", "device", 3)],
        "host": [measured("host-0", "host", 4)],
        "ssd": [measured("ssd-0", "ssd", 4)],
    }
    continuation = validate_measured_continuations(measurements)  # type: ignore[arg-type]
    assert continuation["status"] == "FAIL"
    assert continuation["target_token_status"] == "PASS"
    assert continuation["speculative_counter_status"] == "FAIL"
    assert len(continuation["speculative_counter_mismatches"]) == 1
    matrix = build_regression_matrix(continuation, {})
    assert matrix[0]["status"] == "UNVERIFIED"
    assert matrix[2]["status"] == "PASS"
    assert matrix[3]["status"] == "FAIL"
    assert all(row["status"] == "UNVERIFIED" for row in matrix[4:])
    supplemental = build_supplemental_evidence(
        {"official-tokenizer-lineage": {"status": "PASS"}}
    )
    assert supplemental == [
        {
            "status": "UNVERIFIED",
            "case": "Qwen3.8 artifact-backed Frontend lineage for boundary fixtures",
            "evidence": (
                "target artifact frontend evidence is absent, skipped, or not identity-pinned"
            ),
        },
        {
            "status": "WAIVED",
            "case": "separate Qwen3.6 tokenizer triplet",
            "evidence": (
                "waived by Alin Olteanu on 2026-09-11; the authorized Qwen3.8 artifact-backed "
                "Frontend is the target validation path"
            ),
        },
    ]

    decision = configuration_decision(
        {
            "device": {"material_improvement": True, "reduction_fraction": 0.2},
            "host": {"material_improvement": False, "reduction_fraction": 0.05},
            "ssd": {"material_improvement": False, "reduction_fraction": -0.1},
        }
    )
    assert decision["status"] == "FAIL"
    assert decision["missed_profiles"] == ["host", "ssd"]
    assert decision["slower_than_existing_baseline"] == ["ssd"]
    assert decision["tier_actions"] == {
        "device": "retain",
        "host": "material-improvement-unverified",
        "ssd": "disable",
    }
    assert "Every measured tier" not in decision["reason"]
    blocked = configuration_decision(
        {"device": {"material_improvement": True, "reduction_fraction": 0.2}},
        "FAIL",
    )
    assert blocked["status"] == "FAIL"
    assert blocked["tier_actions"] == {"device": "blocked-by-correctness"}
    assert "no tier or cost-preset/config decision is justified" in blocked["reason"]
    verdict = campaign_verdict(
        {
            "device": {"material_improvement": True},
            "host": {"material_improvement": False},
        },
        matrix,
    )
    assert verdict == {
        "overall": "FAIL",
        "performance": "FAIL",
        "performance_measurement": "FAIL",
        "correctness": "FAIL",
        "reason": (
            "Target production acceptance is not established; inspect required failed and "
            "unverified rows and the measured comparisons."
        ),
    }


def test_supplemental_tokenizer_does_not_gate_target_verdicts() -> None:
    continuation = {
        "target_token_status": "PASS",
        "comparisons": [],
        "target_token_mismatches": [],
        "speculative_counter_status": "PASS",
        "speculative_counter_mismatches": [],
    }
    required_cases = {
        name: {"status": "PASS", "evidence": f"{name}.log"}
        for name in (
            "frontend-boundary-token-lineage",
            "openai-chat-boundary",
            "openai-responses-boundary",
            "anthropic-boundary",
            "pressure-resume",
            "host-restore",
            "shared-snapshot",
            "four-request-root-fallback",
            "delayed-spill",
            "delayed-active-capture",
            "cuda-transfer-failure",
            "pending-snapshot-shutdown",
            "resource-manager",
            "durable-shared-prefix-catalog",
        )
    }
    comparisons = {
        profile: {"material_improvement": True, "reduction_fraction": 0.5}
        for profile in ("device", "host", "ssd")
    }
    boundary = {
        "status": "PASS",
        "case": "live serving boundary protocols",
        "evidence": "5/5",
    }

    required = build_regression_matrix(continuation, required_cases, boundary)
    supplemental = build_supplemental_evidence(required_cases)
    verdict = campaign_verdict(comparisons, required)
    decision = configuration_decision(comparisons, verdict["correctness"])

    assert all(row["status"] == "PASS" for row in required)
    assert [row["status"] for row in supplemental] == ["UNVERIFIED", "WAIVED"]
    assert verdict["correctness"] == "PASS"
    assert verdict["performance"] == "PASS"
    assert verdict["performance_measurement"] == "PASS"
    assert verdict["overall"] == "PASS"
    assert decision["status"] == "PASS"
    assert campaign_exit_status(verdict) == 0

    required_unverified = build_regression_matrix(continuation, {}, boundary)
    unverified = campaign_verdict(comparisons, required_unverified)
    assert unverified["correctness"] == "UNVERIFIED"
    assert unverified["performance"] == "UNVERIFIED"
    assert unverified["performance_measurement"] == "PASS"
    assert unverified["overall"] == "UNVERIFIED"
    assert campaign_exit_status(unverified) == 3

    required_cases["pressure-resume"]["status"] = "FAIL"
    required_failed = build_regression_matrix(continuation, required_cases, boundary)
    failed = campaign_verdict(comparisons, required_failed)
    assert failed["correctness"] == "FAIL"
    assert failed["performance"] == "FAIL"
    assert failed["performance_measurement"] == "PASS"
    assert failed["overall"] == "FAIL"
    assert campaign_exit_status(failed) == 3


def test_current_build_control_is_explicitly_configuration_only(tmp_path) -> None:
    serve = tmp_path / "ninfer-serve"
    serve.write_bytes(b"current executable")

    identity = current_build_control_identity(serve)

    assert identity["kind"] == "current-build-configuration-control"
    assert identity["same_executable_as_optimized_profiles"] is True
    assert identity["serve_sha256"] == sha256_file(serve)
    assert identity["claim_scope"] == (
        "configuration comparison within one current build; no historical speedup claim"
    )


def test_artifact_backed_frontend_lineage_uses_target_artifact_without_triplet(tmp_path) -> None:
    artifact = tmp_path / "qwen3_8_27b.ninfer"
    artifact.write_bytes(b"identity-pinned artifact fixture")
    supplemental = build_supplemental_evidence(
        {
            "official-tokenizer-lineage": {
                "status": "PASS",
                "artifact_frontend": {
                    "authorization": "user-authorized-target-artifact-2026-09-11",
                    "path": str(artifact),
                    "bytes": artifact.stat().st_size,
                    "model_id": "qwen3.8-27b",
                    "weights_id": "groupwise-int",
                    "resources": {
                        "frontend/tokenizer.json": {"sha256": "a" * 64},
                    },
                },
            }
        }
    )

    assert [row["status"] for row in supplemental] == ["PASS", "WAIVED"]
    assert "artifact-backed Frontend" in supplemental[0]["evidence"]


def test_tiered_cache_boundary_verdict_requires_unique_preserved_identities() -> None:
    rows = [{"response_id": "a"}, {"response_id": "b"}]
    assert boundary_replay_status(rows, 2)["status"] == "PASS"
    assert boundary_replay_status([rows[0], rows[0]], 2)["status"] == "FAIL"


def test_tiered_cache_investigates_observed_mtp_counter_signature() -> None:
    def row(
        label: str,
        drafted: int,
        accepted: int,
        fallback: int,
        accepted_per_position: list[int],
    ) -> dict[str, object]:
        return {
            "label": label,
            "payload_sha256": "9" * 64,
            "generated_token_ids": list(range(16)),
            "generated_token_ids_sha256": "1" * 64,
            "speculative": {
                "draft_window": 3,
                "rounds": 5,
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "fallback_steps": fallback,
                "accepted_per_position": accepted_per_position,
            },
        }

    cold = row("cold-1", 15, 9, 1, [4, 3, 2])
    cached = row("cached-1", 14, 10, 0, [5, 4, 1])
    validation = validate_measured_continuations(
        {
            "cold": [cold],
            "device": [dict(cached, label="device-1")],
            "host": [dict(cached, label="host-1")],
            "ssd": [dict(cached, label="ssd-restart-1")],
        }  # type: ignore[arg-type]
    )
    assert validation["target_token_status"] == "PASS"
    assert validation["speculative_counter_status"] == "FAIL"
    assert len(validation["investigation"]) == 3
    assert validation["investigation"][0]["finding"].endswith(
        "draft acceptance versus target fallback accounting difference"
    )
    assert "accepted-per-position deltas [1, 1, -1]" in validation["investigation"][0][
        "finding"
    ]
    assert validation["investigation"][0]["root_cause"] == (
        "not established by the serving replay"
    )


def test_tiered_cache_records_first_exact_token_mismatch() -> None:
    def row(label: str, token_ids: list[int]) -> dict[str, object]:
        return {
            "label": label,
            "payload_sha256": "8" * 64,
            "generated_token_ids": token_ids,
            "generated_token_ids_sha256": label,
            "speculative": {
                "draft_window": 1,
                "rounds": 2,
                "drafted_tokens": 2,
                "accepted_tokens": 1,
                "fallback_steps": 0,
                "accepted_per_position": [1],
            },
        }

    validation = validate_measured_continuations(
        {
            "cold": [row("cold", [10, 20, 30])],
            "device": [row("device", [10, 21, 30])],
            "host": [row("host", [10, 21, 30])],
            "ssd": [row("ssd", [10, 21, 30])],
        }  # type: ignore[arg-type]
    )
    assert validation["target_token_status"] == "FAIL"
    assert validation["cached_tier_consensus"] == "PASS"
    assert validation["target_token_investigation"][0] == {
        "fixture_sha256": "8" * 64,
        "cold_label": "cold",
        "subject_label": "device",
        "index": 1,
        "cold_token_id": 20,
        "subject_token_id": 21,
        "cold_length": 3,
        "subject_length": 3,
        "root_cause": "not established by the serving replay",
        "verdict": "FAIL",
    }
