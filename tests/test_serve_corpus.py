from __future__ import annotations

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
from tools.bench.tiered_cache_replay import (
    boundary_fixtures,
    compare_profile,
    expanded_fixture,
    freeze_material_improvement_threshold,
    pressure_fixture,
    request_measurement,
    validate_manifest,
)


TIERED_FIXTURES = Path("bench/fixtures/tiered_cache/manifest.json")


def test_request_log_v19_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 19,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=18)
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
        "schema_version": 19,
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


def test_tiered_cache_threshold_is_frozen_from_baselines_only() -> None:
    threshold = freeze_material_improvement_threshold([[100.0, 101.0, 99.0], [80.0, 82.0, 78.0]])
    assert threshold["required_reduction_fraction"] == pytest.approx(0.10)
    noisy = freeze_material_improvement_threshold([[100.0, 120.0, 80.0]])
    assert noisy["required_reduction_fraction"] == pytest.approx(0.60)


def test_tiered_cache_measurement_preserves_tier_and_speculative_counters() -> None:
    event = {
        "event": "request_done",
        "result": {
            "prompt_tokens": 100,
            "completion_tokens": 4,
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
            "rounds": 2,
            "drafted_tokens": 6,
            "accepted_tokens": 0,
            "fallback_steps": 2,
        },
    }
    row = request_measurement("ssd", 210.0, event, payload_sha256="a" * 64)
    assert row["selected_tier"] == "ssd"
    assert row["evaluated_tokens"] == 20
    assert row["speculative"]["accepted_tokens"] == 0

    comparison = compare_profile(
        [dict(row, external_ttft_ms=value) for value in (200.0, 202.0, 198.0)],
        [dict(row, external_ttft_ms=value) for value in (150.0, 151.0, 149.0)],
        freeze_material_improvement_threshold([[200.0, 202.0, 198.0]]),
    )
    assert comparison["material_improvement"] is True
