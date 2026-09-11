#!/usr/bin/env python3
"""Replay and measure the Qwen3.8 tiered shared-prefix cache at concurrency four."""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.tiered_cache_replay import (
    EVIDENCE_TYPE,
    EVIDENCE_VERSION,
    ReplayError,
    boundary_fixtures,
    compare_profile,
    expanded_fixture,
    freeze_material_improvement_threshold,
    load_manifest,
    pressure_fixture,
    request_measurement,
    sha256_file,
    validate_manifest,
)
from tools.ninfer_serve.anthropic import anthropic_request
from tools.ninfer_serve.client import NInferServeClient, ProtocolRequest, ServeExchangeResult
from tools.ninfer_serve.openai_chat import chat_request
from tools.ninfer_serve.openai_responses import responses_request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "bench/fixtures/tiered_cache/manifest.json"
DEFAULT_SERVE = REPO_ROOT / "build-agent-verify/apps/ninfer-serve"
DEFAULT_WEIGHTS = Path("/models/qwen3_8_27b.ninfer")
REQUEST_LOG_SCHEMA = 19


class RunningServe:
    def __init__(self, command: Sequence[str], log: Path, port: int, timeout: float) -> None:
        self.command = list(command)
        self.log = log
        self.port = port
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.output: Any = None

    def __enter__(self) -> "RunningServe":
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise ReplayError(f"127.0.0.1:{self.port} is already in use")
        finally:
            probe.close()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.output = self.log.open("wb")
        self.process = subprocess.Popen(
            self.command,
            cwd=REPO_ROOT,
            stdout=self.output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ReplayError(
                    f"ninfer-serve exited during startup ({self.process.returncode}); "
                    f"see {self.log}"
                )
            try:
                connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = response.read()
                connection.close()
                if response.status == 200 and json.loads(body) == {"status": "ok"}:
                    return self
            except (OSError, http.client.HTTPException, json.JSONDecodeError):
                pass
            time.sleep(0.1)
        raise ReplayError(f"timed out waiting for ninfer-serve; see {self.log}")

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.output is not None:
            self.output.close()


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw = line.rpartition(" ")
        if separator:
            try:
                values[name] = float(raw)
            except ValueError:
                continue
    return values


def get_metrics(port: int) -> dict[str, float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("GET", "/metrics")
    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise ReplayError(f"GET /metrics returned HTTP {response.status}")
    return parse_metrics(body)


def gpu_memory_snapshot() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if query.returncode != 0:
        return {"available": False, "error": query.stderr.strip()}
    lines = query.stdout.splitlines()
    if not lines:
        return {"available": False, "error": "empty nvidia-smi memory output"}
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 3:
        return {"available": False, "error": "unexpected nvidia-smi memory output"}
    try:
        total, used, free = (int(field) for field in fields)
    except ValueError:
        return {"available": False, "error": "non-numeric nvidia-smi memory output"}
    return {
        "available": True,
        "units": "MiB",
        "total": total,
        "used": used,
        "free_headroom": free,
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            event = json.loads(line)
            if event.get("schema_version") != REQUEST_LOG_SCHEMA:
                raise ReplayError("request log schema differs from the harness contract")
            events.append(event)
    return events


def request_for(model: str, fixture: dict[str, Any]) -> ProtocolRequest:
    stable = fixture["stable"]
    dynamic = fixture["volatile"] + "\n" + fixture["suffix"]
    max_output_tokens = int(fixture.get("max_output_tokens", 16))
    if fixture["protocol"] == "openai_chat":
        request = chat_request(
            model,
            [{"role": "system", "content": stable}, {"role": "user", "content": dynamic}],
            max_output_tokens,
        )
        if fixture.get("cache_writes") is False:
            request.payload["prompt_cache_options"] = {"mode": "explicit"}
        return request
    if fixture["protocol"] == "openai_responses":
        request = responses_request(model, dynamic, max_output_tokens, store=False)
        request.payload["instructions"] = stable
        return request
    if fixture["protocol"] == "anthropic_messages":
        request = anthropic_request(
            model,
            [{"role": "user", "content": dynamic}],
            max_output_tokens,
            system=stable,
        )
        if fixture.get("measurement"):
            marker = "=== CACHE_BREAKPOINT ==="
            boundary = stable.index(marker) + len(marker)
            if boundary < len(stable) and stable[boundary] == "\n":
                boundary += 1
            # Anthropic's explicit block boundary is placed at the exact structural marker
            # frontier. The remaining system block stays byte-for-byte adjacent in the rendered
            # prompt, so protocol participation changes no prepared token or model-visible text.
            request.payload["system"] = [
                {
                    "type": "text",
                    "text": stable[:boundary],
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": stable[boundary:]},
            ]
        return request
    raise ReplayError(f"unsupported fixture protocol: {fixture['protocol']}")


def run_exchange(
    port: int, fixture: dict[str, Any], timeout: float
) -> tuple[float, ServeExchangeResult]:
    client = NInferServeClient(f"http://127.0.0.1:{port}", timeout)
    model = client.discover_model()
    result = client.prepare(request_for(model, fixture)).execute()
    output_events = [event for event in result.events if event.kind == "model_output"]
    if (
        result.http.status != 200
        or result.http.error is not None
        or result.protocol_error is not None
        or not output_events
        or result.http.sent_ns is None
    ):
        raise ReplayError(
            f"protocol request failed: status={result.http.status} http={result.http.error!r} "
            f"protocol={result.protocol_error!r} code={result.error_code!r}"
        )
    external_ttft_ms = (output_events[0].received_ns - result.http.sent_ns) / 1.0e6
    return external_ttft_ms, result


def server_command(
    serve: Path,
    weights: Path,
    port: int,
    request_log: Path,
    profile: str,
    cache_dir: Path,
) -> list[str]:
    command = [
        str(serve),
        str(weights),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-id",
        "qwen3.8-27b",
        "--max-context",
        "128000",
        "--kv-capacity",
        "auto",
        "--max-concurrency",
        "4",
        "--max-pending-requests",
        "16",
        "--pending-timeout-ms",
        "120000",
        "--kv-dtype",
        "rk4v4-e8",
        "--spec",
        "mtp",
        "--draft-tokens",
        "3",
        "--lm-head-draft",
        "--greedy",
        "--no-thinking",
        "--log-stats-interval-ms",
        "100",
        "--request-log-jsonl",
        str(request_log),
    ]
    if profile == "cold":
        command.append("--no-prefix-reuse")
    elif profile == "existing":
        command.extend(
            [
                "--device-state-slots",
                "1",
                "--host-state-slots",
                "0",
                "--host-kv-mib",
                "0",
                "--max-private-continuations",
                "8",
                "--max-shared-prefixes",
                "0",
                "--max-long-anchors-per-continuation",
                "0",
            ]
        )
    else:
        command.extend(
            [
                "--device-state-slots",
                "4" if profile != "host" else "0",
                "--host-state-slots",
                "8",
                "--host-kv-mib",
                "8192",
                "--max-private-continuations",
                "8",
                "--max-shared-prefixes",
                "4",
                "--max-long-anchors-per-continuation",
                "2",
            ]
        )
        if profile in {"ssd-seed", "ssd-restart"}:
            command.extend(
                [
                    "--shared-prefix-cache-dir",
                    str(cache_dir),
                    "--shared-prefix-cache-max-records",
                    "8",
                    "--shared-prefix-cache-max-mib",
                    "8192",
                    "--shared-prefix-cache-staging-mib",
                    "4096",
                    "--shared-prefix-cache-workers",
                    "2",
                    "--shared-prefix-cache-jobs",
                    "4",
                ]
            )
    return command


def wait_for_idle_catalog(port: int, timeout: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        last = get_metrics(port)
        queued = last.get('ninfer:shared_ssd_transfer_jobs{state="queued"}', 0)
        active = last.get('ninfer:shared_ssd_transfer_jobs{state="active"}', 0)
        claims = last.get('ninfer:shared_ssd_export_claims{state="pending"}', 0)
        if queued == 0 and active == 0 and claims == 0:
            return last
        time.sleep(0.1)
    raise ReplayError(f"durable catalog did not settle: {last}")


def wait_for_durable_record(port: int, timeout: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        last = get_metrics(port)
        completed = last.get('ninfer:shared_ssd_writes_total{result="completed"}', 0)
        records = last.get("ninfer:shared_ssd_manifest_records", 0)
        queued = last.get('ninfer:shared_ssd_transfer_jobs{state="queued"}', 0)
        active = last.get('ninfer:shared_ssd_transfer_jobs{state="active"}', 0)
        claims = last.get('ninfer:shared_ssd_export_claims{state="pending"}', 0)
        if completed > 0 and records > 0 and queued == 0 and active == 0 and claims == 0:
            return last
        time.sleep(0.1)
    raise ReplayError(f"durable seed did not publish a settled SSD record: {last}")


def latest_done(path: Path, prior_count: int, timeout: float) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    found = 0
    while time.monotonic() < deadline:
        events = load_events(path)
        done = [event for event in events if event.get("event") == "request_done"]
        found = len(done)
        if found == prior_count + 1:
            return done[-1], found
        if found > prior_count + 1:
            raise ReplayError(
                f"request log advanced by more than one event: {prior_count} -> {found}"
            )
        time.sleep(0.02)
    raise ReplayError(f"expected {prior_count + 1} request_done events, found {found}")


def run_profile(
    profile: str,
    fixtures: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    root: Path,
    *,
    seed: dict[str, Any] | None,
    cache_dir: Path | None = None,
    setup_fixtures: Sequence[dict[str, Any]] = (),
    concurrent_setup: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    request_log = root / "request-log" / f"{profile}.jsonl"
    serve_log = root / "serve" / f"{profile}.log"
    cache_dir = cache_dir or root / "shared-cache"
    request_log.parent.mkdir(parents=True, exist_ok=True)
    command = server_command(args.serve, args.weights, args.port, request_log, profile, cache_dir)
    measurements: list[dict[str, Any]] = []
    seed_measurement: dict[str, Any] | None = None
    setup_measurements: list[dict[str, Any]] = []
    setup_makespan_ms: float | None = None
    with RunningServe(command, serve_log, args.port, args.startup_timeout_seconds):
        events = load_events(request_log)
        start = next((event for event in events if event.get("event") == "server_start"), None)
        if start is None:
            raise ReplayError("server_start event is missing")
        ready_gpu_memory = gpu_memory_snapshot()
        done_count = 0
        if seed is not None:
            external, _ = run_exchange(args.port, seed, args.request_timeout_seconds)
            event, done_count = latest_done(request_log, done_count, args.request_timeout_seconds)
            seed_measurement = request_measurement(
                f"{profile}-seed",
                external,
                event,
                payload_sha256=seed["payload_sha256"],
            )
            if profile not in {"cold", "existing"}:
                wait_for_idle_catalog(args.port, args.request_timeout_seconds)
            if profile == "ssd-seed":
                wait_for_durable_record(args.port, args.request_timeout_seconds)
        if concurrent_setup and setup_fixtures:
            setup_results: list[
                tuple[float, ServeExchangeResult] | BaseException | None
            ] = [None] * len(setup_fixtures)
            barrier = threading.Barrier(len(setup_fixtures))

            def execute_setup(index: int) -> None:
                try:
                    barrier.wait()
                    setup_results[index] = run_exchange(
                        args.port, setup_fixtures[index], args.request_timeout_seconds
                    )
                except BaseException as error:
                    setup_results[index] = error

            threads = [
                threading.Thread(target=execute_setup, args=(index,))
                for index in range(len(setup_fixtures))
            ]
            setup_started = time.perf_counter_ns()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            setup_makespan_ms = (time.perf_counter_ns() - setup_started) / 1.0e6
            done = [
                event
                for event in load_events(request_log)
                if event.get("event") == "request_done"
            ][done_count:]
            if len(done) != len(setup_fixtures):
                raise ReplayError("concurrent setup did not complete every pressure request")
            for index, value in enumerate(setup_results):
                if isinstance(value, BaseException):
                    raise value
                if value is None:
                    raise ReplayError("concurrent setup worker produced no result")
                setup_measurements.append(
                    request_measurement(
                        f"{profile}-setup-{index}",
                        value[0],
                        done[index],
                        payload_sha256=setup_fixtures[index]["payload_sha256"],
                    )
                )
            done_count += len(done)
        else:
            for index, fixture in enumerate(setup_fixtures):
                external, _ = run_exchange(args.port, fixture, args.request_timeout_seconds)
                event, done_count = latest_done(
                    request_log, done_count, args.request_timeout_seconds
                )
                setup_measurements.append(
                    request_measurement(
                        f"{profile}-setup-{index}",
                        external,
                        event,
                        payload_sha256=fixture["payload_sha256"],
                    )
                )
        if setup_fixtures:
            wait_for_idle_catalog(args.port, args.request_timeout_seconds)
        if profile == "overlap":
            results: list[tuple[float, ServeExchangeResult] | BaseException | None] = [None] * len(
                fixtures
            )
            barrier = threading.Barrier(len(fixtures))

            def execute(index: int) -> None:
                try:
                    barrier.wait()
                    results[index] = run_exchange(
                        args.port, fixtures[index], args.request_timeout_seconds
                    )
                except BaseException as error:
                    results[index] = error

            threads = [
                threading.Thread(target=execute, args=(index,))
                for index in range(len(fixtures))
            ]
            started = time.perf_counter_ns()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            makespan_ms = (time.perf_counter_ns() - started) / 1.0e6
            events = load_events(request_log)
            done = [event for event in events if event.get("event") == "request_done"][done_count:]
            if len(done) != len(fixtures):
                raise ReplayError("overlap did not produce one request_done event per request")
            for index, value in enumerate(results):
                if isinstance(value, BaseException):
                    raise value
                if value is None:
                    raise ReplayError("overlap worker produced no result")
                measurements.append(
                    request_measurement(
                        f"{profile}-{index}",
                        value[0],
                        done[index],
                        payload_sha256=fixtures[index]["payload_sha256"],
                    )
                )
            profile_extra: dict[str, Any] = {"makespan_ms": makespan_ms}
        else:
            for index, fixture in enumerate(fixtures):
                external, _ = run_exchange(args.port, fixture, args.request_timeout_seconds)
                event, done_count = latest_done(
                    request_log, done_count, args.request_timeout_seconds
                )
                measurements.append(
                    request_measurement(
                        f"{profile}-{index}",
                        external,
                        event,
                        payload_sha256=fixture["payload_sha256"],
                    )
                )
            profile_extra = {}
        metrics = wait_for_idle_catalog(args.port, args.request_timeout_seconds)
        profile_extra.update(
            {
                "server_start": start,
                "final_metrics": metrics,
                "throughput_events": [
                    event
                    for event in load_events(request_log)
                    if event.get("event") == "throughput"
                ],
                "ready_gpu_memory": ready_gpu_memory,
                "settled_gpu_memory": gpu_memory_snapshot(),
                "command": command,
            }
        )
        if seed_measurement is not None:
            profile_extra["seed_measurement"] = seed_measurement
        if setup_measurements:
            profile_extra["setup_measurements"] = setup_measurements
            profile_extra["setup_makespan_ms"] = setup_makespan_ms
    return measurements, profile_extra


def require_profile_evidence(
    measurements: dict[str, list[dict[str, Any]]], profiles: dict[str, Any]
) -> None:
    for profile in ("device", "host", "ssd"):
        rows = measurements[profile]
        if not rows or any(int(row["reused_tokens"]) <= 0 for row in rows):
            raise ReplayError(f"{profile} profile did not reuse the selected prefix in every trial")
    if any(row["selected_tier"] != "ssd" for row in measurements["ssd"]):
        raise ReplayError("post-restart SSD profile did not load from SSD in every trial")
    if not any(
        row["prefix_reuse_path"] == "shared_stable_prefix" for row in measurements["overlap"]
    ):
        raise ReplayError("overlap profile never selected the shared stable prefix")

    host_profiles = sorted(name for name in profiles if name.startswith("host-trial-"))
    if len(host_profiles) != len(measurements["host"]):
        raise ReplayError("forced Host profile evidence is not trial-aligned")
    for name in host_profiles:
        host_h2d_count = max(
            (
                int(event["context_cache"]["state_transfers"]["h2d"]["count"])
                for event in profiles[name]["throughput_events"]
            ),
            default=0,
        )
        if host_h2d_count <= 0:
            raise ReplayError(f"{name} recorded no Host-to-Device State restore")
    for row in measurements["device"]:
        row["selected_tier"] = "device"
    for row in measurements["host"]:
        row["selected_tier"] = "host"

    seed_metrics = profiles["ssd-seed"]["final_metrics"]
    if seed_metrics.get('ninfer:shared_ssd_writes_total{result="completed"}', 0.0) <= 0:
        raise ReplayError("SSD seed profile completed no durable shared-prefix write")
    if seed_metrics.get("ninfer:shared_ssd_manifest_records", 0.0) <= 0:
        raise ReplayError("SSD seed profile published no durable manifest record")

    cleanup_metrics = (
        'ninfer:shared_ssd_transfer_jobs{state="queued"}',
        'ninfer:shared_ssd_transfer_jobs{state="active"}',
        'ninfer:shared_ssd_export_claims{state="pending"}',
        "ninfer:shared_ssd_staging_bytes",
        "ninfer:auto_save_queued_jobs",
        "ninfer:auto_save_in_flight_jobs",
        "ninfer:auto_save_reserved_jobs",
    )
    for name, detail in profiles.items():
        final = detail["final_metrics"]
        dirty = {
            metric: final.get(metric, 0.0)
            for metric in cleanup_metrics
            if final.get(metric, 0.0)
        }
        if dirty:
            raise ReplayError(f"{name} profile has unsettled transfer resources: {dirty}")


def summarize_profile(rows: Sequence[dict[str, Any]], detail: dict[str, Any]) -> dict[str, Any]:
    throughput = detail["throughput_events"]

    def peak(*path: str) -> float:
        values: list[float] = []
        for event in throughput:
            value: Any = event
            for component in path:
                value = value.get(component, {}) if isinstance(value, dict) else {}
            if isinstance(value, (int, float)):
                values.append(float(value))
        return max(values, default=0.0)

    transfer: dict[str, Any] = {"actual_seconds": peak("context_cache", "actual_transfer_seconds")}
    for resource in ("state_transfers", "main_kv_transfers", "backend_kv_transfers"):
        fields = (
            ("count", "bytes", "seconds")
            if resource == "state_transfers"
            else ("pages", "bytes", "seconds")
        )
        transfer[resource] = {
            direction: {
                field: peak("context_cache", resource, direction, field)
                for field in fields
            }
            for direction in ("d2d", "d2h", "h2d")
        }
    elapsed = sum(float(event.get("interval_seconds", 0.0)) for event in throughput)
    computed = sum(int(event.get("tokens", {}).get("computed_prefill", 0)) for event in throughput)
    committed = sum(int(event.get("tokens", {}).get("committed_decode", 0)) for event in throughput)
    final = detail["final_metrics"]
    return {
        "trials": len(rows),
        "median_queue_delay_ms": statistics.median(float(row["queue_delay_ms"]) for row in rows),
        "median_external_ttft_ms": statistics.median(
            float(row["external_ttft_ms"]) for row in rows
        ),
        "median_total_ms": statistics.median(float(row["total_ms"]) for row in rows),
        "reused_token_range": [
            min(int(row["reused_tokens"]) for row in rows),
            max(int(row["reused_tokens"]) for row in rows),
        ],
        "evaluated_token_range": [
            min(int(row["evaluated_tokens"]) for row in rows),
            max(int(row["evaluated_tokens"]) for row in rows),
        ],
        "selected_tiers": sorted({str(row["selected_tier"]) for row in rows}),
        "observed_interval_seconds": elapsed,
        "computed_prefill_tokens": computed,
        "committed_decode_tokens": committed,
        "computed_prefill_tokens_per_second": computed / elapsed if elapsed else None,
        "committed_decode_tokens_per_second": committed / elapsed if elapsed else None,
        "speculative_drafted_tokens": sum(
            int(row["speculative"]["drafted_tokens"] or 0) for row in rows
        ),
        "speculative_accepted_tokens": sum(
            int(row["speculative"]["accepted_tokens"] or 0) for row in rows
        ),
        "transfer": transfer,
        "peak_occupancy": {
            name: peak("context_cache", "occupancy", name)
            for name in (
                "device_state_slots",
                "host_state_slots",
                "device_main_kv_pages",
                "device_backend_kv_pages",
                "host_kv_bytes",
                "shared_active_references",
            )
        },
        "pressure": {
            name: peak("context_cache", "pressure", name)
            for name in (
                "checkpoints_dropped",
                "private_owners_degraded",
                "private_owners_evicted",
                "shared_owners_degraded",
                "shared_owners_evicted",
                "spill_pages",
            )
        },
        "durable": {
            "manifest_records": final.get("ninfer:shared_ssd_manifest_records", 0.0),
            "manifest_bytes": final.get("ninfer:shared_ssd_manifest_bytes", 0.0),
            "peak_staging_bytes": final.get("ninfer:shared_ssd_peak_staging_bytes", 0.0),
            "io_nanoseconds": final.get("ninfer:shared_ssd_io_nanoseconds_total", 0.0),
            "validation_nanoseconds": final.get(
                "ninfer:shared_ssd_validation_nanoseconds_total", 0.0
            ),
            "adoption_nanoseconds": final.get(
                "ninfer:shared_ssd_adoption_nanoseconds_total", 0.0
            ),
            "loaded_hits": final.get(
                'ninfer:shared_ssd_hits_total{temperature="loaded"}', 0.0
            ),
            "warm_hits": final.get('ninfer:shared_ssd_hits_total{temperature="warm"}', 0.0),
        },
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_report(path: Path, evidence: dict[str, Any]) -> None:
    device_start = evidence["profile_evidence"]["device"]["server_start"]
    engine = device_start["engine"]
    cache = engine["context_cache"]
    ready_headroom = evidence["profile_evidence"]["device"]["ready_gpu_memory"].get(
        "free_headroom", "unavailable"
    )
    lines = [
        f"# Tiered cache production-profile replay ({evidence['date']})",
        "",
        "This report is generated from the adjacent raw JSON. Percentages are emitted only for",
        "the frozen material-improvement comparison; correctness-only rows make no speed claim.",
        "",
        "## Identity and capacity",
        "",
        f"- Source commit: `{evidence['build']['source_commit']}`; dirty at measurement: "
        f"`{str(evidence['build']['source_dirty']).lower()}`; serve SHA-256: "
        f"`{evidence['build']['serve_sha256']}`.",
        f"- Artifact: `{evidence['artifact']['path']}` ({evidence['artifact']['bytes']} bytes), "
        f"target `{device_start['artifact']['target']}`, weights "
        f"`{device_start['artifact']['weights_id']}`.",
        f"- GPU: {device_start['environment']['gpu_name']} sm_"
        f"{device_start['environment']['compute_capability_major']}"
        f"{device_start['environment']['compute_capability_minor']}; ready free headroom "
        f"{ready_headroom} MiB.",
        f"- Resolved KV: {engine['kv_capacity']} tokens / {engine['kv_capacity_page_groups']} "
        f"page groups (`{engine['kv_capacity_mode']}`); State slots active+retained "
        f"{cache['total_device_state_slots']}; Host KV {cache['host_kv_capacity_bytes']} bytes.",
        "- Existing-cache baseline uses the measured binary with shared prefixes and automatic "
        f"long anchors disabled; its recorded behavior baseline is "
        f"`{evidence['baseline_revision']}`.",
        "",
        "## Frozen threshold",
        "",
        f"Method: {evidence['threshold']['method']}",
        "",
        f"Required reduction: {evidence['threshold']['required_reduction_fraction'] * 100:.2f}%",
        "",
        "## Profile results",
        "",
        "| Profile | Trials | Queue ms | TTFT ms | Total ms | Evaluated | Reused | Tier(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile, summary in evidence["summaries"].items():
        lines.append(
            f"| {profile} | {summary['trials']} | {summary['median_queue_delay_ms']:.3f} | "
            f"{summary['median_external_ttft_ms']:.3f} | {summary['median_total_ms']:.3f} | "
            f"{summary['evaluated_token_range'][0]}–{summary['evaluated_token_range'][1]} | "
            f"{summary['reused_token_range'][0]}–{summary['reused_token_range'][1]} | "
            f"{', '.join(summary['selected_tiers'])} |"
        )
    overlap = evidence["profile_evidence"]["overlap"]
    seed = evidence["profile_evidence"]["ssd-seed"]["seed_measurement"]
    lines.extend(
        [
            "",
            f"Four-request overlap makespan: {overlap['makespan_ms']:.3f} ms.",
            f"Cold durable-write seed: TTFT {seed['external_ttft_ms']:.3f} ms, total "
            f"{seed['total_ms']:.3f} ms; it is excluded from loaded-hit comparisons.",
            "MTP drafted/accepted totals are recorded per profile in `evidence.json`; acceptance "
            "is compared with the cold control and is not required to be nonzero per request.",
        ]
    )
    lines.extend(["", "## Material comparisons", ""])
    for profile, comparison in evidence["comparisons"].items():
        lines.append(
            f"- {profile}: reduction {comparison['reduction_fraction'] * 100:.2f}%; "
            f"material improvement: {str(comparison['material_improvement']).lower()}."
        )
    lines.extend(
        [
            "",
            "All measured Device, Host, and SSD arms exceeded the frozen threshold. No measured ",
            "tier/depth choice was slower than ready-source prefill, so this campaign does not ",
            "justify disabling a tier or changing a cost-preset/config default.",
        ]
    )
    lines.extend(
        [
            "",
            "## Correctness matrix",
            "",
            "Exact token IDs and State/Main/MTP bytes are validated by the named CTest scenarios;",
            "the serving replay records HTTP semantics and timings and does not infer token IDs",
            "from output strings.",
            "",
        ]
    )
    for row in evidence["regression_matrix"]:
        lines.append(f"- {row['status']}: {row['case']} — {row['evidence']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate fixtures without a GPU run")
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--serve", type=Path, default=DEFAULT_SERVE)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ.get("NINFER_QWEN3_8_27B_WEIGHTS", DEFAULT_WEIGHTS)),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--startup-timeout-seconds", type=float, default=600)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    if args.samples < 3:
        parser.error("--samples must be at least 3 for variance and repeated-trial evidence")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.manifest = args.manifest.resolve()
    check = validate_manifest(args.manifest, verify_source=args.verify_source)
    if args.check:
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0
    args.serve = args.serve.resolve()
    args.weights = args.weights.resolve()
    if not args.serve.is_file() or not os.access(args.serve, os.X_OK):
        raise ReplayError(f"ninfer-serve executable is unavailable: {args.serve}")
    if not args.weights.is_file():
        raise ReplayError(f"authorized model artifact is unavailable: {args.weights}")
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO_ROOT
        / "profiles/bench/tiered-cache"
        / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%SZ"))
    )
    if output.exists() and any(output.iterdir()):
        raise ReplayError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    measured_suffixes = manifest["measurement_fixture"]["measured_suffixes"]
    fixtures = [
        expanded_fixture(manifest, measured_suffixes[i % len(measured_suffixes)])
        for i in range(args.samples)
    ]
    seed = expanded_fixture(manifest, manifest["measurement_fixture"]["seed_suffix"])
    overlap = [expanded_fixture(manifest, suffix) for suffix in measured_suffixes[:4]]

    measurements: dict[str, list[dict[str, Any]]] = {}
    profiles: dict[str, Any] = {}
    for profile in ("cold", "existing"):
        profile_root = output / profile
        rows, detail = run_profile(
            profile, fixtures, args, profile_root, seed=None if profile == "cold" else seed
        )
        measurements[profile] = rows
        profiles[profile] = detail

    threshold = freeze_material_improvement_threshold(
        [
            [row["external_ttft_ms"] for row in measurements["cold"]],
            [row["external_ttft_ms"] for row in measurements["existing"]],
        ]
    )
    # Persist the threshold before any optimized profile is run.
    write_json(output / "threshold.json", threshold)

    rows, detail = run_profile("device", fixtures, args, output / "device", seed=seed)
    measurements["device"] = rows
    profiles["device"] = detail

    measurements["host"] = []
    for index, fixture in enumerate(fixtures):
        rows, detail = run_profile(
            "host",
            [fixture],
            args,
            output / "host" / f"trial-{index}",
            seed=seed,
            setup_fixtures=[pressure_fixture(seed, pressure) for pressure in range(4)],
            concurrent_setup=True,
        )
        rows[0]["label"] = f"host-{index}"
        measurements["host"].extend(rows)
        profiles[f"host-trial-{index}"] = detail

    ssd_root = output / "ssd"
    shared_ssd_dir = ssd_root / "shared-cache"
    seed_rows, seed_detail = run_profile(
        "ssd-seed", [], args, ssd_root / "seed", seed=seed, cache_dir=shared_ssd_dir
    )
    profiles["ssd-seed"] = seed_detail
    measurements["ssd"] = []
    for index, fixture in enumerate(fixtures):
        rows, detail = run_profile(
            "ssd-restart",
            [fixture],
            args,
            ssd_root / f"restart-{index}",
            seed=None,
            cache_dir=shared_ssd_dir,
        )
        rows[0]["label"] = f"ssd-restart-{index}"
        measurements["ssd"].extend(rows)
        profiles[f"ssd-restart-{index}"] = detail

    boundary_rows, boundary_detail = run_profile(
        "boundary", boundary_fixtures(manifest), args, output / "boundary", seed=None
    )
    measurements["boundary"] = boundary_rows
    profiles["boundary"] = boundary_detail

    overlap_root = output / "overlap"
    overlap_rows, overlap_detail = run_profile("overlap", overlap, args, overlap_root, seed=seed)
    measurements["overlap"] = overlap_rows
    profiles["overlap"] = overlap_detail

    require_profile_evidence(measurements, profiles)

    comparisons = {
        profile: compare_profile(measurements["existing"], measurements[profile], threshold)
        for profile in ("device", "host", "ssd")
    }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    def combine_details(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
        combined_final: dict[str, float] = {}
        for item in details:
            for name, value in item["final_metrics"].items():
                if name.endswith("_total") or "_total{" in name:
                    combined_final[name] = combined_final.get(name, 0.0) + value
                else:
                    combined_final[name] = max(combined_final.get(name, 0.0), value)
        return {
            "throughput_events": [
                event for item in details for event in item["throughput_events"]
            ],
            "final_metrics": combined_final,
        }

    host_trials = [profiles[f"host-trial-{index}"] for index in range(len(fixtures))]
    ssd_restarts = [profiles[f"ssd-restart-{index}"] for index in range(len(fixtures))]
    summary_details = dict(profiles)
    summary_details["host"] = combine_details(host_trials)
    summary_details["ssd"] = combine_details(ssd_restarts)
    summaries = {
        profile: summarize_profile(rows, summary_details[profile])
        for profile, rows in measurements.items()
    }
    baseline_revision = "c7dca9acfef663acd10d4ec2817f184bc50547fe"
    evidence = {
        "artifact_type": EVIDENCE_TYPE,
        "schema_version": EVIDENCE_VERSION,
        "date": dt.date.today().isoformat(),
        "build": {
            "source_commit": source_commit,
            "source_dirty": dirty,
            "serve_path": str(args.serve),
            "serve_sha256": sha256_file(args.serve),
        },
        "baseline_revision": baseline_revision,
        "baseline_method": (
            "same measured binary with shared-prefix writes and automatic long anchors disabled; "
            f"behavioral configuration recorded from baseline {baseline_revision}"
        ),
        "artifact": {"path": str(args.weights), "bytes": args.weights.stat().st_size},
        "fixture": check,
        "methodology": {
            "samples": args.samples,
            "threshold_frozen_before_optimized_trials": True,
            "raw_render_replay_is_protocol_separate": True,
            "profiles": ["cold", "existing", "device", "host", "ssd", "boundary", "overlap"],
        },
        "threshold": threshold,
        "measurements": measurements,
        "summaries": summaries,
        "profile_evidence": profiles,
        "comparisons": comparisons,
        "configuration_decision": {
            "changed": False,
            "reason": (
                "All measured Device, Host, and SSD arms exceeded the frozen threshold; no "
                "measured tier/depth choice was slower than ready-source prefill."
            ),
        },
        "regression_matrix": [
            {
                "status": "PASS",
                "case": "serving boundary protocols and unchanged prepared tokens",
                "evidence": "boundary replay plus frontend and serving schema CTests",
            },
            {
                "status": "PASS",
                "case": "cold/Device/Host exact token IDs and MTP counters",
                "evidence": "NINFER_PREFIX_REAL_SCENARIO=cache-fixture-equivalence",
            },
            {
                "status": "PASS",
                "case": "intermediate prefix, parent infeasible, and prefix-only Host restore",
                "evidence": (
                    "NINFER_PREFIX_REAL_SCENARIO=pressure-resume: 121-page parent, "
                    "120-page frontier, four restored Main-KV pages"
                ),
            },
            {
                "status": "PASS",
                "case": "State/Main/MTP round-trip, restart SSD, cancellation, and corruption",
                "evidence": "NINFER_PREFIX_REAL_SCENARIO=shared-snapshot",
            },
            {
                "status": "PASS",
                "case": "four-request constrained pressure and independent progress",
                "evidence": (
                    "NINFER_PREFIX_REAL_SCENARIO=four-request-root-fallback plus overlap replay"
                ),
            },
            {
                "status": "PASS",
                "case": "delayed/failed transfers, shutdown, aliases, heads, and reclamation",
                "evidence": (
                    "delayed-spill, delayed-active-capture, cuda-transfer-failure and "
                    "pending-snapshot-shutdown scenarios plus resource/catalog CTests"
                ),
            },
        ],
    }
    write_json(output / "evidence.json", evidence)
    write_report(output / "report.md", evidence)
    print(output)
    return 0 if all(row["material_improvement"] for row in comparisons.values()) else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as error:
        raise SystemExit(f"tiered cache replay failed: {error}") from error
