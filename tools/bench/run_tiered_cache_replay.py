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
    load_predecessor_calibration,
    load_manifest,
    pressure_fixture,
    require_unchanged_frozen_threshold,
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
DEFAULT_PREDECESSOR_CALIBRATION = (
    REPO_ROOT / "bench/fixtures/tiered_cache/predecessor-calibration.json"
)
DEFAULT_SERVE = REPO_ROOT / "build-agent-verify/apps/ninfer-serve"
DEFAULT_WEIGHTS = Path("/models/qwen3_8_27b.ninfer")
REQUEST_LOG_SCHEMA = 20
BASELINE_REQUEST_LOG_SCHEMA = 19
BASELINE_REVISION = "c7dca9acfef663acd10d4ec2817f184bc50547fe"


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
        try:
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
        except BaseException:
            # __exit__ is not invoked when __enter__ raises. Own every post-launch failure here so
            # a failed startup cannot retain GPU memory, the TCP port, or an open log descriptor.
            self.close()
            raise

    def close(self) -> None:
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    self.process.wait()
        finally:
            if self.output is not None:
                self.output.close()
                self.output = None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


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


def load_events(
    path: Path, *, allowed_schemas: Sequence[int] = (REQUEST_LOG_SCHEMA,)
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            event = json.loads(line)
            if event.get("schema_version") not in allowed_schemas:
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
) -> tuple[float, ServeExchangeResult, str]:
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
    response_ids = {event.response_id for event in result.events if event.response_id is not None}
    if len(response_ids) != 1:
        raise ReplayError(f"protocol response has no unique request identity: {response_ids!r}")
    return external_ttft_ms, result, response_ids.pop()


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


def latest_done(
    path: Path,
    prior_count: int,
    timeout: float,
    *,
    expected_response_id: str | None,
    allowed_schemas: Sequence[int],
) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + timeout
    found = 0
    while time.monotonic() < deadline:
        events = load_events(path, allowed_schemas=allowed_schemas)
        done = [event for event in events if event.get("event") == "request_done"]
        found = len(done)
        if found == prior_count + 1:
            event = done[-1]
            logged_response_id = event.get("request", {}).get("response_id")
            if expected_response_id is not None and logged_response_id != expected_response_id:
                raise ReplayError(
                    "sequential response/request-log identity mismatch: "
                    f"{expected_response_id!r} != {logged_response_id!r}"
                )
            return event, found
        if found > prior_count + 1:
            raise ReplayError(
                f"request log advanced by more than one event: {prior_count} -> {found}"
            )
        time.sleep(0.02)
    raise ReplayError(f"expected {prior_count + 1} request_done events, found {found}")


def done_by_response_id(
    path: Path,
    prior_count: int,
    expected: Sequence[str],
    timeout: float,
    *,
    allowed_schemas: Sequence[int],
) -> dict[str, dict[str, Any]]:
    if len(set(expected)) != len(expected):
        raise ReplayError("concurrent protocol responses have duplicate response identities")
    deadline = time.monotonic() + timeout
    done: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        done = [
            event
            for event in load_events(path, allowed_schemas=allowed_schemas)
            if event.get("event") == "request_done"
        ][prior_count:]
        if len(done) >= len(expected):
            break
        time.sleep(0.02)
    if len(done) != len(expected):
        raise ReplayError(
            f"concurrent batch produced {len(done)} request_done events, expected {len(expected)}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for event in done:
        response_id = event.get("request", {}).get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in indexed:
            raise ReplayError("concurrent request log has a missing or duplicate response identity")
        indexed[response_id] = event
    if set(indexed) != set(expected):
        raise ReplayError(
            "concurrent response/request-log identities differ: "
            f"responses={sorted(expected)!r} logs={sorted(indexed)!r}"
        )
    return indexed


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
    baseline_profile = profile == "existing"
    serve = args.baseline_serve if baseline_profile else args.serve
    allowed_schemas = (
        (BASELINE_REQUEST_LOG_SCHEMA,) if baseline_profile else (REQUEST_LOG_SCHEMA,)
    )
    command = server_command(serve, args.weights, args.port, request_log, profile, cache_dir)
    measurements: list[dict[str, Any]] = []
    seed_measurement: dict[str, Any] | None = None
    setup_measurements: list[dict[str, Any]] = []
    setup_makespan_ms: float | None = None
    with RunningServe(command, serve_log, args.port, args.startup_timeout_seconds):
        events = load_events(request_log, allowed_schemas=allowed_schemas)
        start = next((event for event in events if event.get("event") == "server_start"), None)
        if start is None:
            raise ReplayError("server_start event is missing")
        ready_gpu_memory = gpu_memory_snapshot()
        done_count = 0
        if seed is not None:
            external, _, response_id = run_exchange(
                args.port, seed, args.request_timeout_seconds
            )
            event, done_count = latest_done(
                request_log,
                done_count,
                args.request_timeout_seconds,
                expected_response_id=None if baseline_profile else response_id,
                allowed_schemas=allowed_schemas,
            )
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
                tuple[float, ServeExchangeResult, str] | BaseException | None
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
            response_ids = [
                value[2]
                for value in setup_results
                if isinstance(value, tuple)
            ]
            failures = [value for value in setup_results if isinstance(value, BaseException)]
            if failures:
                raise failures[0]
            if len(response_ids) != len(setup_results):
                raise ReplayError("concurrent setup worker produced no result")
            done = done_by_response_id(
                request_log,
                done_count,
                response_ids,
                args.request_timeout_seconds,
                allowed_schemas=allowed_schemas,
            )
            for index, value in enumerate(setup_results):
                assert isinstance(value, tuple)
                setup_measurements.append(
                    request_measurement(
                        f"{profile}-setup-{index}",
                        value[0],
                        done[value[2]],
                        payload_sha256=setup_fixtures[index]["payload_sha256"],
                    )
                )
            done_count += len(done)
        else:
            for index, fixture in enumerate(setup_fixtures):
                external, _, response_id = run_exchange(
                    args.port, fixture, args.request_timeout_seconds
                )
                event, done_count = latest_done(
                    request_log,
                    done_count,
                    args.request_timeout_seconds,
                    expected_response_id=None if baseline_profile else response_id,
                    allowed_schemas=allowed_schemas,
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
            results: list[tuple[float, ServeExchangeResult, str] | BaseException | None] = [
                None
            ] * len(fixtures)
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
            response_ids = [value[2] for value in results if isinstance(value, tuple)]
            failures = [value for value in results if isinstance(value, BaseException)]
            if failures:
                raise failures[0]
            if len(response_ids) != len(results):
                raise ReplayError("overlap worker produced no result")
            done = done_by_response_id(
                request_log,
                done_count,
                response_ids,
                args.request_timeout_seconds,
                allowed_schemas=allowed_schemas,
            )
            for index, value in enumerate(results):
                assert isinstance(value, tuple)
                measurements.append(
                    request_measurement(
                        f"{profile}-{index}",
                        value[0],
                        done[value[2]],
                        payload_sha256=fixtures[index]["payload_sha256"],
                    )
                )
            profile_extra: dict[str, Any] = {"makespan_ms": makespan_ms}
        else:
            for index, fixture in enumerate(fixtures):
                external, _, response_id = run_exchange(
                    args.port, fixture, args.request_timeout_seconds
                )
                event, done_count = latest_done(
                    request_log,
                    done_count,
                    args.request_timeout_seconds,
                    expected_response_id=None if baseline_profile else response_id,
                    allowed_schemas=allowed_schemas,
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
                    for event in load_events(request_log, allowed_schemas=allowed_schemas)
                    if event.get("event") == "throughput"
                ],
                "ready_gpu_memory": ready_gpu_memory,
                "settled_gpu_memory": gpu_memory_snapshot(),
                "command": command,
                "request_log_schema": start["schema_version"],
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
        host_h2d_count = sum(
            int(event["context_cache"]["state_transfers"]["h2d"]["count"])
            for event in profiles[name]["throughput_events"]
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

    def total(*path: str) -> float:
        values: list[float] = []
        for event in throughput:
            value: Any = event
            for component in path:
                value = value.get(component, {}) if isinstance(value, dict) else {}
            if isinstance(value, (int, float)):
                values.append(float(value))
        return sum(values)

    # request_log.cpp publishes these as interval deltas. Totals must sum intervals; only live
    # occupancy is a gauge for which a peak is meaningful.
    transfer: dict[str, Any] = {
        "actual_seconds": total("context_cache", "actual_transfer_seconds")
    }
    for resource in ("state_transfers", "main_kv_transfers", "backend_kv_transfers"):
        fields = (
            ("count", "bytes", "seconds")
            if resource == "state_transfers"
            else ("pages", "bytes", "seconds")
        )
        transfer[resource] = {
            direction: {
                field: total("context_cache", resource, direction, field)
                for field in fields
            }
            for direction in ("d2d", "d2h", "h2d")
        }
    elapsed = sum(float(event.get("interval_seconds", 0.0)) for event in throughput)
    computed = sum(int(event.get("tokens", {}).get("computed_prefill", 0)) for event in throughput)
    committed = sum(int(event.get("tokens", {}).get("committed_decode", 0)) for event in throughput)
    final = detail["final_metrics"]

    def reclaimed(tier: str, resource: str) -> float:
        name = (
            "ninfer:context_cache_reclaimed_capacity_total"
            f'{{tier="{tier}",resource="{resource}"}}'
        )
        return float(final.get(name, 0.0))

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
            name: total("context_cache", "pressure", name)
            for name in (
                "spill_pages",
                "partial_tail_cow_pages",
                "checkpoints_dropped",
                "private_owners_degraded",
                "private_owners_evicted",
                "shared_owners_degraded",
                "shared_owners_evicted",
                "searches",
                "search_budget_exhaustions",
                "maximal_fallback_selections",
                "historical_fork_hits",
            )
        },
        "reclaimed_capacity": {
            "device_state_slots": reclaimed("device", "state_slots"),
            "device_main_kv_pages": reclaimed("device", "main_kv_pages"),
            "device_backend_kv_pages": reclaimed("device", "backend_kv_pages"),
            "host_state_slots": reclaimed("host", "state_slots"),
            "host_kv_bytes": reclaimed("host", "kv_bytes"),
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
            "warm_hits": final.get(
                'ninfer:shared_ssd_hits_total{temperature="warm"}', 0.0
            ),
        },
    }


def validate_measured_continuations(
    measurements: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    cold_by_fixture: dict[str, list[dict[str, Any]]] = {}
    for row in measurements["cold"]:
        fixture_hash = str(row["payload_sha256"])
        if row.get("generated_token_ids") is None:
            raise ReplayError("current cold oracle did not log exact generated token IDs")
        cold_by_fixture.setdefault(fixture_hash, []).append(row)

    comparisons: list[dict[str, Any]] = []
    token_mismatches: list[dict[str, Any]] = []
    speculative_mismatches: list[dict[str, Any]] = []
    speculative_fields = (
        "draft_window",
        "rounds",
        "drafted_tokens",
        "accepted_tokens",
        "fallback_steps",
        "accepted_per_position",
    )
    for profile in ("device", "host", "ssd"):
        if sorted(str(row["payload_sha256"]) for row in measurements[profile]) != sorted(
            str(row["payload_sha256"]) for row in measurements["cold"]
        ):
            raise ReplayError(f"{profile} fixture multiset differs from the cold oracle")
        fixture_occurrences: dict[str, int] = {}
        for row in measurements[profile]:
            fixture_hash = str(row["payload_sha256"])
            occurrence = fixture_occurrences.get(fixture_hash, 0)
            fixture_occurrences[fixture_hash] = occurrence + 1
            cold_rows = cold_by_fixture.get(fixture_hash, [])
            if occurrence >= len(cold_rows):
                raise ReplayError(f"{profile} row has no fixture-aligned cold oracle")
            cold = cold_rows[occurrence]
            token_ids = row.get("generated_token_ids")
            if token_ids is None:
                raise ReplayError(f"{profile} row did not log exact generated token IDs")
            exact_token_ids = token_ids == cold["generated_token_ids"]
            first_token_mismatch: dict[str, Any] | None = None
            if not exact_token_ids:
                cold_ids = cold["generated_token_ids"]
                shared_length = min(len(cold_ids), len(token_ids))
                first_index = next(
                    (
                        index
                        for index in range(shared_length)
                        if cold_ids[index] != token_ids[index]
                    ),
                    shared_length,
                )
                first_token_mismatch = {
                    "index": first_index,
                    "cold_token_id": cold_ids[first_index] if first_index < len(cold_ids) else None,
                    "subject_token_id": (
                        token_ids[first_index] if first_index < len(token_ids) else None
                    ),
                    "cold_length": len(cold_ids),
                    "subject_length": len(token_ids),
                }
            cold_speculative = {
                name: cold["speculative"].get(name) for name in speculative_fields
            }
            subject_speculative = {
                name: row["speculative"].get(name) for name in speculative_fields
            }
            counters_match = cold_speculative == subject_speculative
            counter_deltas = {
                name: subject_speculative[name] - cold_speculative[name]
                for name in speculative_fields
                if isinstance(subject_speculative[name], int)
                and isinstance(cold_speculative[name], int)
            }
            comparison = {
                "fixture_sha256": fixture_hash,
                "fixture_occurrence": occurrence,
                "profile": profile,
                "cold_label": cold["label"],
                "subject_label": row["label"],
                "cold_generated_token_ids_sha256": cold["generated_token_ids_sha256"],
                "generated_token_ids_sha256": row["generated_token_ids_sha256"],
                "exact_generated_token_ids": exact_token_ids,
                "first_token_mismatch": first_token_mismatch,
                "cold_speculative": cold_speculative,
                "subject_speculative": subject_speculative,
                "speculative_counter_deltas": counter_deltas,
                "speculative_counters_match": counters_match,
            }
            comparisons.append(comparison)
            if not exact_token_ids:
                token_mismatches.append(comparison)
            if not counters_match:
                speculative_mismatches.append(comparison)

    token_status = "PASS" if not token_mismatches else "FAIL"
    speculative_status = "PASS" if not speculative_mismatches else "FAIL"
    token_investigation = [
        {
            "fixture_sha256": mismatch["fixture_sha256"],
            "cold_label": mismatch["cold_label"],
            "subject_label": mismatch["subject_label"],
            **mismatch["first_token_mismatch"],
            "root_cause": "not established by the serving replay",
            "verdict": "FAIL",
        }
        for mismatch in token_mismatches
    ]
    cached_consensus = True
    for fixture_hash, cold_rows in cold_by_fixture.items():
        for occurrence in range(len(cold_rows)):
            cached_ids = [
                row["generated_token_ids"]
                for profile in ("device", "host", "ssd")
                for row in measurements[profile]
                if row["payload_sha256"] == fixture_hash
            ][occurrence::len(cold_rows)]
            if cached_ids and any(token_ids != cached_ids[0] for token_ids in cached_ids[1:]):
                cached_consensus = False
    investigation = []
    for mismatch in speculative_mismatches:
        cold = mismatch["cold_speculative"]
        subject = mismatch["subject_speculative"]
        signature = "counter deltas " + json.dumps(
            mismatch["speculative_counter_deltas"], sort_keys=True
        )
        cold_positions = cold["accepted_per_position"]
        subject_positions = subject["accepted_per_position"]
        position_deltas = [
            subject_value - cold_value
            for cold_value, subject_value in zip(cold_positions, subject_positions)
        ]
        signature += f"; accepted-per-position deltas {position_deltas}"
        if (
            mismatch["exact_generated_token_ids"]
            and cold["rounds"] == subject["rounds"]
            and cold["accepted_tokens"] + cold["fallback_steps"]
            == subject["accepted_tokens"] + subject["fallback_steps"]
        ):
            signature += (
                "; the same exact target continuation and round count localize this to a draft "
                "acceptance versus target fallback accounting difference"
            )
        investigation.append(
            {
                "fixture_sha256": mismatch["fixture_sha256"],
                "cold_label": mismatch["cold_label"],
                "subject_label": mismatch["subject_label"],
                "finding": signature,
                "root_cause": "not established by the serving replay",
                "verdict": "FAIL",
            }
        )
    return {
        "status": "PASS" if token_status == speculative_status == "PASS" else "FAIL",
        "target_token_status": token_status,
        "speculative_counter_status": speculative_status,
        "method": "fixture-hash-aligned exact generated token IDs against cache-disabled cold",
        "comparisons": comparisons,
        "target_token_mismatches": token_mismatches,
        "target_token_investigation": token_investigation,
        "cached_tier_consensus": "PASS" if cached_consensus else "FAIL",
        "localization": (
            "Device, Host, and post-restart SSD continuations agree exactly with one another. "
            "The measured divergence is therefore between cache-disabled Root prefill and the "
            "cache-participating prefill/capture schedule, not between retention tiers; this "
            "does not establish the numerical root cause."
            if cached_consensus and token_mismatches
            else "No cross-tier localization is needed."
        ),
        "speculative_counter_mismatches": speculative_mismatches,
        "investigation": investigation,
        "conclusion": (
            "Exact target token IDs and all MTP counters match every fixture-aligned cold control."
            if token_status == speculative_status == "PASS"
            else "The replay does not establish full continuation equivalence; inspect the "
            "recorded mismatch rows before making a production-correctness claim."
        ),
    }


def load_baseline_identity(path: Path, serve: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read baseline identity {path}: {error}") from error
    if not isinstance(value, dict) or (
        value.get("artifact_type"), value.get("schema_version")
    ) != ("ninfer_tiered_cache_baseline_build", 1):
        raise ReplayError("unsupported baseline build identity")
    if value.get("source_revision") != BASELINE_REVISION:
        raise ReplayError(
            f"baseline build revision is {value.get('source_revision')!r}, expected "
            f"{BASELINE_REVISION}"
        )
    archive_sha256 = value.get("source_archive_sha256")
    if not isinstance(archive_sha256, str) or len(archive_sha256) != 64:
        raise ReplayError("baseline build identity has no source archive SHA-256")
    actual_sha256 = sha256_file(serve)
    if value.get("serve_sha256") != actual_sha256:
        raise ReplayError("baseline executable SHA-256 differs from its build identity")
    return {
        "source_revision": BASELINE_REVISION,
        "source_archive_sha256": archive_sha256,
        "serve_path": str(serve),
        "serve_sha256": actual_sha256,
        "build_identity_path": str(path),
        "build_identity_sha256": sha256_file(path),
        "build_command": value.get("build_command"),
        "compiler": value.get("compiler"),
    }


def load_validation_cases(
    path: Path | None, serve_sha256: str | None
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read validation evidence {path}: {error}") from error
    if not isinstance(value, dict) or (
        value.get("artifact_type"), value.get("schema_version")
    ) != ("ninfer_tiered_cache_validation_evidence", 1):
        raise ReplayError("unsupported tiered-cache validation evidence")
    if serve_sha256 is not None and value.get("serve_sha256") != serve_sha256:
        raise ReplayError("validation evidence was produced by a different current executable")
    cases = value.get("cases")
    if not isinstance(cases, dict):
        raise ReplayError("validation evidence has no case map")
    result: dict[str, dict[str, Any]] = {}
    for name, case in cases.items():
        if not isinstance(name, str) or not isinstance(case, dict):
            raise ReplayError("validation evidence case is malformed")
        status = case.get("status")
        if status not in {"PASS", "FAIL", "SKIP", "UNVERIFIED"}:
            raise ReplayError(f"validation evidence case {name!r} has invalid status")
        executable_sha256 = case.get("test_executable_sha256")
        if name != "official-tokenizer-lineage" and status != "UNVERIFIED" and (
            not isinstance(executable_sha256, str) or len(executable_sha256) != 64
        ):
            raise ReplayError(
                f"validation evidence case {name!r} has no test executable SHA-256"
            )
        result[name] = case
    return result


def validation_status(
    cases: dict[str, dict[str, Any]], required: Sequence[str]
) -> tuple[str, str]:
    selected = [cases.get(name) for name in required]
    missing = [name for name, case in zip(required, selected) if case is None]
    statuses = [str(case["status"]) for case in selected if case is not None]
    evidence = "; ".join(
        f"{name}={case['status']} ({case.get('evidence', case.get('command', 'recorded'))}; "
        f"executable sha256={case.get('test_executable_sha256', 'unavailable')})"
        for name, case in zip(required, selected)
        if case is not None
    )
    if "FAIL" in statuses:
        suffix = "; missing: " + ", ".join(missing) if missing else ""
        return "FAIL", evidence + suffix
    if missing:
        return "UNVERIFIED", "missing identified validation evidence: " + ", ".join(missing)
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS", evidence
    return "UNVERIFIED", evidence


def official_tokenizer_status(
    cases: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    case = cases.get("official-tokenizer-lineage")
    if case is None:
        return (
            "UNVERIFIED",
            "missing identified validation evidence: official-tokenizer-lineage",
        )
    if case.get("status") == "FAIL":
        return "FAIL", str(case.get("evidence", "recorded failure"))
    artifact = case.get("tokenizer_artifact")
    if case.get("status") != "PASS" or not isinstance(artifact, dict):
        return (
            "UNVERIFIED",
            "official-tokenizer evidence is absent, skipped, or not identity-pinned",
        )
    path_value = artifact.get("path")
    expected_sha256 = artifact.get("sha256")
    if (
        artifact.get("authorization") != "user-authorized-local-artifact"
        or not isinstance(path_value, str)
        or not isinstance(expected_sha256, str)
    ):
        return "UNVERIFIED", "official-tokenizer evidence lacks authorized local artifact identity"
    tokenizer_path = Path(path_value).resolve()
    if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != expected_sha256:
        return (
            "UNVERIFIED",
            "authorized official-tokenizer artifact is unavailable or hash-mismatched",
        )
    return (
        "PASS",
        f"authorized identity-pinned local tokenizer: {tokenizer_path} ({expected_sha256})",
    )


def build_regression_matrix(
    continuation: dict[str, Any],
    validation_cases: dict[str, dict[str, Any]],
    boundary_replay: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    def external(case: str, required: Sequence[str]) -> dict[str, str]:
        status, evidence = validation_status(validation_cases, required)
        return {"status": status, "case": case, "evidence": evidence}

    return [
        boundary_replay
        or {
            "status": "UNVERIFIED",
            "case": "live serving boundary protocols",
            "evidence": "no live boundary replay result was supplied",
        },
        external(
            "unchanged prepared tokens and explicit/inferred boundary lineage",
            [
                "frontend-boundary-token-lineage",
                "openai-chat-boundary",
                "openai-responses-boundary",
                "anthropic-boundary",
            ],
        ),
        {
            "status": str(continuation["target_token_status"]),
            "case": "measured cold/Device/Host/SSD exact generated token IDs",
            "evidence": (
                f"{len(continuation['comparisons'])} fixture-aligned comparisons; "
                f"{len(continuation['target_token_mismatches'])} token-ID mismatches"
            ),
        },
        {
            "status": str(continuation["speculative_counter_status"]),
            "case": "measured cold/Device/Host/SSD MTP round/drafted/accepted/fallback counters",
            "evidence": (
                f"{len(continuation['comparisons'])} fixture-aligned comparisons; "
                f"{len(continuation['speculative_counter_mismatches'])} counter mismatches with "
                "per-field deltas and investigation records"
            ),
        },
        external(
            "intermediate prefix, parent infeasible, and prefix-only Host restore",
            ["pressure-resume"],
        ),
        external(
            "complete private State/Main/MTP Host materialization",
            ["host-restore"],
        ),
        external(
            "State/Main/MTP round-trip, restart SSD, cancellation, and corruption",
            ["shared-snapshot"],
        ),
        external(
            "four-request constrained pressure and independent progress",
            ["four-request-root-fallback"],
        ),
        external(
            "delayed/failed transfers, shutdown, aliases, heads, and reclamation",
            [
                "delayed-spill",
                "delayed-active-capture",
                "cuda-transfer-failure",
                "pending-snapshot-shutdown",
                "resource-manager",
                "durable-shared-prefix-catalog",
            ],
        ),
    ]


def build_supplemental_evidence(
    validation_cases: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    tokenizer_status, tokenizer_evidence = official_tokenizer_status(validation_cases)
    return [
        {
            "status": tokenizer_status,
            "case": "official-tokenizer lineage for frontend boundary fixtures",
            "evidence": tokenizer_evidence,
        }
    ]


def configuration_decision(
    comparisons: dict[str, dict[str, Any]], correctness_status: str = "PASS"
) -> dict[str, Any]:
    material = [name for name, row in comparisons.items() if row["material_improvement"]]
    missed = [name for name, row in comparisons.items() if not row["material_improvement"]]
    slower = [name for name, row in comparisons.items() if row["reduction_fraction"] < 0]
    actions = {
        name: (
            "retain"
            if row["material_improvement"]
            else "disable"
            if row["reduction_fraction"] < 0
            else "material-improvement-unverified"
        )
        for name, row in comparisons.items()
    }
    if correctness_status != "PASS":
        return {
            "status": correctness_status,
            "material_profiles": material,
            "missed_profiles": missed,
            "slower_than_existing_baseline": slower,
            "tier_actions": {name: "blocked-by-correctness" for name in comparisons},
            "reason": (
                "The speed comparisons are retained as measurements, but correctness is "
                f"{correctness_status}; no tier or cost-preset/config decision is justified."
            ),
        }
    return {
        "status": "PASS" if not missed else "FAIL",
        "material_profiles": material,
        "missed_profiles": missed,
        "slower_than_existing_baseline": slower,
        "tier_actions": actions,
        "reason": (
            "Every measured tier exceeded the frozen threshold; no tier was slower than the "
            "recorded-revision existing-cache baseline, so these measurements do not justify a "
            "cost-preset/config change."
            if not missed
            else "The measured evidence does not support every tier. Profiles below the frozen "
            f"threshold: {', '.join(missed)}; profiles slower than the existing-cache baseline: "
            f"{', '.join(slower) if slower else 'none'}."
        ),
    }


def boundary_replay_status(rows: Sequence[dict[str, Any]], expected: int) -> dict[str, str]:
    identities = [row.get("response_id") for row in rows]
    status = (
        "PASS"
        if len(rows) == expected
        and expected > 0
        and all(isinstance(identity, str) and identity for identity in identities)
        and len(set(identities)) == len(identities)
        else "FAIL"
    )
    return {
        "status": status,
        "case": "live serving boundary protocols",
        "evidence": (
            f"{len(rows)}/{expected} fixture shapes completed through public endpoints with "
            f"{len(set(identities))} unique wire response identities"
        ),
    }


def campaign_verdict(
    comparisons: dict[str, dict[str, Any]], regression_matrix: Sequence[dict[str, str]]
) -> dict[str, Any]:
    measured_performance = (
        "PASS" if all(row["material_improvement"] for row in comparisons.values()) else "FAIL"
    )
    statuses = [row["status"] for row in regression_matrix]
    correctness = (
        "FAIL"
        if "FAIL" in statuses
        else "UNVERIFIED"
        if "UNVERIFIED" in statuses
        else "PASS"
    )
    performance = (
        "FAIL"
        if "FAIL" in {measured_performance, correctness}
        else "UNVERIFIED"
        if correctness == "UNVERIFIED"
        else "PASS"
    )
    overall = "FAIL" if "FAIL" in {performance, correctness} else correctness
    return {
        "overall": overall,
        "performance": performance,
        "performance_measurement": measured_performance,
        "correctness": correctness,
        "reason": (
            "Target production acceptance and the material-improvement claim are established "
            "by the required evidence; supplemental evidence is reported separately."
            if overall == "PASS"
            else "Target production acceptance is not established; inspect required failed and "
            "unverified rows and the measured comparisons."
        ),
    }


def campaign_exit_status(verdict: dict[str, Any]) -> int:
    return 0 if verdict["overall"] == "PASS" else 3


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
    validation_identity = evidence.get("validation_evidence_identity")
    validation_identity_line = (
        "- Required validation revision: "
        f"`{validation_identity['source_revision']}` (descendant of the measured build); "
        f"serve SHA-256 `{validation_identity['serve_sha256']}`; evidence SHA-256 "
        f"`{validation_identity['sha256']}`."
        if isinstance(validation_identity, dict)
        else "- Required validation identity is embedded per case in the raw evidence."
    )
    lines = [
        f"# Tiered cache production-profile replay ({evidence['date']})",
        "",
        "This report is generated from the adjacent raw JSON. Percentages are emitted only for",
        "the frozen material-improvement comparison; correctness-only rows make no speed claim.",
        f"Overall verdict: **{evidence['verdict']['overall']}**. "
        f"{evidence['verdict']['reason']}",
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
        f"- Existing-cache baseline revision: `{evidence['baseline']['source_revision']}`; "
        f"separate serve SHA-256: `{evidence['baseline']['serve_sha256']}`; build identity: "
        f"`{evidence['baseline']['build_identity_path']}`.",
        validation_identity_line,
        "",
        "## Frozen threshold",
        "",
        f"Method: {evidence['threshold']['method']}",
        "",
        "Predecessor calibration: "
        f"`{evidence['threshold']['predecessor_calibration']['source_report']}` at "
        f"`{evidence['threshold']['predecessor_calibration']['source_commit']}`; complete-private "
        "Host H2D effect-size reference "
        f"{evidence['threshold']['calibration_effect_size_ms']:.3f} ms "
        f"({evidence['threshold']['calibration_fraction'] * 100:.3f}% of the existing-cache "
        "median).",
        "The predecessor supplied exact State/Main/MTP bytes and its selected cost-model "
        "coefficients; this effect-size floor is not reported as measured latency.",
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
    lines.extend(["", evidence["configuration_decision"]["reason"]])
    lines.extend(
        [
            "",
            "## Required correctness matrix",
            "",
            "The serving replay retains exact generated token IDs and joins concurrent wire/log",
            "measurements by response identity. Low-level statuses require imported, executable-",
            "identified validation evidence; absent or skipped evidence remains UNVERIFIED.",
            "",
        ]
    )
    for row in evidence["required_regression_matrix"]:
        lines.append(f"- {row['status']}: {row['case']} — {row['evidence']}")
    lines.extend(
        [
            "",
            "## Supplemental evidence",
            "",
            "Supplemental rows record additional provenance and do not gate target correctness,",
            "the material-improvement claim, configuration decisions, the overall verdict, or",
            "the replay exit status.",
            "",
        ]
    )
    for row in evidence["supplemental_evidence"]:
        lines.append(f"- {row['status']}: {row['case']} — {row['evidence']}")
    continuation = evidence["continuation_validation"]
    lines.extend(
        [
            "",
            "## Continuation investigation",
            "",
            continuation["conclusion"],
        ]
    )
    for finding in continuation["investigation"]:
        lines.append(
            f"- {finding['verdict']}: {finding['cold_label']} versus "
            f"{finding['subject_label']} — {finding['finding']}; root cause: "
            f"{finding['root_cause']}."
        )
    if continuation["target_token_investigation"]:
        lines.append(
            f"- Cached tier consensus: {continuation['cached_tier_consensus']}; Device, Host, "
            "and SSD token IDs are compared independently of the cold oracle."
        )
        lines.append(f"- Localization: {continuation['localization']}")
    for finding in continuation["target_token_investigation"]:
        lines.append(
            f"- {finding['verdict']}: {finding['cold_label']} versus "
            f"{finding['subject_label']} first differs at generated index {finding['index']}: "
            f"cold token {finding['cold_token_id']}, cached token "
            f"{finding['subject_token_id']}; root cause: {finding['root_cause']}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reanalyze_evidence(
    path: Path, calibration_path: Path, validation_path: Path | None = None
) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read replay evidence {path}: {error}") from error
    evidence_version = evidence.get("schema_version") if isinstance(evidence, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("artifact_type") != EVIDENCE_TYPE
        or evidence_version not in {2, 3, EVIDENCE_VERSION}
    ):
        raise ReplayError("unsupported replay evidence identity")
    measurements = evidence.get("measurements")
    if not isinstance(measurements, dict):
        raise ReplayError("replay evidence has no raw measurements")
    recorded_threshold = evidence.get("threshold")
    if not isinstance(recorded_threshold, dict):
        raise ReplayError("replay evidence has no frozen threshold")
    predecessor_calibration = load_predecessor_calibration(calibration_path)
    threshold = freeze_material_improvement_threshold(
        [
            [row["external_ttft_ms"] for row in measurements["cold"]],
            [row["external_ttft_ms"] for row in measurements["existing"]],
        ],
        predecessor_calibration,
    )
    require_unchanged_frozen_threshold(recorded_threshold, threshold)
    continuation = validate_measured_continuations(measurements)
    comparisons = {
        profile: compare_profile(measurements["existing"], measurements[profile], threshold)
        for profile in ("device", "host", "ssd")
    }
    embedded_cases = evidence.get("validation_cases")
    if validation_path is not None:
        validation_cases = load_validation_cases(validation_path, None)
        validation_document = json.loads(validation_path.read_text(encoding="utf-8"))
        measured_revision = str(evidence["build"]["source_commit"])
        validation_revision = str(validation_document.get("source_revision", ""))
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", measured_revision, validation_revision],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if ancestry.returncode != 0:
            raise ReplayError(
                "replacement validation evidence is not from a descendant of the measured build"
            )
        evidence["validation_evidence_identity"] = {
            "path": str(validation_path),
            "sha256": sha256_file(validation_path),
            "source_revision": validation_revision,
            "serve_sha256": validation_document.get("serve_sha256"),
            "relationship_to_measured_build": "descendant",
        }
        evidence["validation_evidence_path"] = str(validation_path)
    elif isinstance(embedded_cases, dict):
        validation_cases = embedded_cases
    else:
        validation_path_value = evidence.get("validation_evidence_path")
        validation_path = (
            Path(validation_path_value) if isinstance(validation_path_value, str) else None
        )
        validation_cases = load_validation_cases(
            validation_path, str(evidence["build"]["serve_sha256"])
        )
    expected_boundaries = int(evidence["fixture"]["protocol_cases"])
    regression_matrix = build_regression_matrix(
        continuation,
        validation_cases,
        boundary_replay_status(measurements["boundary"], expected_boundaries),
    )
    evidence["comparisons"] = comparisons
    evidence["threshold"] = threshold
    evidence["continuation_validation"] = continuation
    evidence["schema_version"] = EVIDENCE_VERSION
    evidence.pop("regression_matrix", None)
    evidence["required_regression_matrix"] = regression_matrix
    evidence["supplemental_evidence"] = build_supplemental_evidence(validation_cases)
    evidence["validation_cases"] = validation_cases
    methodology = evidence.get("methodology")
    if isinstance(methodology, dict):
        methodology["predecessor_calibration_incorporated_during_derived_reanalysis"] = True
        methodology["predecessor_calibration_changed_frozen_numeric_threshold"] = False
    evidence["verdict"] = campaign_verdict(comparisons, regression_matrix)
    evidence["configuration_decision"] = configuration_decision(
        comparisons, evidence["verdict"]["correctness"]
    )
    write_json(path, evidence)
    write_json(path.with_name("threshold.json"), threshold)
    write_report(path.with_name("report.md"), evidence)
    return evidence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate fixtures without a GPU run")
    parser.add_argument(
        "--reanalyze-evidence",
        type=Path,
        help="recompute derived verdicts/report from an existing schema-v2/v3 evidence JSON",
    )
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--predecessor-calibration",
        type=Path,
        default=DEFAULT_PREDECESSOR_CALIBRATION,
        help="repository-local predecessor calibration used to freeze materiality",
    )
    parser.add_argument("--serve", type=Path, default=DEFAULT_SERVE)
    parser.add_argument(
        "--baseline-serve",
        type=Path,
        help="ninfer-serve built from the recorded baseline revision",
    )
    parser.add_argument(
        "--baseline-build-identity",
        type=Path,
        help="hash-pinned build identity emitted for --baseline-serve",
    )
    parser.add_argument(
        "--validation-evidence",
        type=Path,
        help=(
            "optional executable-identified focused/canonical validation results; when "
            "reanalyzing, must be from a descendant of the measured build"
        ),
    )
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
    args.predecessor_calibration = args.predecessor_calibration.resolve()
    if args.validation_evidence is not None:
        args.validation_evidence = args.validation_evidence.resolve()
    if args.reanalyze_evidence is not None:
        evidence = reanalyze_evidence(
            args.reanalyze_evidence.resolve(),
            args.predecessor_calibration,
            args.validation_evidence,
        )
        print(args.reanalyze_evidence.resolve().parent)
        return campaign_exit_status(evidence["verdict"])
    args.manifest = args.manifest.resolve()
    check = validate_manifest(args.manifest, verify_source=args.verify_source)
    predecessor_calibration = load_predecessor_calibration(args.predecessor_calibration)
    check["predecessor_calibration_sha256"] = predecessor_calibration["sha256"]
    if args.check:
        print(json.dumps(check, indent=2, sort_keys=True))
        return 0
    args.serve = args.serve.resolve()
    if args.baseline_serve is None or args.baseline_build_identity is None:
        raise ReplayError(
            "measurement requires --baseline-serve and --baseline-build-identity; the current "
            "binary cannot stand in for the recorded revision"
        )
    args.baseline_serve = args.baseline_serve.resolve()
    args.baseline_build_identity = args.baseline_build_identity.resolve()
    args.weights = args.weights.resolve()
    if not args.serve.is_file() or not os.access(args.serve, os.X_OK):
        raise ReplayError(f"ninfer-serve executable is unavailable: {args.serve}")
    if not args.baseline_serve.is_file() or not os.access(args.baseline_serve, os.X_OK):
        raise ReplayError(f"baseline ninfer-serve executable is unavailable: {args.baseline_serve}")
    if not args.weights.is_file():
        raise ReplayError(f"authorized model artifact is unavailable: {args.weights}")
    baseline = load_baseline_identity(args.baseline_build_identity, args.baseline_serve)
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
        ],
        predecessor_calibration,
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
    continuation = validate_measured_continuations(measurements)

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
    current_serve_sha256 = sha256_file(args.serve)
    validation_cases = load_validation_cases(args.validation_evidence, current_serve_sha256)
    boundary_status = boundary_replay_status(
        measurements["boundary"], len(boundary_fixtures(manifest))
    )
    regression_matrix = build_regression_matrix(
        continuation, validation_cases, boundary_status
    )
    verdict = campaign_verdict(comparisons, regression_matrix)
    decision = configuration_decision(comparisons, verdict["correctness"])
    evidence = {
        "artifact_type": EVIDENCE_TYPE,
        "schema_version": EVIDENCE_VERSION,
        "date": dt.date.today().isoformat(),
        "build": {
            "source_commit": source_commit,
            "source_dirty": dirty,
            "serve_path": str(args.serve),
            "serve_sha256": current_serve_sha256,
        },
        "baseline": baseline,
        "baseline_method": "separate hash-pinned executable built from the recorded revision",
        "artifact": {"path": str(args.weights), "bytes": args.weights.stat().st_size},
        "fixture": check,
        "methodology": {
            "samples": args.samples,
            "threshold_frozen_before_optimized_trials": True,
            "predecessor_calibration_consumed_before_optimized_trials": True,
            "raw_render_replay_is_protocol_separate": True,
            "profiles": [
                "cold-current",
                "existing-recorded-revision",
                "device",
                "host",
                "ssd",
                "boundary",
                "overlap",
            ],
        },
        "threshold": threshold,
        "measurements": measurements,
        "summaries": summaries,
        "profile_evidence": profiles,
        "comparisons": comparisons,
        "configuration_decision": decision,
        "continuation_validation": continuation,
        "validation_evidence_path": (
            str(args.validation_evidence) if args.validation_evidence is not None else None
        ),
        "validation_cases": validation_cases,
        "required_regression_matrix": regression_matrix,
        "supplemental_evidence": build_supplemental_evidence(validation_cases),
        "verdict": verdict,
    }
    write_json(output / "evidence.json", evidence)
    write_report(output / "report.md", evidence)
    print(output)
    return campaign_exit_status(verdict)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as error:
        raise SystemExit(f"tiered cache replay failed: {error}") from error
