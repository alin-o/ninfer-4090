# Tiered cache production-profile replay (2026-09-11)

This report is generated from the adjacent raw JSON. Percentages are emitted only for
the frozen material-improvement comparison; correctness-only rows make no speed claim.
Overall verdict: **FAIL**. Production acceptance is not established; inspect failed and unverified rows.

## Identity and capacity

- Source commit: `188a7c23d3a0c0eff89827b30122c9330b453728`; dirty at measurement: `true`; serve SHA-256: `dcd9d987887031836d54967b24e95d2d039962005f238f4f5e12e356cd71ed49`.
- Artifact: `/models/qwen3_8_27b.ninfer` (18210531328 bytes), target `qwen3_8_27b`, weights `groupwise-int`.
- GPU: NVIDIA GeForce RTX 4090 sm_89; ready free headroom 1319 MiB.
- Resolved KV: 233408 tokens / 3647 page groups (`auto`); State slots active+retained 8; Host KV 8589934592 bytes.
- Existing-cache baseline revision: `c7dca9acfef663acd10d4ec2817f184bc50547fe`; separate serve SHA-256: `fa4e2c22fa2729da0ff0cda0193036cd0ad88e20d6ab65a9674f3f8d4bd7cf79`; build identity: `/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-baseline-c7-review/baseline-build.json`.

## Frozen threshold

Method: max(10%, three times the largest baseline relative MAD)

Required reduction: 10.09%

## Profile results

| Profile | Trials | Queue ms | TTFT ms | Total ms | Evaluated | Reused | Tier(s) |
|---|---:|---:|---:|---:|---:|---:|---|
| boundary | 5 | 0.468 | 167.650 | 306.806 | 44–62 | 0–0 | root |
| cold | 3 | 0.476 | 3003.745 | 3164.302 | 6217–6217 | 0–0 | root |
| device | 3 | 1.995 | 129.260 | 257.604 | 41–41 | 6176–6176 | device |
| existing | 3 | 1.170 | 3252.888 | 3396.951 | 6217–6217 | 0–0 | root |
| host | 3 | 19.916 | 211.746 | 332.369 | 58–58 | 6159–6159 | host |
| overlap | 4 | 291.119 | 441.734 | 766.974 | 41–41 | 6176–6176 | memory |
| ssd | 3 | 8.299 | 925.317 | 1065.424 | 58–58 | 6159–6159 | ssd |

Four-request overlap makespan: 946.727 ms.
Cold durable-write seed: TTFT 3076.738 ms, total 3194.772 ms; it is excluded from loaded-hit comparisons.
MTP drafted/accepted totals are recorded per profile in `evidence.json`; acceptance is compared with the cold control and is not required to be nonzero per request.

## Material comparisons

- device: reduction 96.03%; material improvement: true.
- host: reduction 93.49%; material improvement: true.
- ssd: reduction 71.55%; material improvement: true.

The speed comparisons are retained as measurements, but correctness is FAIL; no tier or cost-preset/config decision is justified.

## Correctness matrix

The serving replay retains exact generated token IDs and joins concurrent wire/log
measurements by response identity. Low-level statuses require imported, executable-
identified validation evidence; absent or skipped evidence remains UNVERIFIED.

- PASS: live serving boundary protocols — 5/5 fixture shapes completed through public endpoints with 5 unique wire response identities
- PASS: unchanged prepared tokens and explicit/inferred boundary lineage — frontend-boundary-token-lineage=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/frontend-boundary-token-lineage.log); openai-chat-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/openai-chat-boundary.log); openai-responses-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/openai-responses-boundary.log); anthropic-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/anthropic-boundary.log)
- FAIL: measured cold/Device/Host/SSD exact generated token IDs — 9 fixture-aligned comparisons; 6 token-ID mismatches
- FAIL: measured cold/Device/Host/SSD MTP round/drafted/accepted/fallback counters — 9 fixture-aligned comparisons; 3 counter mismatches with per-field deltas and investigation records
- UNVERIFIED: official-tokenizer lineage for frontend boundary fixtures — missing identified validation evidence: official-tokenizer-lineage
- PASS: intermediate prefix, parent infeasible, and prefix-only Host restore — pressure-resume=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/pressure-resume.log)
- PASS: State/Main/MTP round-trip, restart SSD, cancellation, and corruption — shared-snapshot=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/shared-snapshot.log)
- PASS: four-request constrained pressure and independent progress — four-request-root-fallback=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/four-request-root-fallback.log)
- PASS: delayed/failed transfers, shutdown, aliases, heads, and reclamation — delayed-spill=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/delayed-spill.log); delayed-active-capture=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/delayed-active-capture.log); cuda-transfer-failure=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/cuda-transfer-failure.log); pending-snapshot-shutdown=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/pending-snapshot-shutdown.log); resource-manager=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/resource-manager.log); durable-shared-prefix-catalog=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-review-v2-logs/durable-shared-prefix-catalog.log)

## Continuation investigation

The replay does not establish full continuation equivalence; inspect the recorded mismatch rows before making a production-correctness claim.
- FAIL: cold-1 versus device-1 — counter deltas {"accepted_tokens": 1, "draft_window": 0, "drafted_tokens": -1, "fallback_steps": -1, "rounds": 0}; accepted-per-position deltas [1, 1, -1]; root cause: not established by the serving replay.
- FAIL: cold-1 versus host-1 — counter deltas {"accepted_tokens": 1, "draft_window": 0, "drafted_tokens": -1, "fallback_steps": -1, "rounds": 0}; accepted-per-position deltas [1, 1, -1]; root cause: not established by the serving replay.
- FAIL: cold-1 versus ssd-restart-1 — counter deltas {"accepted_tokens": 1, "draft_window": 0, "drafted_tokens": -1, "fallback_steps": -1, "rounds": 0}; accepted-per-position deltas [1, 1, -1]; root cause: not established by the serving replay.
- Cached tier consensus: PASS; Device, Host, and SSD token IDs are compared independently of the cold oracle.
- Localization: Device, Host, and post-restart SSD continuations agree exactly with one another. The measured divergence is therefore between cache-disabled Root prefill and the cache-participating prefill/capture schedule, not between retention tiers; this does not establish the numerical root cause.
- FAIL: cold-0 versus device-0 first differs at generated index 15: cold token 4216, cached token 32480; root cause: not established by the serving replay.
- FAIL: cold-1 versus device-1 first differs at generated index 10: cold token 26388, cached token 11; root cause: not established by the serving replay.
- FAIL: cold-0 versus host-0 first differs at generated index 15: cold token 4216, cached token 32480; root cause: not established by the serving replay.
- FAIL: cold-1 versus host-1 first differs at generated index 10: cold token 26388, cached token 11; root cause: not established by the serving replay.
- FAIL: cold-0 versus ssd-restart-0 first differs at generated index 15: cold token 4216, cached token 32480; root cause: not established by the serving replay.
- FAIL: cold-1 versus ssd-restart-1 first differs at generated index 10: cold token 26388, cached token 11; root cause: not established by the serving replay.
