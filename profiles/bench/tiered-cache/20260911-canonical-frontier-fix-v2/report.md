# Tiered cache production-profile replay (2026-09-11)

This report is generated from the adjacent raw JSON. Percentages are emitted only for
the frozen material-improvement comparison; correctness-only rows make no speed claim.
Overall verdict: **UNVERIFIED**. Production acceptance is not established; inspect failed and unverified rows.

## Identity and capacity

- Source commit: `ea17cde537f5a803e69e224891c3f3afff8cc883`; dirty at measurement: `true`; serve SHA-256: `c566b543da13cb95e01400207d5fc74658eef49c86701e65b423504e858eea39`.
- Artifact: `/models/qwen3_8_27b.ninfer` (18210531328 bytes), target `qwen3_8_27b`, weights `groupwise-int`.
- GPU: NVIDIA GeForce RTX 4090 sm_89; ready free headroom 1317 MiB.
- Resolved KV: 233408 tokens / 3647 page groups (`auto`); State slots active+retained 8; Host KV 8589934592 bytes.
- Existing-cache baseline revision: `c7dca9acfef663acd10d4ec2817f184bc50547fe`; separate serve SHA-256: `fa4e2c22fa2729da0ff0cda0193036cd0ad88e20d6ab65a9674f3f8d4bd7cf79`; build identity: `/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-baseline-c7-review/baseline-build.json`.

## Frozen threshold

Method: max(10%, three times the largest baseline relative MAD)

Required reduction: 42.48%

## Profile results

| Profile | Trials | Queue ms | TTFT ms | Total ms | Evaluated | Reused | Tier(s) |
|---|---:|---:|---:|---:|---:|---:|---|
| cold | 3 | 0.497 | 3760.381 | 3928.182 | 6217–6217 | 0–0 | root |
| existing | 3 | 0.857 | 3644.484 | 3797.985 | 6217–6217 | 0–0 | root |
| device | 3 | 2.009 | 129.265 | 281.302 | 41–41 | 6176–6176 | device |
| host | 3 | 20.115 | 250.739 | 400.770 | 58–58 | 6159–6159 | host |
| ssd | 3 | 8.553 | 1286.403 | 1522.108 | 58–58 | 6159–6159 | ssd |
| boundary | 5 | 0.763 | 169.386 | 314.041 | 44–62 | 0–0 | root |
| overlap | 4 | 295.212 | 452.562 | 813.706 | 41–41 | 6176–6176 | memory |

Four-request overlap makespan: 1027.553 ms.
Cold durable-write seed: TTFT 3126.202 ms, total 3244.258 ms; it is excluded from loaded-hit comparisons.
MTP drafted/accepted totals are recorded per profile in `evidence.json`; acceptance is compared with the cold control and is not required to be nonzero per request.

## Material comparisons

- device: reduction 96.45%; material improvement: true.
- host: reduction 93.12%; material improvement: true.
- ssd: reduction 64.70%; material improvement: true.

The speed comparisons are retained as measurements, but correctness is UNVERIFIED; no tier or cost-preset/config decision is justified.

## Correctness matrix

The serving replay retains exact generated token IDs and joins concurrent wire/log
measurements by response identity. Low-level statuses require imported, executable-
identified validation evidence; absent or skipped evidence remains UNVERIFIED.

- PASS: live serving boundary protocols — 5/5 fixture shapes completed through public endpoints with 5 unique wire response identities
- PASS: unchanged prepared tokens and explicit/inferred boundary lineage — frontend-boundary-token-lineage=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/frontend-boundary-token-lineage.log); openai-chat-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/openai-chat-boundary.log); openai-responses-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/openai-responses-boundary.log); anthropic-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/anthropic-boundary.log)
- PASS: measured cold/Device/Host/SSD exact generated token IDs — 9 fixture-aligned comparisons; 0 token-ID mismatches
- PASS: measured cold/Device/Host/SSD MTP round/drafted/accepted/fallback counters — 9 fixture-aligned comparisons; 0 counter mismatches with per-field deltas and investigation records
- UNVERIFIED: official-tokenizer lineage for frontend boundary fixtures — missing identified validation evidence: official-tokenizer-lineage
- PASS: intermediate prefix, parent infeasible, and prefix-only Host restore — pressure-resume=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/pressure-resume.log)
- PASS: State/Main/MTP round-trip, restart SSD, cancellation, and corruption — shared-snapshot=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/shared-snapshot.log)
- PASS: four-request constrained pressure and independent progress — four-request-root-fallback=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/four-request-root-fallback.log)
- PASS: delayed/failed transfers, shutdown, aliases, heads, and reclamation — delayed-spill=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/delayed-spill.log); delayed-active-capture=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/delayed-active-capture.log); cuda-transfer-failure=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/cuda-transfer-failure.log); pending-snapshot-shutdown=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/pending-snapshot-shutdown.log); resource-manager=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/resource-manager.log); durable-shared-prefix-catalog=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-frontier-fix-v3-logs/durable-shared-prefix-catalog.log)

## Continuation investigation

Exact target token IDs and all MTP counters match every fixture-aligned cold control.
