# Tiered cache production-profile replay (2026-09-11)

This report is generated from the adjacent raw JSON. Percentages are emitted only for
the frozen material-improvement comparison; correctness-only rows make no speed claim.
Overall verdict: **PASS**. Target production acceptance and the material-improvement claim are established by the required evidence; supplemental evidence is reported separately.

## Identity and capacity

- Source commit: `ea17cde537f5a803e69e224891c3f3afff8cc883`; dirty at measurement: `true`; serve SHA-256: `c566b543da13cb95e01400207d5fc74658eef49c86701e65b423504e858eea39`.
- Artifact: `/models/qwen3_8_27b.ninfer` (18210531328 bytes), target `qwen3_8_27b`, weights `groupwise-int`.
- GPU: NVIDIA GeForce RTX 4090 sm_89; ready free headroom 1317 MiB.
- Resolved KV: 233408 tokens / 3647 page groups (`auto`); State slots active+retained 8; Host KV 8589934592 bytes.
- Existing-cache baseline revision: `c7dca9acfef663acd10d4ec2817f184bc50547fe`; separate serve SHA-256: `fa4e2c22fa2729da0ff0cda0193036cd0ad88e20d6ab65a9674f3f8d4bd7cf79`; build identity: `/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-baseline-c7-review/baseline-build.json`.
- Required validation revision: `e42b22b7e841b675f39b017a2929f2b217f2cfb9` (descendant of the measured build); serve SHA-256 `02d3113da57f716d2ac633da3da79eefa7bf692b9d9463672966b0a8891762f8`; evidence SHA-256 `8573d91e52c92127b342efa2d9d7b60b3291d5aeb530c6a6e88786d7a603f234`.

## Frozen threshold

Method: max(predecessor complete-private Host H2D effect size divided by the existing-cache median, three times the largest baseline relative MAD)

Predecessor calibration: `docs/maintainer/2026-09-10-qwen3.8-27b-context-cache-calibration.md` at `69ef567748cc6b79604caeef64e4bb22309afdab`; complete-private Host H2D effect-size reference 7.809 ms (0.214% of the existing-cache median).
The predecessor supplied exact State/Main/MTP bytes and its selected cost-model coefficients; this effect-size floor is not reported as measured latency.

Required reduction: 42.48%

## Profile results

| Profile | Trials | Queue ms | TTFT ms | Total ms | Evaluated | Reused | Tier(s) |
|---|---:|---:|---:|---:|---:|---:|---|
| boundary | 5 | 0.763 | 169.386 | 314.041 | 44–62 | 0–0 | root |
| cold | 3 | 0.497 | 3760.381 | 3928.182 | 6217–6217 | 0–0 | root |
| device | 3 | 2.009 | 129.265 | 281.302 | 41–41 | 6176–6176 | device |
| existing | 3 | 0.857 | 3644.484 | 3797.985 | 6217–6217 | 0–0 | root |
| host | 3 | 20.115 | 250.739 | 400.770 | 58–58 | 6159–6159 | host |
| overlap | 4 | 295.212 | 452.562 | 813.706 | 41–41 | 6176–6176 | memory |
| ssd | 3 | 8.553 | 1286.403 | 1522.108 | 58–58 | 6159–6159 | ssd |

Four-request overlap makespan: 1027.553 ms.
Cold durable-write seed: TTFT 3126.202 ms, total 3244.258 ms; it is excluded from loaded-hit comparisons.
MTP drafted/accepted totals are recorded per profile in `evidence.json`; acceptance is compared with the cold control and is not required to be nonzero per request.

## Material comparisons

- device: reduction 96.45%; material improvement: true.
- host: reduction 93.12%; material improvement: true.
- ssd: reduction 64.70%; material improvement: true.

Every measured tier exceeded the frozen threshold; no tier was slower than the recorded-revision existing-cache baseline, so these measurements do not justify a cost-preset/config change.

## Required correctness matrix

The serving replay retains exact generated token IDs and joins concurrent wire/log
measurements by response identity. Low-level statuses require imported, executable-
identified validation evidence; absent or skipped evidence remains UNVERIFIED.

- PASS: live serving boundary protocols — 5/5 fixture shapes completed through public endpoints with 5 unique wire response identities
- PASS: unchanged prepared tokens and explicit/inferred boundary lineage — frontend-boundary-token-lineage=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/frontend-boundary-token-lineage.log; executable sha256=6aca8302c032d9427526eff00818f558a51fdae6f1c8d1414aa0e9dc2dfe07b6); openai-chat-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/openai-chat-boundary.log; executable sha256=bdd57e243e26652c518133eae442910e13a11c57ce5d6cb467c2f0ede5cb4597); openai-responses-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/openai-responses-boundary.log; executable sha256=caeb113ba396d7dca3af40e9f25a961f677030f9b65ed90f85352a3136aaa8a8); anthropic-boundary=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/anthropic-boundary.log; executable sha256=463304f8c9db7b753baa45d22b3511eaedb3d5481756be8122bb4e8e46f8cf4b)
- PASS: measured cold/Device/Host/SSD exact generated token IDs — 9 fixture-aligned comparisons; 0 token-ID mismatches
- PASS: measured cold/Device/Host/SSD MTP round/drafted/accepted/fallback counters — 9 fixture-aligned comparisons; 0 counter mismatches with per-field deltas and investigation records
- PASS: intermediate prefix, parent infeasible, and prefix-only Host restore — pressure-resume=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/pressure-resume.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5)
- PASS: complete private State/Main/MTP Host materialization — host-restore=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/host-restore.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5)
- PASS: State/Main/MTP round-trip, restart SSD, cancellation, and corruption — shared-snapshot=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/shared-snapshot.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5)
- PASS: four-request constrained pressure and independent progress — four-request-root-fallback=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/four-request-root-fallback.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5)
- PASS: delayed/failed transfers, shutdown, aliases, heads, and reclamation — delayed-spill=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/delayed-spill.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5); delayed-active-capture=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/delayed-active-capture.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5); cuda-transfer-failure=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/cuda-transfer-failure.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5); pending-snapshot-shutdown=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/pending-snapshot-shutdown.log; executable sha256=ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5); resource-manager=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/resource-manager.log; executable sha256=1258f02ae652a3079720abae2246e352c659cf795909c9d0b37adae21aa28c75); durable-shared-prefix-catalog=PASS (/workspace/ninfer/.worktrees/validate-tiered-cache-correctness-and-speedups-at-concurrency-four-lyhafi/.local/tiered-cache-validation-testing-logs/durable-shared-prefix-catalog.log; executable sha256=2e8ba5fd026d8a86909f4dbe349ba0048289d6e4e9fd5de79e6afb700ce2d765)

## Supplemental evidence

Supplemental rows record additional provenance and do not gate target correctness,
the material-improvement claim, configuration decisions, the overall verdict, or
the replay exit status.

- UNVERIFIED: official-tokenizer lineage for frontend boundary fixtures — missing identified validation evidence: official-tokenizer-lineage

## Continuation investigation

Exact target token IDs and all MTP counters match every fixture-aligned cold control.
