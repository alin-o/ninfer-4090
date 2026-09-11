# Tiered cache production-profile replay (2026-09-11)

This report is generated from the adjacent raw JSON. Percentages are emitted only for
the frozen material-improvement comparison; correctness-only rows make no speed claim.

## Identity and capacity

- Source commit: `ea49c80cc5b5e90bd24252a3ed8584996468d258`; dirty at measurement: `true`; serve SHA-256: `14b3cc02033a8dd3f85e40ddb1b356e4876aa6bad0484f73d1b3c83a663eb54b`.
- Artifact: `/models/qwen3_8_27b.ninfer` (18210531328 bytes), target `qwen3_8_27b`, weights `groupwise-int`.
- GPU: NVIDIA GeForce RTX 4090 sm_89; ready free headroom 1319 MiB.
- Resolved KV: 233408 tokens / 3647 page groups (`auto`); State slots active+retained 8; Host KV 8589934592 bytes.
- Existing-cache baseline uses the measured binary with shared prefixes and automatic long anchors disabled; its recorded behavior baseline is `c7dca9acfef663acd10d4ec2817f184bc50547fe`.

## Frozen threshold

Method: max(10%, three times the largest baseline relative MAD)

Required reduction: 10.00%

## Profile results

| Profile | Trials | Queue ms | TTFT ms | Total ms | Evaluated | Reused | Tier(s) |
|---|---:|---:|---:|---:|---:|---:|---|
| cold | 3 | 0.513 | 2954.853 | 3096.137 | 6217–6217 | 0–0 | root |
| existing | 3 | 0.972 | 3042.955 | 3200.137 | 6217–6217 | 0–0 | root |
| device | 3 | 1.971 | 127.979 | 257.213 | 41–41 | 6176–6176 | device |
| host | 3 | 20.054 | 211.525 | 335.397 | 58–58 | 6159–6159 | host |
| ssd | 3 | 8.343 | 944.525 | 1085.884 | 58–58 | 6159–6159 | ssd |
| boundary | 5 | 0.509 | 168.018 | 307.633 | 44–62 | 0–0 | root |
| overlap | 4 | 293.875 | 447.690 | 774.962 | 41–41 | 6176–6176 | memory |

Four-request overlap makespan: 952.933 ms.
Cold durable-write seed: TTFT 3067.988 ms, total 3186.120 ms; it is excluded from loaded-hit comparisons.
MTP drafted/accepted totals are recorded per profile in `evidence.json`; acceptance is compared with the cold control and is not required to be nonzero per request.

## Material comparisons

- device: reduction 95.79%; material improvement: true.
- host: reduction 93.05%; material improvement: true.
- ssd: reduction 68.96%; material improvement: true.

All measured Device, Host, and SSD arms exceeded the frozen threshold. No measured
tier/depth choice was slower than ready-source prefill, so this campaign does not
justify disabling a tier or changing a cost-preset/config default.

## Correctness matrix

Exact token IDs and State/Main/MTP bytes are validated by the named CTest scenarios;
the serving replay records HTTP semantics and timings and does not infer token IDs
from output strings.

- PASS: serving boundary protocols and unchanged prepared tokens — boundary replay plus frontend and serving schema CTests
- PASS: cold/Device/Host exact token IDs and MTP counters — NINFER_PREFIX_REAL_SCENARIO=cache-fixture-equivalence
- PASS: intermediate prefix, parent infeasible, and prefix-only Host restore — NINFER_PREFIX_REAL_SCENARIO=pressure-resume: 121-page parent, 120-page frontier, four restored Main-KV pages
- PASS: State/Main/MTP round-trip, restart SSD, cancellation, and corruption — NINFER_PREFIX_REAL_SCENARIO=shared-snapshot
- PASS: four-request constrained pressure and independent progress — NINFER_PREFIX_REAL_SCENARIO=four-request-root-fallback plus overlap replay
- PASS: delayed/failed transfers, shutdown, aliases, heads, and reclamation — delayed-spill, delayed-active-capture, cuda-transfer-failure and pending-snapshot-shutdown scenarios plus resource/catalog CTests
