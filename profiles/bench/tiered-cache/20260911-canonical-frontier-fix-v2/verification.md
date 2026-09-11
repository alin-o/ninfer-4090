# Tiered cache canonical-frontier verification (2026-09-11)

## Canonical verification

```bash
NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python \
NINFER_VERIFY_EVAL_PYTHON=/opt/ninfer-venv/bin/python \
  bash .agent/verify.sh
```

Result: PASS. Linux script checks passed; the artifact, converter, and benchmark-consumer suite
reported 93 passes; the evaluation coordinator reported 19 passes; the Release sm_89 build passed;
and CTest reported 105 tests with zero failures in 142.00 seconds.

The canonical invocation did not set a model-artifact variable, so six real-artifact target tests
were skipped. The sm_89-inapplicable NVFP4 A4 test was also skipped. These skips do not establish
real-artifact correctness; the authorized Qwen3.8 workload was exercised separately below.

## Focused and real-artifact validation

The frontend, serve-options, gated-delta-net, causal-convolution, and replay-fold focused suites all
passed. The authorized `/models/qwen3_8_27b.ninfer` cache-equivalence scenario produced the same
deterministic token hash (`16493017374143550425`) for cache-disabled, existing-cache,
Device, and forced-Host execution. The shared-snapshot scenario also passed, covering complete
State/Main/MTP restart restoration and injected failure cleanup.

`tools/bench/run_tiered_cache_validation.py` ran 13 executable-identified cases against Serve
SHA-256 `c566b543da13cb95e01400207d5fc74658eef49c86701e65b423504e858eea39` and the authorized
artifact. All 13 cases passed: the real cache-transition scenarios, ResourceManager and durable
catalog tests, frontend boundary/token-lineage coverage, and the three public protocol boundary
suites. The identified case map is embedded in `evidence.json`.

Official-tokenizer fixture lineage remains UNVERIFIED because no user-authorized,
identity-pinned official tokenizer artifact was established locally. It is supplemental provenance,
not a target acceptance row. No artifact was downloaded.

The localized benchmark-verdict regressions passed (`20 passed` across
`tests/test_serve_corpus.py` and `tests/test_bench_matrix.py`). They establish that unavailable
supplemental tokenizer evidence does not affect target correctness, performance, configuration,
overall verdict, or exit status, while a required FAIL or UNVERIFIED row still blocks those claims.

## Production-profile replay

The schema-v3 derived analysis uses the existing executable-pinned replay, which completed three
trials for every cold, recorded-revision existing-cache,
Device, forced-Host, and post-restart SSD arm, plus the serving-boundary and four-request-overlap
matrices. No GPU campaign was rerun; raw measurements and the frozen threshold are unchanged.

- Exact continuation: PASS. All 9 cold-versus-tier generated-token comparisons match exactly.
- MTP counters: PASS. All 9 round/drafted/accepted/fallback and per-position comparisons match.
- Performance: PASS. The threshold was frozen at 42.48% before optimized arms. Device, Host, and
  SSD reduced median TTFT by 96.45%, 93.12%, and 64.70%, respectively.
- Required matrix and overall: PASS. All nine target acceptance rows pass. Official-tokenizer
  lineage remains separately UNVERIFIED under supplemental evidence.
- Configuration: PASS with Device, Host, and SSD retained. No cost-preset/config change is
  justified because every measured tier exceeded the threshold and none was slower than the
  recorded-revision existing-cache baseline.

The adjacent `evidence.json`, `threshold.json`, and generated `report.md` retain build, artifact,
GPU, resolved-capacity, quota, trial, timing, transfer, pressure, occupancy, and exact-token data.
