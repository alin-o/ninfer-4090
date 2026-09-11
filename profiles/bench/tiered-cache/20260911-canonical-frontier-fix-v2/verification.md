# Tiered cache canonical-frontier verification (2026-09-11)

## Canonical verification

```bash
NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python \
NINFER_VERIFY_EVAL_PYTHON=/opt/ninfer-venv/bin/python \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  bash .agent/verify.sh
```

Result: PASS. Linux script checks passed; the artifact, converter, and benchmark-consumer suite
reported 91 passes; the evaluation coordinator reported 19 passes; the Release sm_89 build passed;
and CTest reported 105 tests with zero failures in 155.93 seconds.

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
identity-pinned official tokenizer artifact was established locally. No artifact was downloaded.

## Production-profile replay

The schema-v2 replay completed three trials for every cold, recorded-revision existing-cache,
Device, forced-Host, and post-restart SSD arm, plus the serving-boundary and four-request-overlap
matrices.

- Exact continuation: PASS. All 9 cold-versus-tier generated-token comparisons match exactly.
- MTP counters: PASS. All 9 round/drafted/accepted/fallback and per-position comparisons match.
- Performance: PASS. The threshold was frozen at 42.48% before optimized arms. Device, Host, and
  SSD reduced median TTFT by 96.45%, 93.12%, and 64.70%, respectively.
- Overall: UNVERIFIED solely because official-tokenizer lineage remains unavailable. No
  cost-preset or configuration change is justified from an unverified production result.

The adjacent `evidence.json`, `threshold.json`, and generated `report.md` retain build, artifact,
GPU, resolved-capacity, quota, trial, timing, transfer, pressure, occupancy, and exact-token data.
