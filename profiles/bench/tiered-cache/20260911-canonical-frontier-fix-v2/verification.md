# Tiered cache canonical-frontier verification (2026-09-11)

## Canonical verification

```bash
NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python \
NINFER_VERIFY_EVAL_PYTHON=/opt/ninfer-venv/bin/python \
  bash .agent/verify.sh
```

Result: PASS. Linux script checks passed; the artifact, converter, and benchmark-consumer suite
reported 93 passes; the evaluation coordinator reported 19 passes; the Release sm_89 build passed;
and CTest reported 105 tests with zero failures in 139.97 seconds.

The canonical invocation did not set a model-artifact variable, so six real-artifact target tests
were skipped. The sm_89-inapplicable NVFP4 A4 test was also skipped. These skips do not establish
real-artifact correctness; the authorized Qwen3.8 workload was exercised separately below.

## Focused and real-artifact validation

The frontend, serve-options, gated-delta-net, causal-convolution, and replay-fold focused suites all
passed. The authorized `/models/qwen3_8_27b.ninfer` cache-equivalence scenario produced the same
deterministic token hash (`16493017374143550425`) for cache-disabled, existing-cache,
Device, and forced-Host execution. The shared-snapshot scenario also passed, covering complete
State/Main/MTP restart restoration and injected failure cleanup.

The preserved `tools/bench/run_tiered_cache_validation.py` run contains 14 executable-identified
cases at source revision `e42b22b7e841b675f39b017a2929f2b217f2cfb9` and Serve SHA-256
`02d3113da57f716d2ac633da3da79eefa7bf692b9d9463672966b0a8891762f8`. All 14 cases passed.
In particular, `host-restore` is pinned to prefix-test executable SHA-256
`ce7574c49ddd8f3f325076e60586fd98d03c10d4ae0868ff243b8ea6d4b344d5`; that scenario selects a
private `PrivateTurnClosure` owner and requires increasing Host-to-Device State, Main KV, and MTP KV
counters. The separate `shared-snapshot` case is used only for the shared SSD/rollback row. The
complete identified case map is embedded in `evidence.json`; no shared-only result substitutes for
private Host materialization.

Official-tokenizer fixture lineage remains UNVERIFIED because no user-authorized,
identity-pinned official tokenizer artifact was established locally. It is supplemental provenance,
not a target acceptance row. No artifact was downloaded.

The localized benchmark-verdict regressions passed (`20 passed` across
`tests/test_serve_corpus.py` and `tests/test_bench_matrix.py`). They establish that unavailable
supplemental tokenizer evidence does not affect target correctness, performance, configuration,
overall verdict, or exit status, while a required FAIL or UNVERIFIED row still blocks those claims.

## Production-profile replay and calibrated threshold

The schema-v4 derived analysis uses the existing executable-pinned replay, which completed three
trials for every cold, recorded-revision existing-cache,
Device, forced-Host, and post-restart SSD arm, plus the serving-boundary and four-request-overlap
matrices. No GPU campaign was rerun; raw measurements and the frozen threshold are unchanged.

The threshold now records and validates the predecessor calibration at commit
`69ef567748cc6b79604caeef64e4bb22309afdab` and report SHA-256
`1c84e30f145ec0d8df79115558ff980d4b6a35e6b49c6449595c290bdd4ea6e3`; the repository-local
calibration input SHA-256 is
`dcc8b6133be5e10c0420fa1e7c96cd185221edc14d26f649a0f2937b8571b17d`. Its exact private Host
payload is 307,908,608 State + 4,194,304 Main KV + 262,144 MTP KV bytes. Applying the predecessor's
selected H2D cost coefficients yields a 7.809 ms effect-size reference (0.214% of the
existing-cache median); three times the measured baseline relative MAD is 42.48% and remains the
dominant frozen threshold. Reanalysis refuses to proceed if incorporating calibration would change
the threshold after optimized trials, so this update does not make a post-hoc comparison.

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
