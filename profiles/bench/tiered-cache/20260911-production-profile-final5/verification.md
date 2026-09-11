# Tiered cache verification (2026-09-11)

## Canonical verification

```bash
export NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer
bash .agent/verify.sh
```

Result: PASS. Linux launcher checks passed; artifact/converter and benchmark-consumer pytest
reported 80 passes; evaluation coordinator reported 19 passes; the Release sm_89 build passed;
CTest reported 105/105 passed with zero failures in 328.90 seconds. The real Qwen3.8 prefix test
ran against the authorized artifact and passed in 191.02 seconds.

Expected skips: Qwen3.6 score/load-plan paths without their separate artifacts, Qwen3.6-35B and
DFlash paths without their artifact, and NVFP4 A4 because sm_89 does not support that path.

## Focused verification

The following real-artifact scenarios passed:

- `cache-fixture-equivalence`
- `target-cache-calibration`
- `host-restore`
- `pressure-resume`
- `shared-snapshot`
- `shared-snapshot-export-allocation`
- `four-request-root-fallback`
- `capture-after-prefix-fork`
- `delayed-spill`
- `snapshot-quota-rejection`
- `delayed-active-capture`
- `delayed-active-capture-pressure`
- `pending-snapshot-shutdown`
- `cuda-transfer-failure`

The injected CUDA failure emitted the expected `cudaErrorLaunchFailure` cleanup diagnostics and
returned `ok`; this was a fault-injection pass, not an unexplained GPU failure.

Additional checks:

```bash
python tools/bench/run_tiered_cache_replay.py --check --verify-source
python -m pytest tests/test_serve_corpus.py -q
clang-format --dry-run --Werror \
  src/targets/qwen3_6/impl/runtime/program_impl.h \
  tests/targets/qwen3_6_27b/test_engine_prefix_real.cpp
git diff --check
```

Results: source identities verified with five protocol cases; 6 pytest cases passed; formatting
and whitespace checks passed.
