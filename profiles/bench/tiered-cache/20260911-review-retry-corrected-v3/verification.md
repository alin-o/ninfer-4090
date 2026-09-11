# Tiered cache review-retry verification (2026-09-11)

## Canonical verification

```bash
NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python \
NINFER_VERIFY_EVAL_PYTHON=/opt/ninfer-venv/bin/python \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
  bash .agent/verify.sh
```

Result: PASS. Linux script checks passed; the artifact, converter, and benchmark-consumer suite
reported 91 passes; the evaluation coordinator reported 19 passes; the Release sm_89 build passed;
and CTest reported 105 tests with zero failures in 132.05 seconds.

The canonical invocation did not set a model-artifact variable, so six real-artifact target tests
were skipped. The sm_89-inapplicable NVFP4 A4 test was also skipped. These skips do not establish
real-artifact correctness.

## Current-binary real-artifact validation

`tools/bench/run_tiered_cache_validation.py` ran against the authorized
`/models/qwen3_8_27b.ninfer` and Serve SHA-256
`dcd9d987887031836d54967b24e95d2d039962005f238f4f5e12e356cd71ed49`. Its 13 executable-identified
cases all passed: the seven real prefix/cache transition scenarios, ResourceManager and durable
catalog tests, frontend boundary/token-lineage coverage, and the three public protocol boundary
suites. The identified case map is embedded in `evidence.json`.

Official-tokenizer fixture lineage remains UNVERIFIED because no user-authorized,
identity-pinned official tokenizer artifact was established locally. No artifact was downloaded.

## Production-profile replay

The schema-v2 replay completed all cold, recorded-revision existing-cache, Device, forced-Host,
post-restart SSD, boundary, and four-request overlap arms. It exited 3 because correctness failed,
not because the performance comparison failed.

- Performance: PASS against the separately built baseline revision
  `c7dca9acfef663acd10d4ec2817f184bc50547fe`, using a threshold frozen at 10.09% before optimized
  trials. The report contains the measured reductions; they are not promoted into a configuration
  recommendation because correctness failed.
- Concurrent correlation: PASS. All four overlap rows retain wire and internal request identities;
  external/server TTFT pairs differ by at most 0.315 ms.
- Exact continuation: FAIL. Six of nine cold-versus-tier comparisons differ. Device, Host, and SSD
  agree exactly with one another. Fixture 0 first differs at generated index 15 (4216 versus
  32480); fixture 1 first differs at index 10 (26388 versus 11); fixture 2 is exact.
- MTP counters: FAIL for fixture 1 in every tier. Relative to cold, cached execution records drafted
  -1, accepted +1, fallback -1, equal rounds, and accepted-per-position deltas `[1, 1, -1]`.
- Accounting: transfer time/count/page/byte and pressure/eviction values are interval sums;
  occupancy values are maxima; alias-aware reclaimed-capacity totals remain separate counters.

The preceding `20260911-review-retry-corrected-v2` raw logs independently reproduce the same cold
and Device token/counter sequences, but that attempt failed before report generation on a now-fixed
Host-counter post-processing typo. It is not used for the final timing comparisons.
