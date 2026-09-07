---
trigger: model_decision
description: Use when implementing, testing, or reviewing NInfer changes, running verification, or diagnosing sandbox test prerequisites.
---

# Canonical verification

From the repository root run `bash .agent/verify.sh`. The command runs existing
Linux launcher checks, native Python artifact/converter and benchmark-consumer
suites, evaluation coordinator unittests, then a Release sm_89 application/test
build and CTest. Independent suites continue after failures; the command exits
nonzero if any step fails. No dependencies are installed and no persistent
server is started. There is no repository CI workflow or browser UI.

Prerequisites: Bash, CMake ≥3.28, Ninja, C++20 compiler, CUDA ≥12.8, FFmpeg dev
libraries and libcurl meeting CMake's constraints. Python needs pytest, torch,
NumPy and converter dependencies; evaluation tests need PyYAML and Rich (see
`eval/requirements.txt` for the separate evaluation environment). The sandbox
Python environment is not evidence that project test dependencies are present.

Use `NINFER_VERIFY_PYTHON` and `NINFER_VERIFY_EVAL_PYTHON` to select provisioned
interpreters. `NINFER_VERIFY_BUILD_DIR` defaults to ignored `build-agent-verify`;
`CMAKE_BUILD_PARALLEL_LEVEL` defaults to 2 for this command. Do not reuse a CMake
cache whose `CMAKE_HOME_DIRECTORY` names another checkout. No worktree setup or
dependency provisioning hook currently exists.

Coding runs focused checks while iterating and the canonical command before
completion. Independent Testing runs the canonical command and applicable
regressions. With a Testing stage, Review assesses the diff and evidence without
rerunning validation. Distinguish baseline failures, regressions, and skips;
never weaken the checks to obtain a pass.

Focused commands are documented in `tests/README.md`: build the affected test
target, then `ctest --test-dir build-agent-verify -R '<test-name>'
--output-on-failure`. Use `NINFER_OP_REPORT_STATS=1` with verbose CTest for
numerical error evidence. The initial two-job native build took about eight
minutes in this sandbox; CTest took about 14 seconds. Python suite runtime is
unmeasured because dependencies were missing. Do not treat this as a fast lint
gate.

GPU/device and real-artifact tests may skip with return code 77. Real engine
checks need the corresponding `NINFER_QWEN3_6_27B_WEIGHTS` or
`NINFER_QWEN3_6_35B_A3B_WEIGHTS` environment and supported hardware; skipped tests
do not establish GPU correctness. Resident-server smoke tests, model downloads,
performance benchmarks and external evaluation services are outside this gate.

Baseline observed during setup on 2026-09-07:

- CMake configure passes with CUDA 13.3.73 and required FFmpeg/libcurl libraries.
- The Release application/test build passes. CTest reports 30 passes, 65
  failures and eight skips out of 103 tests. Failed GPU checks report
  `cudaErrorMemoryAllocation: out of memory` at `cudaSetDevice`/`cudaMalloc`;
  no GPU correctness claim can be made from this run. Do not stop other GPU
  workloads to make verification pass.
- Default Python is `/opt/carapa-harness/venv/bin/python3`; pytest is absent.
  Evaluation unittest discovery reports four import errors for Rich/PyYAML.
- `scripts/check-linux-scripts.sh` fails with `Missing executable:` for its
  generated temporary `ninfer-serve` fixture. This precedes setup changes;
  its underlying environment/launcher cause has not been established.

Refresh these baseline notes when new evidence resolves them.
