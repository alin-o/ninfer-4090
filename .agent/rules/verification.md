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
cache whose `CMAKE_HOME_DIRECTORY` names another checkout. CarapaBox provisions
the task worktree and shared compiler cache deterministically before the agent
runs. Task agents use the prepared build; stage prompts must not instruct them
to invoke or manage worktree lifecycle hooks. Python dependencies come from the
provisioned container environment, not hook-installed packages. Verification
sets `TMPDIR` to the checkout's `.local/test-tmp` because `/tmp` is noexec.

Coding runs focused checks while iterating and the canonical command before
completion. Independent Testing runs the canonical command and applicable
regressions. With a Testing stage, Review assesses the diff and evidence without
rerunning validation. Distinguish baseline failures, regressions, and skips;
never weaken the checks to obtain a pass.

Focused commands are documented in `tests/README.md`: build the affected test
target, then `ctest --test-dir build-agent-verify -R '<test-name>'
--output-on-failure`. Use `NINFER_OP_REPORT_STATS=1` with verbose CTest for
numerical error evidence. The initial two-job native build took about eight
minutes in this sandbox; CTest with the GPU free took about 196 seconds. The
Python artifact/converter suites took about three seconds. Do not treat the
full command as a fast lint gate.

GPU/device and real-artifact tests may skip with return code 77. Real engine
checks need the corresponding `NINFER_QWEN3_6_27B_WEIGHTS` or
`NINFER_QWEN3_6_35B_A3B_WEIGHTS` environment and supported hardware; skipped tests
do not establish GPU correctness. Resident-server smoke tests, model downloads,
performance benchmarks and external evaluation services are outside this gate.

The known Qwen3.8 model is `/models/qwen3_8_27b.ninfer`; use
`export NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer`.

Baseline observed during setup on 2026-09-07:

- CMake configure passes with CUDA 13.3.73 and required FFmpeg/libcurl libraries.
- The Release application/test build passes. After the user freed the GPU,
  CTest reports 96 passes, zero failures and seven skips out of 103 tests.
  This supersedes the earlier GPU allocation failures. Real-artifact tests
  and the NVFP4 A4 test remain skipped. Do not stop other GPU workloads to
  make verification pass.
- `NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python` now selects the provisioned
  Python 3.12 environment with pytest, torch, NumPy, safetensors, PyYAML and
  Rich installed. Evaluation coordinator tests pass (19 tests). The default
  harness Python remains separate; use the selected project interpreter.
- Native Python imports pass. Artifact/converter and benchmark-consumer pytest
  reports 77 passes. The user approved removing the three upstream-local
  official-source cases that required a developer's private model directories.
  Self-contained converter and hash-validation coverage remains; no official
  model source directory is required for this pytest command.
- `/tmp` is mounted `noexec`, which blocks the launcher check's generated
  executable fixture. The existing check passes with a workspace temp root:
  `mkdir -p .local/test-tmp`, then
  `TMPDIR="$PWD/.local/test-tmp" bash scripts/check-linux-scripts.sh`.
  The sandbox now exports `TMPDIR=/workspace/ninfer/.local/test-tmp`, and the
  launcher check passes without a manual override. The directory is gitignored;
  no `/tmp` remount is needed. Use a checkout-local override when running from
  another worktree.

Refresh these baseline notes when new evidence resolves them.
