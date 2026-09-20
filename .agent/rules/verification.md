---
trigger: model_decision
description: Use when implementing, testing, or reviewing NInfer changes, running verification, or diagnosing sandbox test prerequisites.
---

# Verification

## Choose checks by impact

Default to focused verification that covers the changed behavior and realistic
regressions. Choose the scope by behavioral impact, not by the number of changed
lines. A full suite is not an automatic completion requirement.

- Documentation and rule edits: review the affected text and references, then
  run `git diff --check`. No build or runtime tests are needed.
- Local changes to file permissions, logging, CLI options, or other host-side
  behavior: build the affected targets and run the relevant existing tests or a
  direct behavior check. For example, verify log permissions under a restrictive
  umask; do not launch unrelated model or GPU tests.
- Engine, cache, scheduling, model, or CUDA changes: run the affected correctness
  and integration regressions, including the relevant real-model/GPU cases when
  their behavior is affected. Broaden verification when impact crosses several
  paths or a failure leaves the regression boundary uncertain.

Run the full canonical command only when the user explicitly requests it, an
applicable release/acceptance requirement calls for it, or broad changes or
unresolved failures cannot be adequately checked with a focused selection.
Before launching it, state the concrete reason the full suite is needed.

Coding, independent Testing, and Review use the same scope-selection rule.
Entering a stage does not by itself require a full suite or duplicate run.
Review uses existing evidence when it covers the current implementation. Once
the selected checks pass, stop testing; repeat or broaden only when new changes,
failures, or unresolved concerns invalidate that evidence. Distinguish focused
passes, full-suite results, baseline failures, regressions, and skips. Never
weaken checks to obtain a pass or claim results for checks that were not run.

Focused commands are documented in `tests/README.md`: build the affected test
target, then `ctest --test-dir build-agent-verify -R '<test-name>'
--output-on-failure`. Use `NINFER_OP_REPORT_STATS=1` with verbose CTest for
numerical error evidence.

## Full canonical command

When full verification is warranted, run `bash .agent/verify.sh` from the
repository root. The command runs existing Linux launcher checks, native
Python artifact/converter and benchmark-consumer
suites, evaluation coordinator unittests, then a Release sm_89 application/test
build and CTest. Independent suites continue after failures; the command exits
nonzero if any step fails. No dependencies are installed and no persistent
server is started. There is no repository CI workflow or browser UI.

Prerequisites: Bash, CMake ≥3.28, Ninja, C++20 compiler, CUDA ≥12.8, FFmpeg dev
libraries, libcurl and OpenSSL Crypto meeting CMake's constraints. Python needs pytest, torch,
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

The initial two-job native build took about eight minutes in this sandbox.
With the Qwen3.8 real artifact enabled, CTest took
about 765 seconds on 2026-09-12; the Python artifact/converter suites took about
two seconds. Do not treat the full command as a fast lint gate.

GPU/device and real-artifact tests may skip with return code 77. For this
project's RTX 4090 target, the required real artifact is the known Qwen3.8-27B
groupwise-int model at `/models/qwen3_8_27b.ninfer`; use
`export NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer`. Qwen3.8-27B uses
the shared 27B implementation and real-test binary whose historical source and
target names contain `qwen3_6_27b`; those names do not make a separate Qwen3.6
artifact a prerequisite.

Do not request, require, or block on `NINFER_QWEN3_6_*`, Qwen3.6 model/tokenizer
directories, an NVFP4 artifact, or Blackwell FP4 hardware unless the task
explicitly targets that non-default path. Their canonical skips are expected
non-target results: report them separately if useful, but they do not leave the
Qwen3.8 groupwise acceptance boundary unverified and must not affect the stage
verdict. A failure is relevant when it is reproduced on the Qwen3.8 artifact or
on a model-independent path reachable by that target. A skipped applicable
Qwen3.8 check still does not establish GPU correctness.

Resident-server smoke tests, model downloads, performance benchmarks and
external evaluation services are outside this gate.

Baseline refreshed during setup on 2026-09-12:

- CMake configure passes with CUDA 13.3.73 and required FFmpeg/libcurl libraries.
- The Release application/test build passes. With
  `NINFER_QWEN3_8_27B_WEIGHTS=/models/qwen3_8_27b.ninfer`, CTest has 108
  registered tests: 102 pass, zero fail and six are expected non-target skips.
  The skipped set is five inherited Qwen3.6-only real/load-plan checks plus the
  Blackwell-only NVFP4 A4 check. Do not stop other GPU workloads to make
  verification pass.
- `NINFER_VERIFY_PYTHON=/opt/ninfer-venv/bin/python` now selects the provisioned
  Python 3.12 environment with pytest, torch, NumPy, safetensors, PyYAML and
  Rich installed. Evaluation coordinator tests pass (19 tests). The default
  harness Python remains separate; use the selected project interpreter.
- Native Python imports pass. Artifact/converter and benchmark-consumer pytest
  reports 95 passes. The user approved removing the three upstream-local
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
