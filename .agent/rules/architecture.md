---
trigger: model_decision
description: Use when changing NInfer engine, CUDA operators, target programs, artifacts, CLI, or serving protocols.
---

# NInfer architecture and contribution boundaries

- This fork targets Linux and RTX 4090 (`sm_89`), primarily Qwen3.8-27B with
  groupwise-int `.ninfer` weights. Do not infer support for Blackwell NVFP4/W4A4
  execution from inherited code or converter tests. README.md identifies the
  inherited Windows and Qwen3.6-35B-A3B paths as untested on RTX 4090.
- C++20/CUDA20 builds use CMake/Ninja. Preserve `.clang-format` (four-space
  indentation, 100-column limit, existing include order). Python tooling lives
  under `tools/`; the evaluation coordinator is under `eval/ninfer_eval`.
- Preserve Gateway → Frontend → Engine → Program ownership, as defined in
  `docs/maintainer/engine-architecture.md`: gateways own protocol/transport,
  Frontend owns prompt/output semantics, Engine owns lifecycle/publication,
  and Program owns physical model execution and State/KV resources. Scheduler
  owns execution order; ResourceManager owns logical context retention.
- Public Op contracts in `include/ninfer/ops/` define represented inputs,
  formula, effects, aliasing, and workspace. Wrappers validate and dispatch;
  launchers/kernels implement the formula. Model scheduling and transaction
  policy remain outside Ops. Follow `docs/maintainer/op-development.md`.
- Test exact formats/transforms exactly and floating-point behavior against
  independent mathematical oracles with justified tolerances. Stateful checks
  must cover the complete affected transition. Performance claims require
  measurements at the claimed level; use `bench/README.md`.
- Keep serving implementation, schema tests, and active documentation aligned.
  Artifact format/layout authorities are under `docs/maintainer/`; avoid
  introducing a second format or execution path for a localized fix.
- CONTRIBUTING.md requires confirmed scope before external contributions,
  targets PRs at master, and requests checking dev for overlapping work.
  Follow the user's authorized task scope; project setup does not authorize
  unrelated implementation, publishing, or new board tasks.
