---
trigger: model_decision
description: Use when configuring or building ninfer with CMake and Ninja in this sandbox, creating a build directory, or choosing build options.
---

# Building ninfer with CMake

The native build dependencies were confirmed by CMake configuration in this
sandbox: the FFmpeg dev libs (`libavformat`, `libavcodec`, `libavutil`,
`libswscale`), CUDA (≥ 12.8, at `/usr/local/cuda`), and `libcurl`. This does not
cover Python test dependencies; see `verification.md` for the canonical gate
and its prerequisites.

## Configure and build

The top-level `CMakeLists.txt` enforces `CMAKE_CUDA_ARCHITECTURES=89` (also the
default), so the minimum command is:

    cmake -S . -B build-sm89 -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DNINFER_BUILD_APPS=ON \
      -DBUILD_TESTING=OFF \
      -DNINFER_BUILD_BENCHMARKS=OFF

    cmake --build build-sm89 --parallel

- Compiler detection is automatic in the sandbox (`nvcc` at `/usr/local/cuda/bin`);
  you do not need to pass `CMAKE_*_COMPILER` explicitly.
- Build options: `NINFER_BUILD_APPS` (CLI + server, default ON), `BUILD_TESTING`,
  `NINFER_BUILD_BENCHMARKS`. `cmake --build` compiles whichever are enabled.
- Outputs land in `<builddir>/apps/` as `ninfer`, `ninfer-serve`, and
  `ninfer-perplexity`.
- Use `bash .agent/verify.sh` for the application/test build and test suites.
  The application-only command above is not the verification gate.

## Verifying the environment

- Confirm a dependency by running a real CMake configure (it honors `REQUIRED`
  + version constraints), not a bare `pkg-config` / `which` call.
- The `ffmpeg` **CLI is not on `PATH`**, but the FFmpeg **libraries** are
  installed. A `which ffmpeg` miss does NOT mean the build is broken — the build
  links the libs, not the CLI.

## Carapa worktrees and compiler reuse

CarapaBox owns deterministic worktree provisioning. Task agents use the prepared
build directory; do not put hook invocation, fallback setup, or lifecycle
management instructions in worker/stage prompts. The details below are for
maintaining the build and provisioning infrastructure.

Carapa invokes `sh .agent/worktree-setup.sh` from the selected worktree root,
with no arguments, on add/reuse/reset. The POSIX hook configures the local
`build-agent-verify` directory without compiling or installing dependencies.
It uses `NINFER_VERIFY_PYTHON` (otherwise `python3`); the sandbox's provisioned
`/opt/ninfer-venv/bin/python` can be reused across worktrees.

Setup discovers the main checkout with `git worktree list --porcelain` and
stores shared compiler objects in its ignored `.cache/ccache`. Each checkout
gets `.local/ccache.conf` with its own `base_dir` for path normalization.
`NINFER_CCACHE_CONFIG` persists that file's path in the CMake cache; generated
compiler launch commands explicitly set `CCACHE_CONFIGPATH`, so they do not
depend on hook exports surviving. Explicit container `CCACHE_*` overrides still
take precedence over ccache configuration. Setup rejects symlinked local
configuration to avoid overwriting provisioned shared files.

The hook disables CMake C++ module scanning because the project has no C++
modules and the resulting GCC module flags prevent ccache reuse. Other CMake
build directories retain their existing defaults. Keep source paths, compiler
options and build-directory layout consistent to maximize cross-checkout hits.
Do not share or copy CMake caches/build directories across worktrees. The first
cached compilation populates the cache; configuration, linking (including CUDA
device linking), and tests still run. The default ccache size limit applies.

`sh .agent/worktree-teardown.sh` is intentionally a no-op: setup creates no
services or external per-worktree resources, and Carapa owns checkout cleanup.
Preserve the shared cache on removal/reset. These are lifecycle hooks, not
per-stage hooks; failures are best-effort unless the operator enables
`WORKTREE_HOOKS_STRICT`. Do not change that operator setting automatically.
New worktrees use their own tracked hook copies: uncommitted edits in the main
checkout are not automatically included.
