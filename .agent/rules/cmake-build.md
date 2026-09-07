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
