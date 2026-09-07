#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# The sandbox's inherited TMPDIR can point at the main checkout; /tmp is noexec.
export TMPDIR="$root/.local/test-tmp"
mkdir -p "$TMPDIR"

python="${NINFER_VERIFY_PYTHON:-python3}"
eval_python="${NINFER_VERIFY_EVAL_PYTHON:-$python}"
build_dir="${NINFER_VERIFY_BUILD_DIR:-build-agent-verify}"
failed=0

# Independent suites still run after a failure so the report includes the full baseline.
check() {
  local label="$1"
  shift
  printf '\nChecking %s\n' "$label"
  if "$@"; then
    printf 'PASS: %s\n' "$label"
  else
    printf 'FAIL: %s\n' "$label" >&2
    failed=1
  fi
}

check 'Linux scripts' bash scripts/check-linux-scripts.sh
check 'Python artifact and converter suites' "$python" -m pytest -q \
  tests/artifact tests/convert tests/test_bench_matrix.py tests/test_serve_corpus.py
check 'Evaluation coordinator' env PYTHONPATH="$root/eval${PYTHONPATH:+:$PYTHONPATH}" \
  "$eval_python" -m unittest discover -s eval/tests -p 'test_*.py'

if cmake -S . -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF \
    "-DPython3_EXECUTABLE=$(command -v "$python")"; then
  if cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"; then
    check 'CTest' ctest --test-dir "$build_dir" --output-on-failure --no-tests=error
  else
    printf 'FAIL: C++/CUDA build; CTest not run\n' >&2
    failed=1
  fi
else
  printf 'FAIL: CMake configure; build and CTest not run\n' >&2
  failed=1
fi

exit "$failed"
