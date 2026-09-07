#!/bin/sh
set -eu

# Carapa invokes this from the selected checkout, without positional arguments.
root=$(pwd -P)
main_checkout=$(git -c core.quotePath=false worktree list --porcelain | sed -n '1s/^worktree //p')
test -n "$main_checkout"
command -v ccache >/dev/null

# Keep shared objects outside disposable task worktrees. Never overwrite a
# provisioned symlink with checkout-specific configuration.
for path in "$root/.local" "$root/.local/ccache.conf"; do
  if [ -L "$path" ]; then
    printf 'Refusing to write worktree configuration through symlink: %s\n' "$path" >&2
    exit 1
  fi
done
mkdir -p "$root/.local/test-tmp" "$main_checkout/.cache/ccache"
config="$root/.local/ccache.conf"
ccache --config-path "$config" --set-config "cache_dir=$main_checkout/.cache/ccache"
ccache --config-path "$config" --set-config "base_dir=$root"

python=${NINFER_VERIFY_PYTHON:-python3}
cmake -S "$root" -B "$root/build-agent-verify" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF \
  -DNINFER_USE_CCACHE=ON "-DNINFER_CCACHE_CONFIG=$config" \
  "-DPython3_EXECUTABLE=$(command -v "$python")"
