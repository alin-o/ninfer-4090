#!/bin/sh
set -eu

STACK_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CARAPA_STACKS_DIR="${CARAPA_STACKS_DIR:-$STACK_DIR/../carapa/my-stacks}"

case "${1:-}" in
    --warm-cache) CARAPA_WARM_CACHE=1; export CARAPA_WARM_CACHE; shift ;;
    --help|-h)
        echo 'Usage: ./build.sh [--warm-cache] [docker buildx build options]'
        echo 'Builds ninfer-4090:sm89 with caches on /mnt/S/.cache/docker.'
        echo '--warm-cache populates the persistent cache without exporting or loading an image.'
        exit 0 ;;
esac

CARAPA_BUILDX_SCRIPT="$CARAPA_STACKS_DIR/scripts/carapa-buildx.sh"
if [ ! -r "$CARAPA_BUILDX_SCRIPT" ]; then
    echo "Missing Carapa build helper: $CARAPA_BUILDX_SCRIPT" >&2
    exit 1
fi
CARAPA_BUILD_TOOLS_DIR="${CARAPA_BUILD_TOOLS_DIR:-$CARAPA_STACKS_DIR/scripts}"
# shellcheck disable=SC1090
. "$CARAPA_BUILDX_SCRIPT"
cd "$STACK_DIR"
carapa_buildx_build \
    --build-context 'carapa-llama-cpp:latest=docker-image://carapa-llama-cpp:latest' \
    -t ninfer-4090:sm89 "$@" "$STACK_DIR"
