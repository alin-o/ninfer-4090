# syntax=docker/dockerfile:1

FROM carapa-llama-cpp:latest AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ccache \
    && rm -rf /var/lib/apt/lists/*

# Set to 1 to disable parallel CUDA optimization passes.
ARG CUDA_SPLIT_COMPILE=2

WORKDIR /src
COPY . .

# /build is a BuildKit cache mount, so Ninja's tree survives across docker
# builds: even when the source layer changes, only modified files recompile.
# The toolchain stamp forces a clean tree if the base image's compilers move.
# Cache-mount contents are invisible outside this RUN (COPY --from=build sees
# only the image filesystem), so the finished binaries are staged to /out.
RUN --mount=type=cache,id=ninfer-4090-sm89,target=/build \
    --mount=type=cache,id=ninfer-4090-sm89-ccache,target=/root/.cache/ccache \
    set -eux; \
    case "$CUDA_SPLIT_COMPILE" in ''|0*|*[!0-9]*) echo "CUDA_SPLIT_COMPILE must be a positive integer" >&2; exit 1;; esac; \
    export CCACHE_DIR=/root/.cache/ccache; \
    toolchain="cuda$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')-gcc$(gcc -dumpversion)"; \
    if [ -f /build/.toolchain ] && [ "$(cat /build/.toolchain)" != "$toolchain" ]; then \
      echo "toolchain changed to $toolchain; clearing /build"; \
      find /build -mindepth 1 -delete; \
    fi; \
    printf '%s' "$toolchain" > /build/.toolchain; \
    cmake -S /src -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_FLAGS="--split-compile=${CUDA_SPLIT_COMPILE}" \
      -DNINFER_BUILD_APPS=ON \
      -DBUILD_TESTING=OFF \
      -DNINFER_BUILD_BENCHMARKS=OFF; \
    cmake --build /build --parallel --target ninfer ninfer-serve; \
    mkdir -p /out; \
    install -m 0755 /build/apps/ninfer /build/apps/ninfer-serve /out/

FROM carapa-llama-cpp:latest

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    ca-certificates \
    libavcodec62 \
    libavformat62 \
    libavutil60 \
    libcurl4t64 \
    libswscale9 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ninfer /usr/local/bin/ninfer
COPY --from=build /out/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

ENTRYPOINT ["ninfer-serve"]
CMD ["--help"]
