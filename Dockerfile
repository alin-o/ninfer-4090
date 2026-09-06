# syntax=docker/dockerfile:1

FROM carapa-llama-cpp:latest AS build

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
COPY . .

RUN cmake -S . -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNINFER_BUILD_APPS=ON \
    -DBUILD_TESTING=OFF \
    -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve

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

COPY --from=build /build/apps/ninfer /usr/local/bin/ninfer
COPY --from=build /build/apps/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM

ENTRYPOINT ["ninfer-serve"]
CMD ["--help"]
