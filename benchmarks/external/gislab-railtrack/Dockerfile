FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libboost-all-dev \
        libopencv-dev \
        libpcl-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/railroad-src
COPY . .

# This follows the upstream CI recipe: build the pinned LAStools submodule,
# configure src/, then install the benchmark executable and shared library.
RUN make -C vendor -j2 \
    && cmake -S src -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/railroad \
    && cmake --build build --config Release -j2 \
    && cmake --install build

ENV LD_LIBRARY_PATH=/opt/railroad
ENTRYPOINT ["/opt/railroad/railroad_benchmark"]
