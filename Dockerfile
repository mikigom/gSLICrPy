# CUDA 11.8 on Ubuntu 22.04. CUDA 11+ dropped sm_30 (Kepler), so we rewrite
# the compute_30 arch in CMakeLists.txt to compute_75 (Turing: RTX 20xx/T4),
# which also runs on newer arches via forward-compatible PTX.
# Override at build time with:  --build-arg CUDA_ARCH=compute_86
ARG CUDA_ARCH=compute_75
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG CUDA_ARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libopencv-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir \
        numpy \
        scikit-image \
        opencv-python-headless

WORKDIR /app
COPY . /app

RUN sed -i "s/arch=compute_30,code=compute_30/arch=${CUDA_ARCH},code=${CUDA_ARCH}/g" CMakeLists.txt \
 && mkdir -p build \
 && cd build \
 && cmake .. \
 && make -j"$(nproc)"

ENV LD_LIBRARY_PATH=/app/build:${LD_LIBRARY_PATH}

CMD ["python3", "example.py"]
