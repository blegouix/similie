ARG BASE_IMAGE = nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
FROM $BASE_IMAGE

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: MIT

# -----------------
# ----- USAGE -----
# -----------------

# docker build -t similie .
# sudo docker run -ti similie
# git config --global url."https://github.com/".insteadOf git@github.com:
# git clone --recurse-submodules https://github.com/blegouix/similie.git
# for cpu:
# cmake -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -B similie/build -S similie
# for cuda:
# cmake -DCMAKE_CXX_COMPILER=$PWD/similie/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=RELEASE -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B similie/build -S similie
# cmake --build similie/build -j 8
# ctest --test-dir similie/build

# -----------------------
# ----- BUILD CHAIN -----
# -----------------------

ARG BACKEND=cuda

RUN apt update

# vim
RUN apt install -y vim
ENV EDITOR=vim

# wget
RUN apt install -y wget

# git
RUN apt install -y git

# CMake 
RUN apt install  -y cmake cmake-curses-gui

# gcc
RUN apt install -y gcc g++

# OpenMPI
RUN apt install -y libopenmpi-dev

# FFTW3
RUN apt install -y libfftw3-dev

# LAPACKE
RUN apt install -y liblapacke-dev

# Ginkgo
RUN git clone -b v1.8.0 --depth 1 https://github.com/ginkgo-project/ginkgo.git
RUN if [ "${BACKEND}" = "cpu" ]; then \
      cmake -DCMAKE_BUILD_TYPE=Release \
            -DGINKGO_BUILD_OMP=ON \
            -DGINKGO_BUILD_TESTS=OFF \
            -DGINKGO_BUILD_EXAMPLES=OFF \
            -DGINKGO_BUILD_BENCHMARKS=OFF \
            -B ginkgo/build -S ginkgo; \
    elif [ "${BACKEND}" = "cuda" ]; then \
      cmake -DCMAKE_C_COMPILER=gcc \
            -DCMAKE_CXX_COMPILER=g++ \
            -DCMAKE_CUDA_HOST_COMPILER=g++ \
            -DCMAKE_BUILD_TYPE=Release \
            -DGINKGO_CUDA_ARCHITECTURES=70 \
            -DGINKGO_BUILD_CUDA=ON \
            -DGINKGO_BUILD_TESTS=OFF \
            -DGINKGO_BUILD_EXAMPLES=OFF \
            -DGINKGO_BUILD_BENCHMARKS=OFF \
            -B ginkgo/build -S ginkgo; \
    else \
      echo "ERROR: BACKEND must be 'cpu' or 'cuda' (got: '${BACKEND}')" >&2; \
      exit 1; \
    fi
RUN cmake --build ginkgo/build -j 8
RUN cmake --install ginkgo/build

# PDI 
RUN cat > /etc/apt/sources.list.d/pdi.sources <<'EOF'
Types: deb deb-src
URIs: https://repo.pdi.dev/ubuntu
Suites: noble
Components: main
Architectures: amd64
Signed-By: /usr/share/keyrings/pdidev-archive-automatic.gpg
EOF
RUN wget -O /usr/share/keyrings/pdidev-archive-automatic.gpg https://repo.pdi.dev/ubuntu/pdidev-archive-keyring.gpg
RUN apt update
RUN apt install -y libpdi-dev pdiplugin-all
