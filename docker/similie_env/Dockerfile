ARG BASE_IMAGE=nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
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
# cmake -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -B similie/build -S similie
# for cuda:
# cmake -DCMAKE_CXX_COMPILER=$PWD/similie/vendor/ddc/vendor/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=RELEASE -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B similie/build -S similie
# cmake --build similie/build -j 8
# ctest --test-dir similie/build

# -----------------------
# ----- BUILD CHAIN -----
# -----------------------

ARG BACKEND=cuda

# vim
RUN apt update && apt install -y vim
ENV EDITOR=vim

# apt helpers
RUN apt update && apt install -y ca-certificates gpg lsb-release wget

# git
RUN apt update && apt install -y git

# CMake
RUN apt update \
  && apt install -y cmake cmake-curses-gui \
  && cmake --version

# gcc
RUN if [ "${BACKEND}" = "cuda" ]; then \
      apt update \
      && apt install -y software-properties-common \
      && add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
    fi
RUN apt update && apt install -y gcc-15 g++-15
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-15 100 \
 && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-15 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-15 100

# OpenMPI
RUN apt update && apt install -y libopenmpi-dev

# FFTW3
RUN apt update && apt install -y libfftw3-dev

# LAPACKE
RUN apt update && apt install -y liblapacke-dev

# Thrust
RUN if [ "${BACKEND}" = "cuda" ]; then \
  apt update && apt install -y libthrust-dev; \
fi

# Ginkgo
RUN git clone -b v1.11.0 --depth 1 https://github.com/ginkgo-project/ginkgo.git
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
            -DGINKGO_BUILD_CUDA=ON \
            -DGINKGO_CUDA_ARCHITECTURES=89 \
            -DGINKGO_BUILD_TESTS=OFF \
            -DGINKGO_BUILD_EXAMPLES=OFF \
            -DGINKGO_BUILD_BENCHMARKS=OFF \
            -B ginkgo/build -S ginkgo; \
    else \
      echo "ERROR: BACKEND must be 'cpu' or 'cuda' (got: '${BACKEND}')" >&2; \
      exit 1; \
    fi
RUN cmake --build ginkgo/build
RUN cmake --install ginkgo/build

# OpenCASCADE
RUN apt update && apt install -y \
  libfreetype-dev \
  libocct-foundation-dev \
  libocct-modeling-data-dev \
  libocct-modeling-algorithms-dev \
  libocct-data-exchange-dev \
  libocct-ocaf-dev

# gmsh
RUN apt update && apt install -y libfltk-dev
RUN git clone https://gitlab.onelab.info/gmsh/gmsh.git
RUN cmake -DENABLE_OCC=ON -DENABLE_OCC_CAF=ON -B gmsh/build -S gmsh
RUN cmake --build gmsh/build -j 4
RUN cmake --install gmsh/build

# PDI
RUN update-ca-certificates
RUN printf '%s\n' \
  'Types: deb deb-src' \
  'URIs: https://repo.pdi.dev/ubuntu' \
  "Suites: $(lsb_release -cs)" \
  'Components: main' \
  'Architectures: amd64' \
  'Signed-By: /usr/share/keyrings/pdidev-archive-automatic.gpg' \
  > /etc/apt/sources.list.d/pdi.sources
RUN wget -O /usr/share/keyrings/pdidev-archive-automatic.gpg https://repo.pdi.dev/ubuntu/pdidev-archive-keyring.gpg \
  && chmod a+r /usr/share/keyrings/pdidev-archive-automatic.gpg
RUN apt update && apt install -y libpdi-dev pdiplugin-all

# Sympy
RUN apt install -y python3-sympy python3-h5py
