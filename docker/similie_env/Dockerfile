ARG BASE_IMAGE=nvidia/cuda:13.4.1-cudnn-devel-ubuntu26.04
FROM $BASE_IMAGE

# SPDX-FileCopyrightText: 2026 Baptiste Legouix
# SPDX-License-Identifier: AGPL-3.0-or-later

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
# cmake -DCMAKE_CXX_COMPILER=/opt/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=RELEASE -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_<YOUR_ARCH_NAMECODE>=ON -B similie/build -S similie
# cmake --build similie/build -j 8
# ctest --test-dir similie/build

# ----------------
# ----- MISC -----
# ----------------

ARG BACKEND=cuda

# vim
RUN apt update && apt install -y vim
ENV EDITOR=vim

# apt helpers
RUN apt update && apt install -y ca-certificates gpg lsb-release wget

# git
RUN apt update && apt install -y git

# ---------------------------
# ----- BUILD TOOLCHAIN -----
# ---------------------------

# CMake 4.4
RUN apt update && apt install -y ca-certificates gpg wget \
  && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
  && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ resolute main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
  && apt update \
  && apt install -y kitware-archive-keyring cmake cmake-curses-gui \
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

# ----------------------------------
# ----- TESTING & IO LIBRARIES -----
# ----------------------------------

# GTest & GMock
RUN apt install libgtest-dev libgmock-dev

# Google Benchmark
RUN git clone --depth 1 --branch v1.9.5 https://github.com/google/benchmark.git /opt/benchmark \
 && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_TESTING=OFF -B /opt/benchmark/build -S /opt/benchmark \
 && cmake --build /opt/benchmark/build \
 && cmake --install /opt/benchmark/build

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

# -----------------------------------------
# ----- REFERENCE COMPUTING LIBRARIES -----
# -----------------------------------------

# FFTW3
RUN apt update && apt install -y libfftw3-dev

# LAPACKE
RUN apt update && apt install -y liblapacke-dev

# Thrust
RUN if [ "${BACKEND}" = "cuda" ]; then \
  apt update && apt install -y libthrust-dev; \
fi

# ------------------
# ----- ONELAB -----
# ------------------

# OpenCASCADE
RUN apt update && apt install -y \
  libfreetype-dev \
  libocct-foundation-dev \
  libocct-modeling-data-dev \
  libocct-modeling-algorithms-dev \
  libocct-data-exchange-dev \
  libocct-ocaf-dev

# gmsh
RUN apt update && apt install -y libfltk1.3-dev
RUN git clone https://gitlab.onelab.info/gmsh/gmsh.git
RUN cmake -DENABLE_BUILD_LIB=ON -DENABLE_BUILD_SHARED=ON -DENABLE_PRIVATE_API=ON -DENABLE_OCC=ON -DENABLE_OCC_CAF=ON -B gmsh/build -S gmsh
RUN cmake --build gmsh/build
RUN cmake --install gmsh/build

# GetDP
RUN git clone https://gitlab.onelab.info/getdp/getdp.git
RUN cmake -B getdp/build -S getdp
RUN cmake --build getdp/build
RUN cmake --install getdp/build

# ------------------
# ----- GINKGO -----
# ------------------

# Ginkgo
RUN git clone https://github.com/ginkgo-project/ginkgo.git \
 && git -C ginkgo checkout --detach 527f52e1f6de69b6a5ef81c4c2e61144c53e5131 # Contains fixes for CUDA 13.4
RUN if [ "${BACKEND}" = "cpu" ]; then \
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DGINKGO_BUILD_OMP=ON \
        -DGINKGO_BUILD_TESTS=OFF \
        -DGINKGO_BUILD_EXAMPLES=OFF \
        -DGINKGO_BUILD_BENCHMARKS=OFF \
        -B ginkgo/build -S ginkgo; \
elif [ "${BACKEND}" = "cuda" ]; then \
  # WORKAROUND FOR CUDA 13.4: Thrust reduce_by_key fails to compile for Ginkgo's half and bfloat16 types, which are thus disabled.
  cmake -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_HOST_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DGINKGO_BUILD_CUDA=ON \
        -DGINKGO_CUDA_ARCHITECTURES=89 \
        -DGINKGO_ENABLE_HALF=OFF \
        -DGINKGO_ENABLE_BFLOAT16=OFF \
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

# ----------------------
# ----- KOKKOS-ISH -----
# ----------------------

# Kokkos
RUN git clone --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos.git /opt/kokkos
RUN if [ "${BACKEND}" = "cpu" ]; then \
      cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON \
        -B /opt/kokkos/build -S /opt/kokkos; \
    else \
      cmake -DCMAKE_CXX_COMPILER=/opt/kokkos/bin/nvcc_wrapper -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_ADA89=ON \
        -DKokkos_ENABLE_CUDA_CONSTEXPR=ON -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
        -B /opt/kokkos/build -S /opt/kokkos; \
    fi
RUN cmake --build /opt/kokkos/build && cmake --install /opt/kokkos/build

# Kokkos FFT
RUN git clone --depth 1 --branch v1.1.0 https://github.com/kokkos/kokkos-fft.git /opt/kokkos-fft \
 && cmake -DCMAKE_CXX_COMPILER=$(if [ "${BACKEND}" = "cpu" ]; then echo g++; else echo /usr/local/bin/nvcc_wrapper; fi) \
      -DCMAKE_BUILD_TYPE=Release -DKokkosFFT_ENABLE_FFTW=ON \
      -B /opt/kokkos-fft/build -S /opt/kokkos-fft \
 && cmake --build /opt/kokkos-fft/build \
 && cmake --install /opt/kokkos-fft/build

# Kokkos Kernels
RUN git clone --depth 1 --branch 5.2.2 https://github.com/kokkos/kokkos-kernels.git /opt/kokkos-kernels \
 && cmake -DCMAKE_CXX_COMPILER=$(if [ "${BACKEND}" = "cpu" ]; then echo g++; else echo /usr/local/bin/nvcc_wrapper; fi) \
      -DCMAKE_BUILD_TYPE=Release -B /opt/kokkos-kernels/build -S /opt/kokkos-kernels \
 && cmake --build /opt/kokkos-kernels/build \
 && cmake --install /opt/kokkos-kernels/build

# ------------------------------
# ----- PYTHON ENVIRONMENT -----
# ------------------------------

# Python packages
RUN apt install -y python3-sympy python3-h5py
