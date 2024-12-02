# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

export SPACK_USER_PREFIX="${WORKDIR}/spack"

. ${SPACK_USER_PREFIX}/share/spack/setup-env.sh

spack load cuda
spack load netlib-lapack
spack load fftw
spack load ginkgo
