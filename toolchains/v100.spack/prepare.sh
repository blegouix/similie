# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

export SPACK_USER_PREFIX="${WORKDIR}/spack"

git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ${SPACK_USER_PREFIX}
. ${SPACK_USER_PREFIX}/share/spack/setup-env.sh

spack external find cmake
spack compiler find
spack install cuda
spack install openblas
spack install fftw -mpi +openmp
spack install ginkgo@1.8.0 cuda_arch=70 +cuda
