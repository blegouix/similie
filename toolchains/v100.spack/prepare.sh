# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: GPL-3.0-or-later

export SPACK_USER_PREFIX="${WORKDIR}/spack"

git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ${SPACK_USER_PREFIX}
. ${SPACK_USER_PREFIX}/share/spack/setup-env.sh

spack env create similie
spacktivate similie
spack external find
spack external find cuda
spack compiler find
spack add cuda
spack add netlib-lapack
spack add fftw -mpi +openmp
spack add ginkgo@1.8.0 cuda_arch=70 +cuda
spack install
