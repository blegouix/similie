# SPDX-FileCopyrightText: 2024 Baptiste Legouix
# SPDX-License-Identifier: MIT

export SPACK_USER_PREFIX="${WORKDIR}/spack"

# Spack
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ${SPACK_USER_PREFIX}
. ${SPACK_USER_PREFIX}/share/spack/setup-env.sh

# PDI repo
git clone https://github.com/pdidev/spack.git ${SPACK_USER_PREFIX}/var/spack/repos/pdi
spack repo add ${SPACK_USER_PREFIX}/var/spack/repos/pdi

# Install environment
spack env create similie
spacktivate similie
spack external find
spack external find cuda
spack compiler find
spack add cmake
spack add cuda
spack add netlib-lapack
spack add fftw -mpi +openmp
spack add ginkgo@1.8.0 cuda_arch=70 +cuda
spack add pdi -fortran -python
spack add pdiplugin-decl-hdf5 -fortran -mpi
spack install
