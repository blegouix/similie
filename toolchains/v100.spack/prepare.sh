export SPACK_USER_PREFIX="${WORKDIR}/spack"

git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ${SPACK_USER_PREFIX}
. ${SPACK_USER_PREFIX}/share/spack/setup-env.sh

spack external find cmake 
spack external find cuda
spack install ginkgo@1.8.0%gcc@11 cuda_arch=70 ^cuda@11.5+allow-unsupported-compilers
spack install fftw -mpi +openmp precision=double 
