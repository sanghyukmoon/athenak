#!/bin/bash
# Build only; no correctness or performance job is launched.
set -euo pipefail
backend=${1:?usage: build.sh cpu\|gpu\|a100 SCRATCH_DIRECTORY}
artifact=${2:?supply an artifact directory under the approved scratch root}
case "$artifact/" in
  /scratch/gpfs/sm69/onthefly-rprof-test/*/) ;;
  *) echo 'Build artifacts must be under approved scratch root' >&2; exit 2 ;;
esac
source_root=$(cd "$(dirname "$0")/../.." && pwd)
module purge
case "$backend" in
  cpu)
    module load gcc-toolset/10 openmpi/gcc-toolset-10/4.1.0
    options=(-DAthena_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER=mpicxx)
    ;;
  gpu|a100)
    module load gcc-toolset/10 nvhpc/25.5 openmpi/cuda-12.9/nvhpc-25.5/4.1.8 cudatoolkit/12.9
    export OMPI_CXX=g++ NVCC_WRAPPER_DEFAULT_COMPILER=g++
    arch=VOLTA70
    if [[ "$backend" == a100 ]]; then arch=AMPERE80; fi
    options=(-DKokkos_ENABLE_CUDA=ON "-DKokkos_ARCH_${arch}=ON"
      "-DCMAKE_CXX_COMPILER=$source_root/kokkos/bin/nvcc_wrapper"
      "-DMPI_CXX_COMPILER=$(command -v mpicxx)" -DCMAKE_CXX_FLAGS=-DOMPI_SKIP_MPICXX=1)
    ;;
  *) echo 'backend must be cpu, gpu (V100), or a100' >&2; exit 2 ;;
esac
cmake -S "$source_root" -B "$artifact/build-$backend" \
  -DPROBLEM=radial_profile_benchmark -DAthena_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release "${options[@]}"
cmake --build "$artifact/build-$backend" -j 4
