#!/bin/bash
#SBATCH --job-name=rprof-density
#SBATCH --account=astro
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Submit only AFTER the user's explicit code-review approval:
# sbatch --chdir=/scratch/.../approved-campaign performance.slurm --review-approved EXE
set -euo pipefail
if [[ ${1:-} != --review-approved || $# != 2 ]]; then
  echo 'Requires code-review approval and executable path: --review-approved EXE' >&2
  exit 2
fi
case "$PWD/" in
  /scratch/gpfs/sm69/onthefly-rprof-test/*/) ;;
  *) echo 'Run directory must be under approved scratch root' >&2; exit 2 ;;
esac
module purge
module load gcc-toolset/10 nvhpc/25.5 openmpi/cuda-12.9/nvhpc-25.5/4.1.8 cudatoolkit/12.9
exe=$(realpath "$2")
source_root=/home/sm69/athenak
export OMP_NUM_THREADS=1
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,smcuda,tcp
export OMPI_MCA_coll=^hcoll
nvidia-smi > gpu-placement.txt
ompi_info --parsable --all > mpi-capabilities.txt
for cells in 128 256; do
  for block in 32 64; do
    run="N${cells}-B${block}"
    mkdir "$run"
    cp "$source_root/inputs/radial_profile/athinput.density" "$run/input"
    (
      cd "$run"
      srun "$exe" -i input problem/performance=true problem/repeats=20 \
        mesh/nx1="$cells" mesh/nx2="$cells" mesh/nx3="$cells" \
        meshblock/nx1="$block" meshblock/nx2="$block" meshblock/nx3="$block" \
        > run.log 2>&1
    )
  done
done
