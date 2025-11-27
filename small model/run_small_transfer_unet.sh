#!/bin/bash
#SBATCH --account=def-mere
#SBATCH --job-name=small_u_transfer
#SBATCH --time=70:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=fixed_small_unet_transfer_%j.out
#SBATCH --error=fixed_small_unet_transfer_%j.err

set -euo pipefail
set -x

module purge
module load StdEnv/2023
module load intel/2023.2.1
module load cuda/11.8
module load mpi4py/4.0.3
module load proj/9.4.1
module load netcdf/4.9.2

source /project/def-mere/merra2/g5nr_env_beluga/bin/activate
echo "Python: $(which python)"
python - <<'PY'
import torch, os, sys
print("Torch-CUDA OK:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

BASE=/project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets
RUNDIR=$BASE/small_transfer_model/new_small_result
mkdir -p "$RUNDIR"

python -u $BASE/get_large_seasonal_data_ready.py "$RUNDIR" /project/def-mere/merra2/merra2/merged_global/MERRA2_merged_2000-2024_fixed.nc

echo "Training small transfer models…"

for s in {1..4}; do
  for a in {0..1}; do
    AREADIR=$RUNDIR/Season${s}/Area${a}
    mkdir -p "$AREADIR"
    echo "Starting training the new small transfer model for Season $s Area $a"
    python -u $BASE/small_transfer_model/transfer_unet_fur_diffuser.py "$AREADIR"

    if [[ -f "$AREADIR/downscaled_mean.npy" ]]; then
      echo "Evaluating Season ${s} Area ${a}"
      python -u $BASE/small_transfer_model/evaluate_small.py "$AREADIR"
    else
      echo "downscaled_mean.npy not found for $AREADIR -> skipped evaluation"
    fi
  done
done


echo "We finished training the new small transfer model for all seasons / areas."
echo "We start doing evaluations."
