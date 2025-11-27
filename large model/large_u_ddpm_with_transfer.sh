#!/bin/bash
#SBATCH --account=def-mere
#SBATCH --job-name=large_3ddpm_wt
#SBATCH --time=28:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=fixed3_large_ddpm_with_transfer_%j.out
#SBATCH --error=fixed3_large_ddpm_with_transfer_%j.err

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
RUNDIR=$BASE/new_large_ddpm_3_with_transfer_result      # directory of where the result of the training will be saved. this is the third try. 
# mkdir -p "$RUNDIR"

# 3. (Optional, commented out this part if this script is runned and reached here before.) create seasonal splits (RUN ONLY ONCE)
echo "Creating seasonal train and test indices for the large DDPM model with 2 years G5NR data."
python -u $BASE/get_large_seasonal_data_ready.py "$RUNDIR" /project/def-mere/merra2/g5nr/G5NR_daily_merged_noclip/G5NR_merged_daily_noclip_2005-2007.nc
# /project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets/get_large_seasonal_data_ready.py

echo "Training large DDPM models with small transfer models. This is our third try!"

for s in {1..4}; do
  for a in {0..1}; do
    AREADIR=$RUNDIR/Season${s}/Area${a}
    mkdir -p "$AREADIR"
    echo "Starting training the new large DDPM with small transfer model for Season $s Area $a"
    python -u $BASE/train_ddpm_with_transfer.py "$AREADIR"     # /project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets/train_ddpm_with_transfer.py
    python -u $BASE/evaluate_large_ddpm.py "$AREADIR"   # /project/def-mere/merra2/Ace_Transfer_Downscale/Ace_forward_unets/evaluate_large_unet.py
  done
done

echo "Congratulations! We finished training the new large DDPM with small transfer model for all seasons / areas."
echo "We are done! :)"