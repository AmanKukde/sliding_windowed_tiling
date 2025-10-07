#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/sliding_windowed_tiling/mu-split/logs/2025-10-06_15-51-09/inf_PAVIA_ATN_ActTub_DeepLC_slideon.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=inf_PAVIA_ATN_ActTub_DeepLC_slideon
#SBATCH --time=36:00:00

cd /home/aman.kukde/sliding_windowed_tiling/mu-split

# Run inference
if [ "on" == "on" ]; then
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/mu-split/inference.py --dataset PAVIA_ATN --modality ActTub --lc_type DeepLC --sliding_window_flag
else
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/mu-split/inference.py --dataset PAVIA_ATN --modality ActTub --lc_type DeepLC
fi
