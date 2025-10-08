#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/sliding_windowed_tiling/mu-split/logs/2025-10-08_11-00-27/inf_PAVIA_ATN_ActNuc_LeanLC_slideon.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=inf_PAVIA_ATN_ActNuc_LeanLC_slideon
#SBATCH --time=36:00:00

cd /home/aman.kukde/sliding_windowed_tiling/mu-split

# Run inference
if [ "on" == "on" ]; then
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/mu-split/inference.py --dataset PAVIA_ATN --modality ActNuc --lc_type LeanLC --sliding_window_flag --results_root /group/jug/aman/Results_06Oct25/Results_usplit_64 --batch_size 64 --stitch_only

else
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/mu-split/inference.py --dataset PAVIA_ATN --modality ActNuc --lc_type LeanLC --results_root /group/jug/aman/Results_06Oct25/Results_usplit_64 --batch_size 64
fi
