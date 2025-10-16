#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/home/aman.kukde/sliding_windowed_tiling/mu-split/logs/2025-10-13_13-49-31/inf_Hagen_MitoVsAct_DeepLC_slideon.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=inf_Hagen_MitoVsAct_DeepLC_slideon
#SBATCH --time=36:00:00

cd /home/aman.kukde/sliding_windowed_tiling/mu-split

# Run inference
if [ "on" == "on" ]; then
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10  --dataset Hagen --modality MitoVsAct --lc_type DeepLC --sliding_window_flag --results_root /group/jug/aman/usplit_13Oct25/ --batch_size 64

else
  /scratch/aman.kukde/conda/envs/msr/bin/python3.10  --dataset Hagen --modality MitoVsAct --lc_type DeepLC --results_root /group/jug/aman/usplit_13Oct25/ --batch_size 64
fi
