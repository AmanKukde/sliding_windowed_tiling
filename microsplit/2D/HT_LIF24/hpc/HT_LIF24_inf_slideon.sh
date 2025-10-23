#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/microsplit-runs/HT_LIF24/logs/HT_LIF24_slideon.log
#SBATCH --nodes=1
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dgx
#SBATCH --gres=gpu:71gb:1
#SBATCH --job-name=DGX_MSR-notebooks-2D-HT_LIF24_slideon
#SBATCH --time=36:00:00

source ~/.bashrc
cd /group/jug/aman/microsplit_runs_22Oct25/HT_LIF24
conda activate msr
# Run inference
/scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/microsplit/2D/HT_LIF24/inference_lif24.py  --results_root /group/jug/aman/microsplit_runs_22Oct25/ --batch_size 256 --sliding_window_flag --stitch_only