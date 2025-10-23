#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/microsplit-runs/HT_H23B/logs/HT_H23B.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=MSR-notebooks-2D-HT_H23B
#SBATCH --time=36:00:00

cd /group/jug/aman/microsplit-runs/HT_H23B
conda source msr

# Run inference
/scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/microsplit/run-jupyter-as-script/run_notebooks.py --notebook "/home/aman.kukde/sliding_windowed_tiling/microsplit/2D/HT_H23B/02_predict_sw.ipynb" --outputdir "/group/jug/aman/microsplit-runs/HT_H23B/notebooks/"