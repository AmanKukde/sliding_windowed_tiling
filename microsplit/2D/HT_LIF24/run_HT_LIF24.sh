#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/microsplit-runs/HT_LIF24/logs/HT_LIF24.log
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=MSR-notebooks-2D-HT_LIF24
#SBATCH --time=36:00:00
conda source microsplit
cd /group/jug/aman/microsplit-runs/HT_LIF24


# Run inference
/scratch/aman.kukde/conda/envs/microsplit/bin/python3.9 /home/aman.kukde/sliding_windowed_tiling/microsplit/run-jupyter-as-script/run_notebooks.py --notebook "/home/aman.kukde/sliding_windowed_tiling/MicroSplit-reproducibility/examples/2D/HT_LIF24/02_predict.ipynb" --outputdir "/group/jug/aman/microsplit-runs/HT_LIF24/notebooks/"