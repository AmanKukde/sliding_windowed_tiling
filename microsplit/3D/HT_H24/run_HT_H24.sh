#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/microsplit_runs_22Oct25/HT_H24/logs/HT_H24_3D.log
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dgx
#SBATCH --gres=gpu:35gb:1
#SBATCH --job-name=MSR-notebooks-3D-HT_H24
#SBATCH --time=36:00:00
source ~/.bashrc
cd /group/jug/aman/microsplit_runs_22Oct25/HT_H24
conda activate microsplit
# Run inference
/scratch/aman.kukde/conda/envs/microsplit/bin/python3.9 /home/aman.kukde/sliding_windowed_tiling/microsplit/run-jupyter-as-script/run_notebooks.py --notebook "/home/aman.kukde/sliding_windowed_tiling/microsplit/3D/HT_H24/02_predict.ipynb" --outputdir "/group/jug/aman/microsplit_runs_22Oct25/HT_H24/notebooks/"
