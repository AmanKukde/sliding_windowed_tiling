#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=/group/jug/aman/microsplit_HT_H24_27Oct25/logs/HT_H24_3D_sw_%x_%j.log
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=dgx
#SBATCH --gres=gpu:71gb:1
#SBATCH --job-name=MSR-notebooks-3D-HT_H24_sw
#SBATCH --time=36:00:00
source ~/.bashrc
cd /group/jug/aman/microsplit_HT_H24_27Oct25
conda activate msr
# Run inference
/scratch/aman.kukde/conda/envs/msr/bin/python3.10 /home/aman.kukde/sliding_windowed_tiling/microsplit/3D/HT_H24/inference_HT_H24/inference_HT_H24.py --batch_size 128
