#!/bin/bash
# Wrapper script to SSH into HPC and submit all jobs

# SSH into HPC using login shell so sbatch is available
ssh hpc "bash -l -c 'cd ~/sliding_windowed_tiling/mu-split/run_on_hpc && bash submit_jobs.sh'"
