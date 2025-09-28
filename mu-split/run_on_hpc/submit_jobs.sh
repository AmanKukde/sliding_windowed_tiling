#!/bin/bash
set -e

# ------------------------
# Job sweep parameters
# ------------------------
DATASETS=("PAVIA_ATN" "Hagen")
LC_TYPES=("LeanLC" "DeepLC")
SLIDING=("off" "on")

# Dataset-specific modalities
declare -A MODALITIES
MODALITIES["PAVIA_ATN"]="ActTub ActNuc TubNuc"
MODALITIES["HAGEN"]="MitoVsAct"

# HPC settings
PARTITION="gpuq"
GPUS=1
MEM="64GB"
CPUS=4
TIME="24:00:00"

# Paths (absolute)
PROJECT_DIR="/home/aman.kukde/sliding_windowed_tiling/mu-split"
PYTHON_BIN="/scratch/aman.kukde/conda/envs/msr/bin/python3.10"
SCRIPT="${PROJECT_DIR}/inference.py"
LOGDIR="${PROJECT_DIR}/logs"

mkdir -p "${LOGDIR}"

# ------------------------
# Job submission loop
# ------------------------
for dataset in "${DATASETS[@]}"; do
  for modality in ${MODALITIES[$dataset]}; do
    for lc in "${LC_TYPES[@]}"; do
      for slide in "${SLIDING[@]}"; do

        # job name + log file
        JOBNAME="inf_${dataset}_${modality}_${lc}_slide${slide}"
        LOGFILE="${LOGDIR}/${JOBNAME}.log"

        # sbatch script filename
        SBFILE="sbatch_${JOBNAME}.sh"

        # ------------------------
        # Write SBATCH file
        # ------------------------
        cat > $SBFILE <<EOL
#!/bin/bash
#SBATCH --mail-type=NONE
#SBATCH --output=${LOGFILE}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --gres=gpu:${GPUS}
#SBATCH --mem=${MEM}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --job-name=${JOBNAME}
#SBATCH --time=${TIME}

cd ${PROJECT_DIR}

# Run inference
if [ "${slide}" == "on" ]; then
  ${PYTHON_BIN} ${SCRIPT} --dataset ${dataset} --modality ${modality} --lc_type ${lc} --sliding_window_flag
else
  ${PYTHON_BIN} ${SCRIPT} --dataset ${dataset} --modality ${modality} --lc_type ${lc}
fi
EOL

        # ------------------------
        # Submit to SLURM
        # ------------------------
        echo "Submitting job: ${JOBNAME}"
        sbatch $SBFILE

      done
    done
  done
done
