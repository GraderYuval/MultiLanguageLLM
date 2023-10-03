#!/bin/bash

#SBATCH --job-name=xnli_task_%A_%a
#SBATCH --output=outputs/%A/output_%a.out # redirect stdout
#SBATCH --error=errors/%A/error_%a.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --array=0-14

# Define your list of languages
languages=("ar" "bg" "de" "el" "en" "es" "fr" "hi" "ru" "sw" "th" "tr" "ur" "vi" "zh")

# Get the language for this job
language=${languages[$SLURM_ARRAY_TASK_ID]}

# Run your command for the specific language
echo "Processing language: $language"
python -m tasks.xnli_task --language "$language"
