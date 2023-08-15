#! /bin/sh

#SBATCH --job-name=base_task
#SBATCH --output=base_task/awesome.out # redirect stdout
#SBATCH --error=base_task/awesome.err # redirect stderr
#SBATCH --partition=studentbatch # (see resources section)
#SBATCH --time=1 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total


python3 -m MultiLanguageLLM.tasks.base_task language=ar
