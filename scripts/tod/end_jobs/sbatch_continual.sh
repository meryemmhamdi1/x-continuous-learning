#!/usr/bin/env bash

#SBATCH --partition=isi
#SBATCH --mem=100g
#SBATCH --time=7-24:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --output=jobs/R-%x.out.%j
#SBATCH --error=jobs/R-%x.err.%j
#SBATCH --export=NONE

# rest of the script.

module purge
module load conda

conda init bash
source ~/.bashrc

conda activate zsl_nlu
sh scripts/tod/spaced_repetition/continual.sh $1 $2 $3 $4 $5 $6 $7 $8 $9

