#!/bin/bash
#SBATCH --account=def-foutsekh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#sbatch --mem=32G
#SBATCH --mail-user=a.majdinasab@hotmail.com
#SBATCH --mail-type=ALL

# Set the working directory and run number as variables
working_dir="/home/vamaj/scratch/TraWiC"
module load arrow
source /home/vamaj/scratch/twmc/bin/activate
cd /home/vamaj/scratch/TraWiC
/home/vamaj/scratch/twmc/bin/python3.10 /home/vamaj/scratch/TraWiC/src/models/finetune.py

