#!/bin/bash
#SBATCH --account=def-foutsekh
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00
#SBATCH --mail-user=a.majdinasab@hotmail.com
#SBATCH --mail-type=ALL
source /home/vamaj/twmc/bin/activate
cd /home/vamaj/jenova
python /home/vamaj/jenova/src/main.py