#!/bin/bash
#SBATCH --account=def-foutsekh
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=1-00:00
#SBATCH --mail-user=a.majdinasab@hotmail.com
#SBATCH --mail-type=ALL

# Set the working directory and run number as variables
working_dir="/home/vamaj/scratch/TWMC"
run_num=0
sorted="True"

source /home/vamaj/twmc/bin/activate
cd /home/vamaj/scratch/TWMC

# Loop to keep running the script if exit code is 2
while true; do
    python /home/vamaj/scratch/TWMC/src/main_block.py --run_num "$run_num" --working_dir "$working_dir"
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "Script executed successfully."
        break
    elif [ $exit_code -eq 1 ]; then
        echo "Received exit code 1. Performing another action..."
        break
    elif [ $exit_code -eq 2 ]; then
        echo "CUDA error encountered, rerunning script..."
    else
        echo "Unknown exit code $exit_code. Exiting the loop."
        break
    fi
done
