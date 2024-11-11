#!/bin/tcsh
#SBATCH --job-name=sbert
#SBATCH -N 1 -n 1
#SBATCH -t 60:00
#SBATCH --gpus=1

# load and activate conda
module load anaconda3/2023.09
conda activate torch-tik-env

# ensure we're in the correct directory
cd ~/projects/reddit/embed

# run the script in this directory and save outputs to file
python -u embedder.py > out.log

# print something to shell as confirmation
echo "Complete"
