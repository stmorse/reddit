#!/bin/tcsh
#SBATCH --job-name=LE_2
#SBATCH -N 1 -n 1
#SBATCH -t 180:00
#SBATCH --gpus=1

# load the anaconda module
module load anaconda3/2023.09

# activate your environment
conda activate torch-tik-env

# ensure we're in the correct directory
cd ~/projects/reddit

# run the script in this directory and save outputs to file
# python cluster1.py > output.out
python -u load_encode.py > output.out

# print something to shell as confirmation
echo "Complete"
