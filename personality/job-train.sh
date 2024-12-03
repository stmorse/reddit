#!/bin/tcsh
#SBATCH --job-name=hf_p
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gpus=1

# load and activate conda
module load anaconda3/2023.09
conda activate torch-tik-env

# ensure we're in the correct directory
cd ~/projects/reddit/personality

# run the script in this directory and save outputs to file
python -u train.py > out.log

# print something to shell as confirmation
echo "Complete"
