#!/bin/tcsh
#SBATCH --job-name=sbert
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --gpus=4

# load and activate conda
module load anaconda3/2023.09
conda activate torch-tik-env

# ensure we're in the correct directory
cd ~/projects/reddit/embed

# run the script in this directory and save outputs to file
python -u tmp_slurm.py > out_slurm.log

# print something to shell as confirmation
echo "Complete"
