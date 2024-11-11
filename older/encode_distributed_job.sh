#!/bin/tcsh
#SBATCH --job-name=sbert_d
#SBATCH --nodes=1                 
#SBATCH --ntasks=2                # needs to match ppn in torchrun
#SBATCH --gpus=2                  # or --gres=gpu:2
#SBATCH --time=00:45:00
#SBATCH --output=out.log

module load anaconda3/2023.09

conda activate torch-tik-env

cd ~/projects/reddit

# omitting `srun` because we're just on one node
torchrun --nproc_per_node=2 encode_distributed.py
