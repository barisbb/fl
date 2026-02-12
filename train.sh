#!/bin/bash
#SBATCH --job-name=distill       
#SBATCH --output=try1.log        
#SBATCH --error=error.log                       
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1          



python train.py
