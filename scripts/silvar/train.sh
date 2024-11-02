#!/bin/bash

#SBATCH --job-name=train    
#SBATCH --output=gpu_output/Train-JOB_ID_%j-%N.log 

#SBATCH --nodes=1                 
#SBATCH --ntasks-per-node=1       
#SBATCH --mem=100G
#SBATCH --partition=gpu2 --gres=gpu  

echo "Job ID: $SLURM_JOBID" 
echo "Node names: $SLURM_JOB_NODELIST"
echo "Notes: Training Silvar with LLama3.1"


torchrun --nproc_per_node 2 train.py \
      --cfg-path train_configs/train.yaml\
      --cfg-eval-path eval_configs/evaluate.yaml\
      --eval-dataset audio_val
