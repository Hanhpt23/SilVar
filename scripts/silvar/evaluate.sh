#!/bin/bash

#SBATCH --job-name=eval       
#SBATCH --output=gpu_output/Eval-JOB_ID_%j-%N.log 

#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1        
#SBATCH --mem=100G
#SBATCH --partition=gpu2 --gres=gpu  

echo "Job ID: $SLURM_JOBID" 
echo "Node names: $SLURM_JOB_NODELIST"
echo "Notes: Evaluating with SilVar"

torchrun --nproc_per_node 2 evaluate.py \
      --cfg-path eval_configs/evaluate.yaml\
      --eval-dataset audio_val
