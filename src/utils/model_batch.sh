#!/bin/bash
#SBATCH --job-name=get_iceberg_model
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Start des Trainings
python3 model_train.py
python3 model_inference.py