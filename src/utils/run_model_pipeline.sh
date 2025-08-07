#!/bin/bash
#SBATCH --job-name=iceberg_detector
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=clara
#SBATCH --gpus=v100
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load CUDA module
module load CUDA/12.6.0

# Activate your conda environment
source ~/.bashrc
conda activate iceberg-tracking

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Uncomment one (or both) of those scripts to run them
#python ../embedding.py
#python ../detection.py