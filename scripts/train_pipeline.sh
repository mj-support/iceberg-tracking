#!/bin/bash
#SBATCH --job-name=iceberg_train
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

# Execute following installation command only once to enable GPU based computation with CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

 # Training requires GPU processing
python run_pipeline.py train-embedding dataset=hill/train
python run_pipeline.py train-detection dataset=hill/train

# Tracking, evaluation, visualization and if applicable detection does not require GPU processing
# You could run the following commands on your local machine instead
python run_pipeline.py detect dataset=hill/test
python run_pipeline.py track dataset=hill/test
python run_pipeline.py eval dataset=hill/test
