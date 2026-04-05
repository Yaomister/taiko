#!/bin/bash
#SBATCH --job-name=taiko_cnn
#SBATCH --output=logs/taiko_cnn_%j.out
#SBATCH --error=logs/taiko_cnn_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

python model/training.py \
    --data_dir data/preprocessed/exports/ese \
    --out test_model/test.pth \
    --epochs 20
