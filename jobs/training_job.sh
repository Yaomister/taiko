#!/bin/bash
#SBATCH --job-name=bigka
#SBATCH --output=logs/bigka_cnn_%j.out
#SBATCH --error=logs/bigka_cnn_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

python model/training.py \
    --data_dir data/preprocessed/exports/ese_bigka \
    --out binary/bigka.pth \
    --epochs 100 \
    --lr 0.003
