#!/bin/bash
#SBATCH --job-name=train_three
#SBATCH --output=logs/train_three_%j.out
#SBATCH --error=logs/train_three_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

python model/training.py \
    --data_dir data/preprocessed/exports/ese_don_ka_bigDon \
    --out models/three.pth \
    --epochs 100 \
    --lr 0.003