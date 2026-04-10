#!/bin/bash
#SBATCH --job-name=train_don_ka
#SBATCH --output=logs/train_don_ka_%j.out
#SBATCH --error=logs/train_don_ka_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4



source /home/yao.eric/venv/taiko/bin/activate


mkdir -p models


python model/training.py \
--data_dir data/preprocessed/exports/ese_bigka \
--out binary/bigka.pth \
--epochs 100 \
--lr 0.003