#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --output=logs/build_dataset_%j.out
#SBATCH --error=logs/build_dataset_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

./data/src/build_dataset.sh --difficulty normal --folder ese_balanced --notes don,ka,bigDon,bigKa --batch-size 500 --ratio 0.25