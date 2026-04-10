#!/bin/bash
#SBATCH --job-name=build_don_ka
#SBATCH --output=logs/build_don_ka_%j.out
#SBATCH --error=logs/build_don_ka_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

./data/src/build_dataset.sh -d normal -f ese_don_ka -n don,ka -b 500 -r 0.25