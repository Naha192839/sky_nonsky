#!/bin/bash
#SBATCH --job-name sky_nonsky_classification
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv trial.sif python ResNet50.py
date