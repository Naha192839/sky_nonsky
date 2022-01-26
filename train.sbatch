#!/bin/bash
#SBATCH --job-name sky_nonsky_classification
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
# singularity exec --nv trial.sif python ResNet50.py
singularity exec --nv trial.sif python top_cnn.py
# singularity exec --nv trial.sif python bottom_cnn.py
# singularity exec --nv trial.sif python vgg16.py
# singularity exec --nv trial.sif python 3C-CNN.py
# singularity exec --nv trial.sif python score.py
# singularity exec --nv trial.sif python ResNet50_kfold.py
# singularity exec --nv trial.sif python 3C-CNN_score.py
# singularity exec --nv trial.sif python gradcam_check.py
date