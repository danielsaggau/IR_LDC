#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --gres=gpu:1 --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml --mem=16GB
#SBATCH --time=0-12:00:00
#SBATCH --e scotus.err
#SBATCH --output=experiments-feature_extraction
#SBATCH --job-name=pipeline_featureextraction
#SBATCH --mail-user=daniel.saggau@gmail.com
#SBATCH --mail-type=ALL

python pipeline_features.py 
