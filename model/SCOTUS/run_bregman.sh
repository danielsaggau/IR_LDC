#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --gres=gpu:1 --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml --mem=16GB
#SBATCH --time=0-07:00:00
#SBATCH --error scotus.err
#SBATCH --output=experiments-bregman
#SBATCH --job-name=bregman-scotus-10k-s1
#SBATCH --mail-user=daniel.saggau@gmail.com
#SBATCH --mail-type=ALL

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')" 
python simcse_bregman_scotus.py
