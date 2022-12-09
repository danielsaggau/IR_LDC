#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --gres=gpu:1 --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml --mem=16GB
#SBATCH --time=0-07:00:00
#SBATCH --e scotus.err
#SBATCH --output=bregman_classification
#SBATCH --job-name=bregmanclassification
#SBATCH --mail-user=daniel.saggau@gmail.com
#SBATCH --mail-type=ALL
wandb login fd6f7deb3126d40be9abf77ee753bf45f00e2a9a
python bregman_scotus_classification.py \
    --output_dir logs/output_1 \
    --model_name 'danielsaggau/bregman_scotus_k8_ep10' \
    --model_type 'mean' \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --num_train_epochs 10 \
    --weight_decay 0.01 \
    --fp16 \
    --gradient_accumulation_steps 1 \
    --metric_for_best_model "f1-micro" \
    --greater_is_better 1 \
    --report_to 'wandb'
