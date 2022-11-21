#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --gres=gpu:1 --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml --mem=16GB
#SBATCH --time=0-07:00:00
#SBATCH --output=experiments-longformer-ecthr
#SBATCH --job-name=lexglue-longformer-ecthr
#SBATCH --mail-user=daniel.saggau@gmail.com
#SBATCH --mail-type=ALL

#your script, in this case: write the hostname and the ids of the chosen gpus.
#hostname
echo $CUDA_VISIBLE_DEVICES
export PYTHONPATH=.
wandb login fd6f7deb3126d40be9abf77ee753bf45f00e2a9a
python IR_LDC/model/ECTHR/ecthr_clean.py \
        --output_dir  logs/echtr_b \
        --overwrite_output_dir \
        --load_best_model_at_end \
        --metric_for_best_model micro-f1 \
        --greater_is_better 1 \
        --evaluation_strategy  epoch \
        --save_strategy epoch \
        --save_total_limit  5 \
        --num_train_epochs  20 \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 6 \
        --per_device_eval_batch_size  6 \
        --fp16 \
