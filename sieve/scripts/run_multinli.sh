#!/bin/bash
set -e
cd "$(dirname "$0")/.."

SEED=${SEED:-1}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT_DIR=${ROOT_DIR:-./data/multinli}
python run_exp.py \
  --method sieve-erm --dataset multinli \
  --gpu 0 --seed "$SEED" \
  --target_name gold_label_random --confounder_names sentence2_has_negation \
  --root_dir "$ROOT_DIR" \
  --model_type bert \
  --lr 2e-5 --wd 0 --batch_size 32 --erm_epochs 3 \
  --use_val_data --use_val_data_for_selection \
  --n_confusing 200 --iterations 3 \
  --high_threshold_spurious_ratio 0.8 --high_threshold_non_spurious_ratio 0.8 \
  --selected_examples_weight 1.0 --weight_decay_rate 0.2 \
  --val_mode use_our_metric \
  --remove_selected_train
