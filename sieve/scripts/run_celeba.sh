#!/bin/bash
set -e
cd "$(dirname "$0")/.."

SEED=${SEED:-1}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU"
ROOT_DIR=${ROOT_DIR:-./data}
python run_exp.py \
  --method sieve-erm --dataset celeba \
  --gpu 0 --seed "$SEED" \
  --target_name Blond_Hair --confounder_names Male \
  --root_dir "$ROOT_DIR" \
  --lr 1e-3 --wd 1e-4 --batch_size 128 --erm_epochs 30 \
  --augment_data --balanced_class --balanced_distance \
  --use_val_data --use_val_data_for_selection \
  --n_confusing 200 --iterations 20 \
  --high_threshold_spurious_ratio 0.8 --high_threshold_non_spurious_ratio 0.8 \
  --selected_examples_weight 1.0 --weight_decay_rate 0 \
  --val_mode use_our_metric \
  --remove_selected_train
