#!/bin/bash

python im2txt/train.py \
  --init_from="${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --batch_size=8 \
  --number_of_steps=1500000 \
  --loss_weight_value=10
