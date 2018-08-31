#!/bin/bash

python im2txt/train.py \
  --init_from="${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/baseline/train" \
  --train_inception=True \
  --number_of_steps=1500000 \
  --batch_size=8
