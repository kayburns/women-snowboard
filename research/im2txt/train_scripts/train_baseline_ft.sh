#!/bin/bash

#Example inputs:
#INIT_MODEL_DIR: final_weights_eccv2018/inception_checkpoint
#BLOCKED_MSCOCO_DIR: im2txt/data/bias_and_blocked
#INCEPTION_CHECKPOINT: final_weights_eccv2018/inception_checkpoint
#MODEL: where you would like to save your trained models

python im2txt/train.py \
  --init_from="${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/baseline/train" \
  --train_inception=True \
  --number_of_steps=1500000 \
  --batch_size=8
