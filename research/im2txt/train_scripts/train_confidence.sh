#!/bin/bash

#INIT_MODEL_DIR: final_weights_eccv2018/inception_checkpoint
#BLOCKED_MSCOCO_DIR: im2txt/data/bias_and_blocked
#INCEPTION_CHECKPOINT: final_weights_eccv2018/inception_checkpoint
#MODEL_DIR: where you would like to save your trained models

python im2txt/train.py \
  --init_from="${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/inception/train" \
  --train_inception=true \
  --blocked_image=false \
  --confusion_word_non_blocked \
  --confusion_word_non_blocked_weight=1 \
  --confusion_word_non_blocked_type="quotient" \
  --number_of_steps=1500000 \
  --batch_size=8
