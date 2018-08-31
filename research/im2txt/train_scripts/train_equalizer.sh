#!/bin/bash

python im2txt/train.py \
  --init_from="${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --blocked_input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/equalizer/train" \
  --train_inception=true \
  --blocked_image=true \
  --blocked_loss_weight=10 \
  --blocked_image_ce \
  --blocked_image_ce_weight=1 \
  --confusion_word_non_blocked \
  --confusion_word_non_blocked_weight=1 \
  --confusion_word_non_blocked_type="quotient" \
  --number_of_steps=1500000 \
  --batch_size=8
