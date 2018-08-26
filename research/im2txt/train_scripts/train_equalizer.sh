#!/bin/bash

export CUDA_VISIBLE_DEVICES="4,5"

export MSCOCO_DIR='/data1/caption_bias/models/research/im2txt/im2txt/data/preprocessed' 
export BLOCKED_MSCOCO_DIR='/data1/caption_bias/models/research/im2txt/im2txt/data/blocked_subset_data_avg' #blocked data
export INCEPTION_CHECKPOINT='/data1/caption_bias/models/research/im2txt/im2txt/data/inception_v3.ckpt'
export MODEL_DIR='/data2/kaylee/caption_bias/models/research/im2txt/model/quotient_confusion'

python im2txt/train.py \
  --init_from="/data1/caption_bias/models/research/im2txt/model/train/model.ckpt-1000000" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --blocked_input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
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
