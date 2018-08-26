#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

export MSCOCO_DIR='/data1/caption_bias/models/research/im2txt/im2txt/data/preprocessed'
export INCEPTION_CHECKPOINT='/data1/caption_bias/models/research/im2txt/im2txt/data/inception_v3.ckpt'
export MODEL_DIR='/data2/kaylee/caption_bias/models/research/im2txt/model/train_fine_tune_incep_bias_split'
export BLOCKED_MSCOCO_DIR='/data1/caption_bias/models/research/im2txt/im2txt/data/blocked_subset_data_avg' #blocked data

python im2txt/train.py \
  --init_from="/data2/kaylee/caption_bias/models/research/im2txt/model/train_fine_tune_incep_bias_split/train/" \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=True \
  --number_of_steps=1500000 \
  --batch_size=8
