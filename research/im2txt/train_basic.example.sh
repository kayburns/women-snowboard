#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"

export MSCOCO_DIR='/data1/caption_bias/models/research/im2txt/im2txt/data/preprocessed'
#export MSCOCO_DIR='/home/lisaanne/lev/data2/kaylee/caption_bias/models/research/im2txt/im2txt/data/blocked_data' #blocked data
export INCEPTION_CHECKPOINT='/data1/caption_bias/models/research/im2txt/im2txt/data/inception_v3.ckpt'
export MODEL_DIR='tmp/'

python im2txt/train.py \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
