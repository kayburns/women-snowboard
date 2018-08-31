#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

export MSCOCO_DIR='/home/lisaanne/lev/data1/caption_bias/models/research/im2txt/im2txt/data/preprocessed' 
export BLOCKED_MSCOCO_DIR='/home/lisaanne/lev/data1/caption_bias/models/research/im2txt/im2txt/data/blocked_subset_data_avg' #blocked data
export INCEPTION_CHECKPOINT='/home/lisaanne//lev/data1/caption_bias/models/research/im2txt/im2txt/data/inception_v3.ckpt'
export MODEL_DIR='/data/lisaanne/fairness/checkpoints/blocked_image_blockedLW100/'

python im2txt/train.py \
  --input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --blocked_input_file_pattern="${BLOCKED_MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --blocked_image=true \
  --number_of_steps=1000000
