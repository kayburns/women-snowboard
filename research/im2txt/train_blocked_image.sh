#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"

#export MSCOCO_DIR='/home/lisaanne/lev/data1/caption_bias/models/research/im2txt/im2txt/data/preprocessed' #orig data
export MSCOCO_DIR='/home/lisaanne/lev/data2/kaylee/caption_bias/models/research/im2txt/im2txt/data/blocked_data' #blocked data
export INCEPTION_CHECKPOINT='/home/lisaanne//lev/data1/caption_bias/models/research/im2txt/im2txt/data/inception_v3.ckpt'
export MODEL_DIR='/data/lisaanne/fairness/checkpoints/baseline_train_debug/'
export INIT_FROM='/data/lisaanne/fairness/checkpoints/LW_train/train/checkpoint'

python im2txt/train.py \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --blocked_image=true \
  --number_of_steps=1000000 \
  --init_from /data/lisaanne/fairness/checkpoints/LW_train/train/model.ckpt-658278
