#!/bin/bash

#INIT_MODEL_DIR: final_weights_eccv2018/inception_checkpoint
#WOMAN_MSCOCO_DIR: im2txt/data/woman
#WOMAN_MSCOCO_DIR: im2txt/data/man
#INCEPTION_CHECKPOINT: final_weights_eccv2018/inception_checkpoint
#MODEL: where you would like to save your trained models

python im2txt/train.py \
  --init_from=${INIT_MODEL_DIR}/train" \
  --input_file_pattern="${WOMAN_MSCOCO_DIR}/train-?????-of-00256" \
  --input_file_pattern2="${MAN_MSCOCO_DIR}/train-?????-of-00256" \
  --two_input_queues=true \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=true \
  --number_of_steps=1500000 \
  --batch_size=8
