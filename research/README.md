# TensorFlow Research Models

## Training

1) To train with loss weight: train_loss_weight.sh
2) To train with blocked image: train_blocked_image.sh

## New flags:

Pattern for the blocked image tfrecord:
--blocked_input_file_pattern 

Loss weight to be applied to the words "man" and "woman":
--loss_weight_value

Whether or not to train with blocked image:
--blocked_image

Weight loss for blocked image and normal image:
--blocked_loss_weight 
