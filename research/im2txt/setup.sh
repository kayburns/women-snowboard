#!/bin/bash

cd ./data/
mkdir bias_splits
cd bias_splits/
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data

#get pretrained models
cd ../..
wget https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip
unzip final_weights_eccv2018.zip
rm final_weights_eccv2018.zip

wget https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip
unzip final_captions_eccv2018.zip
rm final_captions_eccv2018.zip
