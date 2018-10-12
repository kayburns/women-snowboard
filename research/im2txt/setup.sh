#!/bin/bash

#optional input to make softlink of mscoco dir
mscoco_dir=$1

#get bias splits
cd ./data/
mkdir -p bias_splits
cd bias_splits/
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data

#get pretrained models
cd ../..
wget https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip
unzip final_weights_eccv2018.zip
rm final_weights_eccv2018.zip

#get inception network
INCEPTION_DIR=final_weights_eccv2018/inception_checkpoint
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"

#get precomputed captions
wget https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip
unzip final_captions_eccv2018.zip
rm final_captions_eccv2018.zip

if [[ $# -eq 0 ]] ; then
    echo 'Did not indicate MSCOCO path; expect MSCOCO to be at data/mscoco'
else
    ln -s $mscoco_dir data/mscoco    
fi
