# Women also Snowboard: Overcoming Bias in Captioning Models 

This repository contains everything necessary to replicate the results in our [2018 ECCV paper](https://arxiv.org/abs/1803.09797). To skip training, use our [pretrained models](/todo) or the [captions](/todo) themselves. The captioning model (most of the code) was built off of the Tensorflow [implementation](https://github.com/tensorflow/models/tree/master/research/im2txt). Thank you to the original author @cshallue.

## Getting Started

### Install the required packages.

- Tensorflow v1.0 
- NumPy v??
- nltk
- unzip

Or see the [`requirements.txt`](??) file.

### Prepare the training data.

To train the model you will need to provide training data in native TFRecord format. Code is available [here](im2txt/im2txt/data/download_and_preprocess_mscoco.sh) and detailed [instructions](https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data) about downloading and preprocessing the data are available in the original repo.

We run our experiments on the "bias split" defined in [Men Also Like Shopping (Zhao et. al.)](https://github.com/uclanlp/reducingbias.git). It can be downloaded as follows:

```
cd im2txt/data/bias_splits
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data
```

The Appearance Confusion Loss requires masked images. To create masks, please see the code for [creating masked images](todo).

Our experiments fine tune standard im2txt on the bias coco split. Please see the code for [storing data as tfrecord files](im2txt/im2txt/data/build_scripts/build_mscoco_blocked_data.py). The link provided also loads blocked images into the tfrecords, so you will need to specify the location of the blocked images.

## Training Models
Training scripts are provided [here](im2txt/train_scripts/).

## Running Analysis on Generated COCO Captions
To create "confident subset", see `scripts/make_confident_set.py`
For amplification bias of different nouns, see `find_bigrams.py`
Any result from the paper can be recreated with [this](im2txt/data_analysis/eccv_results_2018.py) script.

## Generating saliency results (TODO: incorporate with other analysis)
```
python im2txt/run_inference_with_saliency_with_gt.py --checkpoint_path=./model/DESIRED_MODEL/train --vocab_file=./data/word_counts.txt --dump_file=./FILE_NAME --model_name=./MODEL_NAME --img_path=im2txt/data/val_dataset.txt --save_path=SAVE_PATH/
```

## TODO
Kaylee
- [ ] test: all scripts to download and tfrecord-ify all data (normal, bias, blocked), inference with saved checkpoints,  (need to rerun but data1 and data2 are full)
- [ ] include yaml file to set up environment + setup instructions
- [ ] send Lisa all weights

Anja
- [ ] code to run GradCam. should print results when `table_3_main` or `table_2_supp` of the eccv results [script](im2txt/data_analysis/eccv_results_2018.py) is called.

Lisa
- [ ] add training scripts for balanced and upweight baselines to training scripts folder
- [ ] Code to create blocked images
