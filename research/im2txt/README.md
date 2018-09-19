# Women also Snowboard: Overcoming Bias in Captioning Models 

This repository contains everything necessary to replicate the results in our 2018 ECCV paper ([arXiv](https://arxiv.org/abs/1803.09797) and [eccv](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lisa_Anne_Hendricks_Women_also_Snowboard_ECCV_2018_paper.pdf)). To skip training, use our [pretrained models](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip) or the [captions](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip) themselves. The captioning model (most of the code) was built off of the Tensorflow [implementation](https://github.com/tensorflow/models/tree/master/research/im2txt). Thank you to the original author @cshallue.

## Getting Started

### Prepare the training data.

To train the model you will need to provide training data in native TFRecord format. Code is available [here](im2txt/data/download_and_preprocess_mscoco.sh) and detailed [instructions](https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data) about downloading and preprocessing the data are available in the original repo.

We run our experiments on the "bias split" defined in [Men Also Like Shopping (Zhao et. al.)](https://github.com/uclanlp/reducingbias.git). It can be downloaded as follows:

```
cd im2txt/data/bias_splits
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data
```

The Appearance Confusion Loss requires masked images. To create masks, please see the code for [creating masked images](scripts/SegmentationMasks.ipynb).

Our experiments fine tune standard im2txt on the bias coco split. Please see the code for [storing data as tfrecord files](im2txt/data/build_scripts/build_mscoco_blocked_data.py). The link provided also loads blocked images into the tfrecords, so you will need to specify the location of the blocked images.

## Training Models
Training scripts are provided [here](train_scripts).

## Running Analysis on Generated COCO Captions
Any result from the paper can be recreated with [this](data_analysis/eccv_results_2018.py) script. You can generate all of the numbers from the tables and figures by running:
```
python data_analysis/eccv_results_2018.py --experiments all
```

To generate saliency maps for pointing game please run.
```
python im2txt/run_inference_with_saliency_with_gt.py --checkpoint_path=./model/DESIRED_MODEL/train --vocab_file=./data/word_counts.txt --dump_file=./FILE_NAME --model_name=./MODEL_NAME --img_path=im2txt/data/val_dataset.txt --save_path=SAVE_PATH/
```
