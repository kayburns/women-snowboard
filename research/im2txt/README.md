# Women also Snowboard: Overcoming Bias in Captioning Models 

This repository contains everything necessary to replicate the results in our [2018 ECCV paper](https://arxiv.org/abs/1803.09797). The captioning model (most of the code) was built off of the Tensorflow [implementation](https://github.com/tensorflow/models/tree/master/research/im2txt). Thank you to the original author @cshallue.

## Getting Started

### Install the required packages.

- Tensorflow v1.0 or greater
- NumPy
- nltk, punkt (nltk.download('punkt'))
- pattern
- unzip

Or see the [`requirements.txt`](../../requirements.txt) file.

### Prepare the data.

To skip training, use our [pretrained models](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip) or the [captions](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip) themselves. Extract them both under "caption-bias/research/im2txt/".

We expect mscoco in the directory `caption-bias/research/im2txt/data/mscoco`. To train the model you will need to provide training data in native TFRecord format. Code is available [here](im2txt/data/download_and_preprocess_mscoco.sh) and detailed [instructions](https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data) about downloading and preprocessing the data are available in the original repo.

We run our experiments on the "Bias split" defined in [Men Also Like Shopping (Zhao et. al.)](https://github.com/uclanlp/reducingbias.git). It can be downloaded as follows (note: this is the data folder in the higher level im2txt directory):

```
cd ./data/
mkdir bias_splits
cd bias_splits/
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data
```

We also construct a ["Balanced split"](data/balanced_split/), where we randomly choose 500 images with women and 500 images with men from the "Bias split".

The Appearance Confusion Loss requires masked images. To create masks, please see the code for [creating masked images](scripts/SegmentationMasks.ipynb).

Our experiments fine tune standard im2txt on the COCO Bias split. Please see the code for [storing data as tfrecord files](im2txt/data/build_scripts/build_mscoco_blocked_data.py). The link provided also loads blocked images into the tfrecords, so you will need to specify the location of the blocked images.

In order to evaluate the GradCam/Saliency maps with the pointing game, we additionally save binary person masks. Please, see the code [here](im2txt/save_coco_person_segmentations.py).

## Training Models
Training scripts are provided [here](train_scripts/).

## Generating GradCam and Saliency maps

Below we provide example commands for computing GradCam and Saliency maps for a given model (note CHECKPOINT_PATH, MODEL_NAME and JSON_PATH) and a given set of images (note IMG_PATH).

### GradCam maps
Example commands for using ground-truth captions (as reported in the paper):
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_gradcam_test_gt/"
CHECKPOINT_PATH="./final_weights_eccv2018/baseline_ft/train/model.ckpt-1500000"
MODEL_NAME="baseline_ft"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_gradcam_with_gt.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --model_name=${MODEL_NAME} --img_path=${IMG_PATH} --save_path=${SAVE_PATH}
```

Example commands for using the predicted captions:
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_gradcam_test_pred/"
CHECKPOINT_PATH="./final_weights_eccv2018/baseline_ft/train/model.ckpt-1500000"
JSON_PATH="./final_captions_eccv2018/baseline_ft.json"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_gradcam.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --json_path=${JSON_PATH} --img_path=${IMG_PATH} --save_path=${SAVE_PATH}
```

We thank @PAIR-code for providing the GradCam [implementation](https://github.com/PAIR-code/saliency), which we [include](gradcam) in our repository.

### Saliency maps
Example commands to generate Saliency maps using ground-truth captions (as reported in the paper):
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_saliency_test_gt/"
CHECKPOINT_PATH="./final_weights_eccv2018/baseline_ft/train/model.ckpt-1500000"
MODEL_NAME="baseline_ft"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_saliency_with_gt.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --model_name=${MODEL_NAME} --img_path=${IMG_PATH} --save_path=${SAVE_PATH} --mask_size=32
```

## Running Analysis on Generated COCO Captions
Results from the paper can be recreated with [this](data_analysis/eccv_2018_results.py) script. You can generate all of the numbers from the tables and figures by running:
```
python data_analysis/eccv_2018_results.py --experiments all
```
Sentence scores can be generated by running `python data_analysis/data_analysis_sentence.py`.
Object correlations can be generated by running `python object_correlation.py` from within the data_analysis folder.

## TODO
https://docs.google.com/spreadsheets/d/1x1CYyvOO0WOcV2ksqZL3O5VxXkw2ns8MxY7aURjS7Oc/edit#gid=333766892

