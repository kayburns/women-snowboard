# Women also Snowboard: Overcoming Bias in Captioning Models 

This repository contains everything necessary to replicate the results in our 2018 ECCV paper ([arXiv](https://arxiv.org/abs/1803.09797) and [ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Lisa_Anne_Hendricks_Women_also_Snowboard_ECCV_2018_paper.pdf)). The captioning model was built off of the Tensorflow [implementation](https://github.com/tensorflow/models/tree/master/research/im2txt). Thank you to the original author @cshallue.

## Getting Started

### Install the required packages.

- Tensorflow v1.0 or greater
- NumPy
- Python 2
- nltk, punkt (nltk.download('punkt'))
- pattern
- unzip

Or see the [`requirements.txt`](../../requirements.txt) file.

### Prepare the data.

#### Download the Data

Our datasets are built off the [
MSCOCO dataset](http://cocodataset.org/#download).  Our code expects the MSCOCO dataset to be in `women-snowboard/research/im2txt/data/mscoco`.

We run our experiments on the "Bias split" defined in [Men Also Like Shopping (Zhao et. al.)](https://github.com/uclanlp/reducingbias.git) as well as a ["Balanced split"](data/balanced_split/) which we define.  Our "Balanced split" contain 500 randomly selected images with women, and 500 randomly selected images with men. 

You can use `./setup.sh` to download data, pre-trained models (including Inception-v3 which is the convolutional network base we use), and pre-extracted captions.  You can optionally provide a path to the MSCOCO dataset and the script will automatically create a softlink for the MSCOCO dataset in the "data" folder.  If you would like to do this run `.setup.sh PATH/TO/MSCOCO`.

To skip training, use our [pretrained models](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip) or the [captions](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip) themselves. These are expected to be under `women-snowboard/research/im2txt` and will be automatically downloaded with `setup.sh`.

#### Data Preprocessing: Creating Masked Images

The Appearance Confusion Loss requires images in which humans are blocked out to train.  Additionally, to evaluate GradCam/Saliency maps with the pointing game, we save binary person masks (everything except the person is masked out).  To create this, please see code to create [Segmentation Masks](scripts/SegmentationMasks.ipynb).

#### Data Preprocessing: TFRecord Files

To train the model you will need to provide training data in native TFRecord format. To train all the models you will need to construct three sets of TFRecord files: full MSCOCO for pretraining, the MSCOCO-Bias with blocked out images to finetune our models, and TFRecords for training the "Balanced" model.  

Create TFRecord files on the MSCOCO dataset:
```
python im2txt/data/build_scripts/build_mscoco_data.py \
 --train_image_dir=data/mscoco/images/train2014 \
 --val_image_dir=data/mscoco/images/val2014 \
 --train_captions_file=data/mscoco/annotations/captions_train2014.json \
 --val_captions_file=data/mscoco/annotations/captions_val2014.json \
 --output_dir=im2txt/data/mscoco_base \
 --word_counts_output_file="data/word_counts.txt" \
```

Create TFRecord files for the MSCOCO-Bias split, including blocked images:
```
python im2txt/data/build_scripts/build_mscoco_blocked_and_biased.py \
 --train_image_dir=data/mscoco/images/train2014 \
 --val_image_dir=data/mscoco/images/val2014 \
 --train_captions_file=data/mscoco/annotations/captions_train2014.json \
 --val_captions_file=data/mscoco/annotations/captions_val2014.json \
 --output_dir=im2txt/data/bias_and_blocked \
 --word_counts_output_file="data/word_counts.txt" \
 --blocked_dir="data/blocked_images_average/"
```

Training the "Balanced" model requires tfrecord files for each gender. To create these substitute GENDER with "man" and "woman":

```
python im2txt/data/build_scripts/build_mscoco_single_gender_blocked.py \
 --train_image_dir=data/mscoco/images/train2014 \
 --val_image_dir=data/mscoco/images/val2014 \
 --train_captions_file=data/mscoco/annotations/captions_train2014.json \
 --val_captions_file=data/mscoco/annotations/captions_val2014.json \
 --output_dir=im2txt/data/${GENDER} \
 --word_counts_output_file="data/word_counts.txt" \
 --blocked_dir="data/blocked_images_average/" \
 --gender="${GENDER}"
```

## Training Models

First train a model on the full MSCOCO dataset using the training script [here](train_scripts/train_base.sh).  Then you can train our models by fine-tuning on the MSCOCO-Bias set.  Please see all training scripts [here](train_scripts/).  

Each training script requires inputs including pretrained weights, TFRecord files, and which file to save to.  For example, train the [Baseline-FT](train_scripts/train_baseline_ft.sh) model as follows:

```
INIT_MODEL_DIR='final_weights_eccv2018/mscoco_base/train'
BLOCKED_MSCOCO_DIR='im2txt/data/bias_and_blocked'
INCEPTION_CHECKPOINT='final_weights_eccv2018/inception_checkpoint'
MODEL_DIR='/where/you/want/to/save/model'
./train_baseline_ft.sh
```

Double check the experiment bash script to see which variables should be set.

## Generating sentences.
Run the following from `women-snowboard/research/im2txt`.
```
python im2txt/run_inference.py \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file='./data/word_counts.txt' \
  --input_files='./data/mscoco/${SPLIT}/COCO_${SPLIT}_*.jpg'
  --dump_file='${DUMP_FILE}.json'
```


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
