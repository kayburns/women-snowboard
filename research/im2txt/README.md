# Women also Snowboard: Overcoming Bias in Captioning Models 

This repository contains everything necessary to replicate the results in our [2018 ECCV paper](https://arxiv.org/abs/1803.09797). To skip training, use our [pretrained models](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_weights_eccv2018.zip) or the [captions](https://people.eecs.berkeley.edu/~lisa_anne/snowboard_misc/final_captions_eccv2018.zip) themselves. The captioning model (most of the code) was built off of the Tensorflow [implementation](https://github.com/tensorflow/models/tree/master/research/im2txt). Thank you to the original author @cshallue.

## Getting Started

### Install the required packages.

- Tensorflow v1.0 
- NumPy v??
- nltk, punkt (>> import nltk  >> nltk.download('punkt'))
- pattern
- unzip

Or see the [`requirements.txt`](??) file.

### Prepare the data.

To train the model you will need to provide training data in native TFRecord format. Code is available [here](im2txt/data/download_and_preprocess_mscoco.sh) and detailed [instructions](https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data) about downloading and preprocessing the data are available in the original repo.

We run our experiments on the "bias split" defined in [Men Also Like Shopping (Zhao et. al.)](https://github.com/uclanlp/reducingbias.git). It can be downloaded as follows:

```
cd ./im2txt/data/
mkdir bias_splits
cd bias_splits/
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/dev.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/train.data
curl -O https://raw.githubusercontent.com/uclanlp/reducingbias/master/data/COCO/test.data
```

We also construct a "balanced split", where we randomly choose 500 images with women and 500 images with men from the "bias split". We include it under "./im2txt/data/balanced_split/".


The Appearance Confusion Loss requires masked images. To create masks, please see the code for [creating masked images](scripts/SegmentationMasks.ipynb).

Our experiments fine tune standard im2txt on the bias coco split. Please see the code for [storing data as tfrecord files](im2txt/data/build_scripts/build_mscoco_blocked_data.py). The link provided also loads blocked images into the tfrecords, so you will need to specify the location of the blocked images.

In order to evaluation the GradCam/Saliency maps with the pointing game, we additionally save binary person masks. Please see the code [here](im2txt/save_coco_person_segmentations.py).

## Training Models
Training scripts are provided [here](im2txt/train_scripts/).

## Generating GradCam and Saliency maps

### GradCam maps
Example commands for using ground-truth captions:
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_gradcam_test_gt/"
CHECKPOINT_PATH="./model/baseline_ft/train/model.ckpt-1500000"
MODEL_NAME="baseline_ft"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_gradcam_with_gt.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --model_name=${MODEL_NAME} --img_path=${IMG_PATH} --save_path=${SAVE_PATH}
```

To compute GradCam maps using the predicted captions, please, run:
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_gradcam_test_pred/"
CHECKPOINT_PATH="./model/baseline_ft/train/model.ckpt-1500000"
JSON_PATH="../final_captions_eccv2018/baseline_ft.json"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_gradcam.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --json_path=${JSON_PATH} --img_path=${IMG_PATH} --save_path=${SAVE_PATH}
```

Note that CHECKPOINT_PATH, MODEL_NAME and JSON_PATH are model-specific, IMG_PATH is a file with a list of image IDs.

We thank @PAIR-code for providing the GradCam [implementation](https://github.com/PAIR-code/saliency), which we include under "./im2test/gradcam".

### Saliency maps
Example commands to generate Saliency maps using ground-truth captions:
```
VOCAB_FILE="./data/word_counts.txt"
SAVE_PATH="./results_saliency_test_gt/"
CHECKPOINT_PATH="./model/baseline_ft/train/model.ckpt-1500000"
MODEL_NAME="baseline_ft"
IMG_PATH="./data/balanced_split/test_woman.txt"
python im2txt/run_inference_with_saliency_with_gt.py   --checkpoint_path=${CHECKPOINT_PATH}   --vocab_file=${VOCAB_FILE} --model_name=${MODEL_NAME} --img_path=${IMG_PATH} --save_path=${SAVE_PATH} --mask_size=32
```

## Running Analysis on Generated COCO Captions
Any result from the paper can be recreated with [this](data_analysis/eccv_2018_results.py) script. You can generate all of the numbers from the tables and figures by running:
```
python data_analysis/eccv_2018_results.py --experiments all
```

## TODO
Anja
- [+] code to run GradCam/Saliency
- [ ] print results when `table_3_main` of the eccv results [script](im2txt/data_analysis/eccv_2018_results.py)
- [+] include the balanced_split
- [ ] make coco location an argument

Lisa
- [ ] add training scripts for balanced and upweight baselines to training scripts folder
- [ ] Code to create blocked images

Other
- [ ] print results for `table_2_supp` of the eccv results [script](im2txt/data_analysis/eccv_2018_results.py)
- [ ] link to requirements.txt is broken
- [ ] specify where pretrained models and captions should be extracted
- [ ] Tab :2 does not have the Outcome Divergence
- [ ] "table_1_supp" does not work, as data/object_lists files are missing
- [ ] minor: fix the order of models
- [ ] add ./data/captions_only_valtrain2014.json or the code to get it

- [ ] 
