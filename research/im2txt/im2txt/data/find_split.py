# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading



import nltk.tokenize
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("train_image_dir", "/data1/coco/train2014/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/data1/coco/val2014",
                       "Validation image directory.")

tf.flags.DEFINE_string("train_captions_file", "/data1/caption_bias/models/research/im2txt/im2txt/data/raw-data/annotations/captions_train2014.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_captions_file", "/data1/caption_bias/models/research/im2txt/im2txt/data/raw-data/annotations/captions_val2014.json",
                       "Validation captions JSON file.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/tmp/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

def load_image_dataset(captions_file, image_dir):
  """Loads image metadata from a JSON file and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.

  Returns:
    A list of ImageMetadata.
  """
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  # Extract the filenames.
  id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = {}
  for annotation in caption_data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]
    id_to_captions.setdefault(image_id, [])
    id_to_captions[image_id].append(caption)

  # Process the captions and combine the data into a list of ImageMetadata.
  image_names = []
  num_captions = 0
  for image_id, filename in id_to_filename:
    image_names.append(filename)

  return image_names


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Load image metadata from caption files.
  mscoco_val_dataset = load_image_dataset(FLAGS.val_captions_file,
                                                  FLAGS.val_image_dir)

  # Redistribute the MSCOCO data as follows:
  #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
  #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
  #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
  # train_cutoff = int(0.85 * len(mscoco_val_dataset))
  val_cutoff = int(0.5 * len(mscoco_val_dataset))
  # train_dataset = mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
  val_dataset = mscoco_val_dataset[:val_cutoff]
  test_dataset = mscoco_val_dataset[val_cutoff:]
  [print(x) for x in val_dataset]


if __name__ == "__main__":
  tf.app.run()
