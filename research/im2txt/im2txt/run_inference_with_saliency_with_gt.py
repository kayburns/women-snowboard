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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import glob
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('im2txt/')
import json
import os.path as osp
import os

import tensorflow as tf
import PIL.Image
import numpy as np

from im2txt import configuration
from im2txt import saliency_wrapper
from im2txt.inference_utils import vocabulary

from mask_utils import generate_image_masks

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("dump_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("model_name", "", "Model name equivalebt to the JSON prediction file.")
tf.flags.DEFINE_string("img_path", "", "Text file containing image IDs.")
tf.flags.DEFINE_string("save_path", "", "Path to the location where outputs should be saved.")

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  import pdb
  pdb.set_trace()
  with g.as_default():
    model = saliency_wrapper.SaliencyWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  #g.finalize()
  save_path = osp.join(FLAGS.save_path, osp.basename(FLAGS.model_name)+'_gt')
  if FLAGS.save_path != "" and not osp.isdir(save_path):
    os.makedirs(save_path)

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  man_id = vocab.word_to_id('man')
  woman_id = vocab.word_to_id('woman')
  person_id = vocab.word_to_id('person')

  of = open(FLAGS.img_path, 'r')
  image_ids = of.read().split('\n')
  if image_ids[-1] == '':
    image_ids = image_ids[0:-1]

  json_path='/data1/coco/annotations_trainval2014/captions_val2014.json'
  json_data = json.load(open(json_path, 'r'))
  json_dict = {}
  for entry in json_data['annotations']:
    image_id = entry['image_id']
    if str(image_id) not in image_ids: continue
    if image_id not in json_dict:
      caption = entry['caption']
      tokens = caption.split(' ')      
      if '-man' in FLAGS.img_path: look_for = 'man'
      elif '-woman' in FLAGS.img_path: look_for = 'woman'
      else: assert(False)
      if look_for in tokens:
        json_dict[image_id] = entry['caption']
    if len(json_dict) == 500: break

  image_ids = json_dict.keys()

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    global_index = 0
    for i, image_id in enumerate(image_ids):
      image_id = int(image_id)
      sys.stdout.write('\r%d/%d' %(i, len(image_ids)))
      original_filename = '/data1/coco/val2014/COCO_val2014_' + "%012d" % (image_id) +'.jpg'
      # create masks
      generate_image_masks(original_file_name, mask_dir)
      mask_filenames = glob.glob('masks/*')

      # loop over all masked images
      for filename in mask_filenames: 
          with tf.gfile.GFile(filename, "r") as f:
            image = f.read()
          if image_id not in json_dict:        
            assert(False)
          caption = json_dict[image_id]
          print(caption)
          if caption[-1] == '.':
            caption = caption[0:-1]      
          tokens = caption.split(' ')
          tokens.insert(0, '<S>')
          encoded_tokens = [vocab.word_to_id(w) for w in tokens]
          man_ids = [i for i, c in enumerate(encoded_tokens) if c == man_id]
          woman_ids = [i for i, c in enumerate(encoded_tokens) if c == woman_id]
          person_ids = [i for i, c in enumerate(encoded_tokens) if c == person_id]
          if not (man_ids or woman_ids or person_ids):
            assert(False)
          else:
            for wid in man_ids: 
              if FLAGS.save_path != "":
               save_path_pre = save_path + '/' + "%06d" % (global_index) + '_'
              else:
               save_path_pre = ""
              model.process_image(sess, image, encoded_tokens, filename, vocab, word_index=wid-1, word_id=man_id, save_path=save_path_pre)
              global_index += 1
            for wid in woman_ids: 
              if FLAGS.save_path != "":
                save_path_pre = save_path + '/' + "%06d" % (global_index) + '_'
              else:
                save_path_pre = ""
              model.process_image(sess, image, encoded_tokens, filename, vocab, word_index=wid-1, word_id=woman_id, save_path=save_path_pre)
              global_index += 1
    #        for wid in person_ids: 
    #          if FLAGS.save_path != "":
    #            save_path_pre = save_path + '/' + "%06d" % (global_index) + '_'
    #          else:
    #            save_path_pre = ""
              model.process_image(sess, image, encoded_tokens, filename, vocab, word_index=wid-1, word_id=person_id, save_path=save_path_pre)
              global_index += 1
      os.remove(mask_filenames)
      import gc
      gc.collect()

if __name__ == "__main__":
  tf.app.run()
