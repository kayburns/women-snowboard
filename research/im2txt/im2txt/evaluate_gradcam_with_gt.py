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
import json
import os.path as osp
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
import PIL.Image
import numpy as np
import glob

#sys.path.append('eval_att')
#from metrics import metrics
import metrics

from im2txt import configuration
from im2txt import gradcam_wrapper
from im2txt.inference_utils import vocabulary

def prepare_resize_gradcam(grad_mask_2d, w, h):
  grad_mask_2d_norm = grad_mask_2d / np.max(grad_mask_2d)
  grad_mask_2d_upscaled = scipy.misc.imresize(grad_mask_2d_norm, (w, h), interp='bilinear', mode='F')    
  percentile = 99
  vmax = np.percentile(grad_mask_2d_upscaled, percentile)
  vmin = np.min(grad_mask_2d_upscaled)
  mask_grayscale_upscaled = np.clip((grad_mask_2d_upscaled - vmin) / (vmax - vmin), 0, 1)
  return mask_grayscale_upscaled

def transparent_cmap(cmap, N=255):
  "Copy colormap and set alpha values"

  mycmap = cmap
  mycmap._init()
  mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
  return mycmap

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("model_name", "", "Model name equivalebt to the JSON prediction file.")
tf.flags.DEFINE_string("img_path", "", "Text file containing image IDs.")
tf.flags.DEFINE_string("save_path", "", "Path to the location where outputs should be saved.")

tf.logging.set_verbosity(tf.logging.INFO)

coco_dir = 'im2txt/data/mscoco/'
dataType = 'val2014'
cocoImgDir = '{}/images/{}/'.format(coco_dir, dataType)
coco_masks = '{}/masks/{}/'.format(coco_dir, dataType)

W = -1 # -1 8 32
H = -1 # -1 8 32

exclude = [] #  'man', 'woman', 'person'
#exclude = ['_person']
#exclude = ['_person', '_man']
#exclude = ['_person', '_woman']
#exclude = ['_man']
#exclude = ['_woman']

def main(_):
  save_path = osp.join(FLAGS.save_path, osp.basename(FLAGS.model_name)+'_gt')

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  man_id = vocab.word_to_id('man')
  woman_id = vocab.word_to_id('woman')
  #person_id = vocab.word_to_id('person')

  of = open(FLAGS.img_path, 'r')
  image_ids = of.read().split('\n')
  if image_ids[-1] == '':
    image_ids = image_ids[0:-1]

  json_path = 'im2txt/data/mscoco/annotations/captions_val2014.json'
  json_data = json.load(open(json_path, 'r'))
  json_dict = {}
  for entry in json_data['annotations']:
    image_id = entry['image_id']
    if str(image_id) not in image_ids: continue
    if image_id not in json_dict:
      caption = entry['caption']
      caption = caption.lower()
      tokens = caption.split(' ')      
      if '_man' in FLAGS.img_path: look_for = 'man' # Anja: expect a certain filename
      elif '_woman' in FLAGS.img_path: look_for = 'woman' # Anja: expect a certain filename
      else: assert(False)
      if look_for in tokens:
        json_dict[image_id] = entry['caption']
    if len(json_dict) == 500: break

  image_ids = json_dict.keys()

  emd_sum = 0
  spear_sum = 0
  rank_sum = 0
  iou_sum = 0
  pointing_sum = 0

  global_count = 0
  for i, image_id in enumerate(image_ids):
    image_id = int(image_id)
    sys.stdout.write('\r%d/%d' %(i, len(image_ids)))
    filename = 'im2txt/data/mscoco/images/val2014/COCO_val2014_' + "%012d" % (image_id) +'.jpg'

    #input_image = PIL.Image.open(filename)
    #input_image = input_image.convert('RGB')    
    #im = np.asarray(input_image)
    #im_resized = scipy.misc.imresize(im, (W, H), interp='bilinear', mode=None)    
    #im_resized = im_resized / 127.5 - 1.0
    #w = im_resized.shape[0]
    #h = im_resized.shape[1]
    #y, x = np.mgrid[0:h, 0:w]
    #mycmap = transparent_cmap(plt.cm.jet)

    coco_mask_file = '%s/COCO_%s_%012d.npy' %(coco_masks, dataType, image_id)
    coco_mask = np.load(coco_mask_file)
    if np.sum(coco_mask) == 0: 
      # no person mask
      #import ipdb; ipdb.set_trace()
      continue
    if W > 0:
      coco_mask_resized = scipy.misc.imresize(coco_mask, (W, H), interp='bilinear', mode=None)
      coco_mask_resized = coco_mask_resized / float(np.sum(coco_mask_resized))

    #fig = plt.figure(frameon=False)
    #plt.imshow(coco_mask_resized)
    #plt.show()
    #plt.close()

    if image_id not in json_dict: # Anja: unnecessary     
      continue
    caption = json_dict[image_id]
    caption = caption.lower()
    print(caption)
    if caption[-1] == '.':
      caption = caption[0:-1]      
    tokens = caption.split(' ')
    tokens.insert(0, '<S>')
    encoded_tokens = [vocab.word_to_id(w) for w in tokens]
    man_ids = [i for i, c in enumerate(encoded_tokens) if c == man_id]
    woman_ids = [i for i, c in enumerate(encoded_tokens) if c == woman_id]
    #person_ids = [i for i, c in enumerate(encoded_tokens) if c == person_id]
    if not (man_ids or woman_ids): #or person_ids):
      assert(False)
    else:
      files = glob.glob(save_path + "/*COCO_val2014_" + "%012d*.npy" % (image_id))
      for f in files:
        gradcam_file = f
        exclude_file = False
        for w in exclude:
          if w in gradcam_file:
            exclude_file = True
            break
        if exclude_file: continue
        gradcam_mask = np.load(gradcam_file)
        if W > 0:
          mask_grayscale_upscaled = prepare_resize_gradcam(gradcam_mask, W, H)
          mask_grayscale_upscaled = mask_grayscale_upscaled / float(np.sum(mask_grayscale_upscaled))
          met = metrics.heatmap_metrics(coco_mask_resized,
                              mask_grayscale_upscaled,
                              gt_type='human', SIZE=(W,H))
        else:
          mask_grayscale_upscaled = prepare_resize_gradcam(gradcam_mask, coco_mask.shape[0], coco_mask.shape[1])
          mask_grayscale_upscaled = mask_grayscale_upscaled / float(np.sum(mask_grayscale_upscaled))
          met = metrics.heatmap_metrics(coco_mask,
                              mask_grayscale_upscaled,
                              gt_type='human', SIZE=coco_mask.shape)

        # Compute EMD
        if W>0 and W<16:
          emd_mean_score, emd_scores = met.earth_mover(distance='euclidean')
          emd_sum += emd_mean_score
        # Compute Spearman
        if W>0:
          spear_mean_score, spear_scores = met.spearman_correlation()
          spear_sum += spear_mean_score
        # Compute Rank correlation
        if W>0:
          rank_mean_score = met.mean_rank_correlation()
          rank_sum += rank_mean_score
        # Compute IOU
        if W>0:
          mean_iou = met_notnormalized.iou()
          iou_sum += mean_iou
        # pointing
        if W<0:
          pointing_sum += met.pointing()#coco_mask.flatten()[np.argmax(mask_grayscale_upscaled.flatten())]
        global_count += 1
  print("\ncount: %d instances" % (global_count))
  #print("EMD: %.3f" % float(emd_sum/global_count))
  #print("SPEAR: %.5f" % float(spear_sum/global_count))
  #print("rank: %.5f" % float(rank_sum/global_count))
  #print("iou: %.8f" % float(iou_sum/global_count))
  print("pointing: %.5f" % float(pointing_sum/global_count))
  #print("%.3f\t%.5f\t%.5f\t%.8f\t%d" % (float(emd_sum/global_count), float(spear_sum/global_count),
  #                                      float(rank_sum/global_count), float(iou_sum/global_count), global_count))

if __name__ == "__main__":
  tf.app.run()

