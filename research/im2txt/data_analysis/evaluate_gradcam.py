# Anja: TODO

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import sys
import json
import os.path as osp
import scipy
import numpy as np
import argparse

from im2txt import metrics

def prepare_resize_gradcam(grad_mask_2d, w, h):
  grad_mask_2d_norm = grad_mask_2d / np.max(grad_mask_2d)
  grad_mask_2d_upscaled = scipy.misc.imresize(grad_mask_2d_norm, (w, h), interp='bilinear', mode='F')
  # Anja: I know that scipy.misc.imresize is depricated but skimage.transform.resize gives a different result :(  
  percentile = 99
  vmax = np.percentile(grad_mask_2d_upscaled, percentile)
  vmin = np.min(grad_mask_2d_upscaled)
  mask_grayscale_upscaled = np.clip((grad_mask_2d_upscaled - vmin) / (vmax - vmin), 0, 1)
  return mask_grayscale_upscaled

coco_dir = 'im2txt/data/mscoco/'# Anja: TODO fix coco
dataType = 'val2014'
cocoImgDir = '{}/images/{}/'.format(coco_dir, dataType)
coco_masks = '{}/masks/{}/'.format(coco_dir, dataType)

def evaluate(checkpoint_path, vocab_file, json_path, img_path, save_path):
  save_path = osp.join(save_path, osp.basename(json_path)[0:-5])

  json_data = json.load(open(json_path, 'r'))
  json_dict = {}
  for entry in json_data:
    image_id = entry['image_id']
    if image_id not in json_dict:
      caption = entry['caption']
      caption = caption.lower()
      json_dict[image_id] = caption

  of = open(img_path, 'r')
  image_ids = of.read().split('\n')
  if image_ids[-1] == '':
    image_ids = image_ids[0:-1]

  pointing_sum = 0
  global_count = 0

  for i, image_id in enumerate(image_ids):
    image_id = int(image_id)
    #sys.stdout.write('\r%d/%d' %(i, len(image_ids)))
    filename = 'im2txt/data/mscoco/images/val2014/COCO_val2014_' + "%012d" % (image_id) +'.jpg'

    coco_mask_file = '%s/COCO_%s_%012d.npy' %(coco_masks, dataType, image_id)
    coco_mask = np.load(coco_mask_file)
    if np.sum(coco_mask) == 0: 
      # no person, perhaps man/woman was not referring to an actual person
      #import ipdb; ipdb.set_trace()
      continue

    if image_id not in json_dict:
      ipdb.set_trace() # Anja: necessary ???
      continue

    caption = json_dict[image_id]
    #print(caption)
    if caption[-1] == '.':
      caption = caption[0:-1]      
    tokens = caption.split(' ')
    tokens.insert(0, '<S>')
    man_ids = [i for i, c in enumerate(tokens) if c == 'man']
    woman_ids = [i for i, c in enumerate(tokens) if c == 'woman']
    if not (man_ids or woman_ids):
      # nothing to do
      continue
    else:
      files = glob.glob(save_path + "/*COCO_val2014_" + "%012d*.npy" % (image_id))
      for f in files:
        gradcam_file = f        
        gradcam_mask = np.load(gradcam_file)
        mask_grayscale_upscaled = prepare_resize_gradcam(gradcam_mask, coco_mask.shape[0], coco_mask.shape[1])
        mask_grayscale_upscaled = mask_grayscale_upscaled / float(np.sum(mask_grayscale_upscaled))
        met = metrics.heatmap_metrics(coco_mask, mask_grayscale_upscaled, gt_type='human', SIZE=coco_mask.shape)
        pointing_sum += met.pointing()
        global_count += 1
  #print("\ncount: %d instances" % (global_count))
  #print("pointing: %.3f" % float(pointing_sum/global_count))
  return (global_count, float(pointing_sum/global_count))

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
  parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='Model checkpoint file.', default='', type=str)
  parser.add_argument('--vocab_file', dest='vocab_file', help='Text file containing the vocabulary.', default='', type=str)
  parser.add_argument('--json_path', dest='json_path', help='JSON file with model predictions.', default='', type=str)
  parser.add_argument('--img_path', dest='img_path', help='Text file containing image IDs', default='', type=str)
  parser.add_argument('--save_path', dest='save_path', help='Path to the location where outputs are saved.', default='', type=str)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  count, acc = evaluate(args.checkpoint_path, args.vocab_file, args.json_path, args.img_path, args.save_path)
  print("\ncount: %d instances" % (count))
  print("pointing: %.5f" % acc)

