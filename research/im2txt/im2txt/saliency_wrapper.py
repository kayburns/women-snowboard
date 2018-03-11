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

"""Model wrapper class for performing inference with a ShowAndTellModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from im2txt import show_and_tell_model
from im2txt.inference_utils import inference_wrapper_base
import numpy as np
import copy

import tensorflow as tf
import PIL.Image
from matplotlib import pylab as P
import pickle
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as mcolors
import os
import os.path as osp
slim=tf.contrib.slim
import scipy
import sys
sys.path.append('gradcam')

def transparent_cmap(cmap, N=255):
  "Copy colormap and set alpha values"

  mycmap = cmap
  mycmap._init()
  mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
  return mycmap

class SaliencyWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with a ShowAndTellModel."""

  def __init__(self, batch_size=64):
    super(SaliencyWrapper, self).__init__()
    self.batch_size = batch_size

  def build_model(self, model_config):
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="saliency")
    model.config.batch_size = self.batch_size
    model.build()
    return model


  def process_image(self, sess, encoded_images, input_feed, filename, mask_filenames, vocab, word_index=1, word_id=None, save_path=None):
    graph = tf.get_default_graph()
    softmaxes = np.zeros((len(encoded_images), len(input_feed)-1, 12000)) #<--BAD PROGRAMMING
    for i in range(0, len(encoded_images), self.batch_size):
      batch_images = encoded_images[i:i+self.batch_size]
      num_images = len(batch_images)
      if num_images < self.batch_size:
         for j in range(self.batch_size-num_images): 
           batch_images.append(encoded_images[-1]) 
      softmax = sess.run(fetches=["softmax:0"], feed_dict={"image_feed:0": encoded_images, "input_feed:0": input_feed})
      #import pdb; pdb.set_trace()
      softmaxes[i:i+num_images, ...] = copy.deepcopy(softmax[0][:num_images,...])

    #START VISUALIZATION
    input_image = PIL.Image.open(filename)
    input_image = input_image.convert('RGB')
    
    im = np.asarray(input_image)
    im_resized = scipy.misc.imresize(im, (299, 299), interp='bilinear', mode=None)    
    im_resized = im_resized / 127.5 - 1.0

    # find gender predicitions per pixel
    vocab_id = word_id
    sentence_id = word_index
    saliency = -np.log(softmaxes[:, sentence_id, vocab_id])
    # normalize to [0,1]
    w = im_resized.shape[0]
    h = im_resized.shape[1]
    y, x = np.mgrid[0:h, 0:w]
    saliency =  (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency)) 
    saliency = saliency.reshape((10, 10))
    saliency_resized = scipy.misc.imresize(saliency, (w, h), interp='bilinear', mode='F')   
    mycmap = transparent_cmap(plt.cm.jet)
    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    ax.imshow( ((im_resized + 1.0) * 127.5)/255.0)
    cb = ax.contourf(x, y, saliency_resized, 15, cmap=mycmap)

    #save saliency map
    pred_max = np.argmax(softmax[0][0][word_index])
    if word_id != None:
      print('predicted: ' + vocab.id_to_word(pred_max) + ', beam search: ' + vocab.id_to_word(word_id))
      pred_max = word_id
    if save_path:
      np.save(save_path + osp.basename(filename)[0:-4] + '_' + vocab.id_to_word(pred_max) + '.npy', saliency)
      plt.savefig(save_path + osp.basename(filename)[0:-4] + '_' + vocab.id_to_word(pred_max) + '.jpg', bbox_inches='tight')
      plt.close()
 
    return saliency
