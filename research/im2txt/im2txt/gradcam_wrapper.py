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

import matplotlib
# Fix to run remotely (with no display)
# matplotlib.use('agg')

import tensorflow as tf
import PIL.Image
from matplotlib import pylab as P
import pickle
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

class GradCamWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with a ShowAndTellModel."""

  def __init__(self):
    super(GradCamWrapper, self).__init__()

  def build_model(self, model_config):
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="gradcam")
    model.build()
    return model


  def process_image(self, sess, encoded_image, input_feed, filename, vocab, word_index=1, word_id=None, save_path=None):
    graph = tf.get_default_graph()
    softmax = sess.run(fetches=["softmax:0"], feed_dict={"image_feed:0": encoded_image, "input_feed:0": input_feed})
    logits = graph.get_tensor_by_name('softmax:0')
    neuron_selector = tf.placeholder(tf.int32)
    neuron_pred = logits[0,word_index][neuron_selector]
        
    pred_max = np.argmax(softmax[0][0][word_index])
    if word_id != None:
      print('%s\tpredicted: %s with prob %f , given: %s with prob %.10f' % (filename, vocab.id_to_word(pred_max), np.max(softmax[0][0][word_index]), vocab.id_to_word(word_id), softmax[0][0][word_index][word_id]))
      pred_max = word_id
    
    from grad_cam import GradCam
    grad_cam = GradCam(graph, sess, neuron_pred, graph.get_tensor_by_name('concat:0'), conv_layer = graph.get_tensor_by_name('InceptionV3/InceptionV3/Mixed_7c/concat:0'))

    input_image = PIL.Image.open(filename)
    input_image = input_image.convert('RGB')
    
    im = np.asarray(input_image)
    im_resized = scipy.misc.imresize(im, (299, 299), interp='bilinear', mode=None)    
    im_resized = im_resized / 127.5 - 1.0

    grad_mask_2d = grad_cam.GetMask(im_resized, feed_dict = {neuron_selector: pred_max, "input_feed:0": input_feed}, should_resize = False, three_dims = False)   
    # if np.min(grad_mask_2d) == np.max(grad_mask_2d): grad_mask_2d[0,0]=1.0000001 # Fix for a bug that happens very rarely
    
    mycmap = transparent_cmap(plt.cm.jet)

    w = im_resized.shape[0]
    h = im_resized.shape[1]
    y, x = np.mgrid[0:h, 0:w]

    grad_mask_2d_norm = grad_mask_2d / np.max(grad_mask_2d)
    grad_mask_2d_upscaled = scipy.misc.imresize(grad_mask_2d_norm, (w, h), interp='bilinear', mode='F')    
    
    percentile = 99
    vmax = np.percentile(grad_mask_2d_upscaled, percentile)
    vmin = np.min(grad_mask_2d_upscaled)
    mask_grayscale_upscaled = np.clip((grad_mask_2d_upscaled - vmin) / (vmax - vmin), 0, 1)

    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    ax.imshow( ((im_resized + 1.0) * 127.5)/255.0)
    cb = ax.contourf(x, y, mask_grayscale_upscaled, 15, cmap=mycmap)
    
    if save_path != None and save_path != '':
      np.save(save_path + osp.basename(filename)[0:-4] + '_' + vocab.id_to_word(pred_max) + '.npy', grad_mask_2d)
      plt.savefig(save_path + osp.basename(filename)[0:-4] + '_' + vocab.id_to_word(pred_max) + '.jpg', bbox_inches='tight')
      plt.close()
    else:
      plt.show()


