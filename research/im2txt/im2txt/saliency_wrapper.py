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


import tensorflow as tf
import PIL.Image
from matplotlib import pylab as P
import pickle
import matplotlib
import matplotlib.pyplot as plt
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


  def process_image(self, sess, encoded_image, input_feed, word_index=1):
    graph = tf.get_default_graph()
    softmax = sess.run(fetches=["softmax:0"], feed_dict={"image_feed:0": encoded_image, "input_feed:0": input_feed})
    logits = graph.get_tensor_by_name('softmax:0')
    neuron_selector = tf.placeholder(tf.int32)
    neuron_pred = logits[0,word_index][neuron_selector]        
    pred_max = np.argmax(softmax[0][0][word_index])
    return softmax
