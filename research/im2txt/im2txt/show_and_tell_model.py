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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

try:
    from im2txt.ops import image_embedding
    from im2txt.ops import image_processing
    from im2txt.ops import inputs as input_ops
except:
    sys.path.append('im2txt/ops/')
    sys.path.append('inference_utils/')
    import image_embedding
    import image_processing
    import inputs as input_ops

from inference_utils import vocabulary

vocab_file = 'im2txt/data/word_counts.txt'
try:
    vocab = vocabulary.Vocabulary(vocab_file)
except:
    vocab_file = 'im2txt/data/word_counts.txt'
    vocab = vocabulary.Vocabulary(vocab_file) 

#sanity check on vocabulary words
assert vocab.word_to_id('man') == 11
assert vocab.word_to_id('woman') == 23
assert vocab.word_to_id('brother') == 6056 
assert vocab.word_to_id('wife') == 4691

#synonym list used for ECCV 2018 paper
man_word_list_synonyms = ['man']
woman_word_list_synonyms = ['woman']
#Uncomment below if you would like to use more synonyms (see supplemental for more details)
#man_word_list_synonyms = ['boy', 'brother', 'dad', 'husband', 'man', 'groom', 'male', 'guy', 'men']
#woman_word_list_synonyms = ['girl', 'sister', 'mom', 'wife', 'woman', 'bride', 'female', 'lady', 'women']

confusion_words = [man_word_list_synonyms, woman_word_list_synonyms] #for rebuttal experiment
confusion_word_idx = [[vocab.word_to_id(word) for word in confusion_word_set] for confusion_word_set in confusion_words]
all_confusion_idx = confusion_word_idx[0] + confusion_word_idx[1] #useful for blocking
assert len(confusion_word_idx) == 2  

from PIL import Image
import numpy as np

class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False, flags={}):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    print("FLAGS")
    print (flags)
    assert mode in ["train", "eval", "saliency", "inference", "gradcam"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception
    self.flags = flags
 
    #set up default flags
    if 'loss_weight_value' not in self.flags.keys(): self.flags['loss_weight_value'] = None
    if 'blocked_image' not in self.flags.keys(): self.flags['blocked_image'] = False 
 
    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None
    
    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    # Number of patches generated in saliency model.
    self.num_patches = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "gradcam":
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      images = self.process_image(image_feed)
      input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
      # image is a Tensor of shape [height, width, channels] 
      # caption is a 1-D Tensor of any length
      self.config.batch_size = 1 
      queue_capacity = (2 * self.config.num_preprocess_threads * self.config.batch_size)

      num_queues = 1
      all_images = []
      all_input_seqs = []
      all_target_seqs = []
      all_input_masks = []
      enqueue_list = input_ops.batch_with_dynamic_pad(
                                               [[images, input_feed]],
                                               batch_size=self.config.batch_size,
                                               queue_capacity=queue_capacity,
                                               return_enqueue_list = True)
      all_images.append(tf.expand_dims(enqueue_list[0][0],0))
      all_input_seqs.append(tf.expand_dims(enqueue_list[0][1],0))
      all_target_seqs.append(tf.expand_dims(enqueue_list[0][2],0))
      all_input_masks.append(tf.expand_dims(enqueue_list[0][3],0))

      self.target_seqs = all_target_seqs 
      self.input_mask = all_input_masks 
      self.num_parallel_batches = 1    
    elif self.mode == "saliency":
#      import pdb; pdb.set_trace()
      image_feed = tf.placeholder(dtype=tf.string, shape=[None], name="image_feed")

      images = []
      for i in range(self.config.batch_size):
        images.append(self.process_image(image_feed[i]))
      input_feed = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
      # image is a Tensor of shape [height, width, channels] 
      # caption is a 1-D Tensor of any length
      queue_capacity = (2 * self.config.num_preprocess_threads * self.config.batch_size)

      images_and_captions = []
      for i in range(self.config.batch_size):
        images_and_captions.append([images[i], input_feed])

      num_queues = 1
      all_images = []
      all_input_seqs = []
      all_target_seqs = []
      all_input_masks = []
      enqueue_list = input_ops.batch_with_dynamic_pad(
                                               images_and_captions,
                                               batch_size=self.config.batch_size,
                                               queue_capacity=queue_capacity,
                                               return_enqueue_list = True)
      for i in range(self.config.batch_size):
        all_images.append(tf.expand_dims(enqueue_list[i][0],0))
        all_input_seqs.append(tf.expand_dims(enqueue_list[i][1],0))
        all_target_seqs.append(tf.expand_dims(enqueue_list[i][2],0))
        all_input_masks.append(tf.expand_dims(enqueue_list[i][3],0))
#        all_images.append(enqueue_list[0])
#        all_input_seqs.append(enqueue_list[1])
#        all_target_seqs.append(enqueue_list[2])
#        all_input_masks.append(enqueue_list[3])

      self.target_seqs = [tf.concat(all_target_seqs, 0)]
      self.input_mask = [tf.concat(all_input_masks, 0) ]
      self.num_parallel_batches = 1    
      all_input_seqs = [tf.concat(all_input_seqs, 0)]
    elif self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      all_images = [tf.expand_dims(self.process_image(image_feed), 0)]
      all_input_seqs = [tf.expand_dims(input_feed, 1)]

      # No target sequences or input mask in inference mode.
      # No input mask in saliency mode. Single sentence not padded.
      input_mask = None
      self.num_parallel_batches = 1 
    else:
      # Prefetch serialized SequenceExample protos.
      input_queues = []  #input queues is a list so we can easily handle data from other tfrecord files
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)
      input_queues.append(input_queue)

      if self.flags['blocked_image'] or self.flags['two_input_queues']:
          #start a new input queue for the blocked images
          input_queue2 = input_ops.prefetch_input_data(
              self.reader,
              self.config.blocked_input_file_pattern,
              is_training=self.is_training(),
              batch_size=self.config.batch_size,
              values_per_shard=self.config.values_per_input_shard,
              input_queue_capacity_factor=self.config.input_queue_capacity_factor,
              num_reader_threads=self.config.num_input_reader_threads)
          input_queues.append(input_queue2)

      self.num_parallel_batches = len(input_queues)

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()

      images_and_captions_list = [[] for _ in range(len(input_queues))]
      for thread_id in range(self.config.num_preprocess_threads):

        for i, input_queue in enumerate(input_queues): 
            serialized_sequence_example = input_queue.dequeue()
            encoded_image, caption = input_ops.parse_sequence_example(
                serialized_sequence_example,
                image_feature=self.config.image_keys[i], #TODO change this!
                caption_feature=self.config.caption_feature_name)
            image = self.process_image(encoded_image, thread_id=thread_id)
            images_and_captions_list[i].append([image, caption])
             
      # Batch inputs.

      queue_capacity = (2 * self.config.num_preprocess_threads *
                        self.config.batch_size)

      num_queues = len(images_and_captions_list)
      all_images = []
      all_input_seqs = []
      all_target_seqs = []
      all_input_masks = []
      for i in range(len(input_queues)): 
          outputs = input_ops.batch_with_dynamic_pad(
                                               images_and_captions_list[i],
                                               batch_size=self.config.batch_size,
                                               num_queues=num_queues,
                                               queue_capacity=queue_capacity,
                                               loss_weight_value=self.flags['loss_weight_value'])
          all_images.append(outputs[0])
          all_input_seqs.append(outputs[1])
          all_target_seqs.append(outputs[2])
          all_input_masks.append(outputs[3]) 

      self.target_seqs = all_target_seqs 
      self.input_mask = all_input_masks 
    self.images = tf.concat(all_images, 0)
    self.input_seqs = all_input_seqs 

  def build_image_embeddings(self):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      self.images

    Outputs:
      self.image_embeddings
    """
    inception_output = image_embedding.inception_v3(
        self.images,
        trainable=self.train_inception,
        is_training=self.is_training())
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.
    # An int32 Tensor with shape [batch_size, padded_length].

    Inputs:
      self.input_seqs

    Outputs:
      self.seq_embeddings
    """
    seq_embeddings = []
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      tf.get_variable_scope().reuse_variables()  #needed to share paramters
      for input_seq in self.input_seqs:
          seq_embeddings.append(tf.nn.embedding_lookup(embedding_map, input_seq))

    print(seq_embeddings[0])
    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training, eval, and saliency only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":  #this add dropout (and thus stochasticity to lstm cell)
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      self.image_embeddings = tf.split(self.image_embeddings, 
                                       self.num_parallel_batches, 
                                       axis=0)
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings[0].get_shape()[0], dtype=tf.float32)
      initial_states = []
      for image_embedding in self.image_embeddings:
          _, initial_state = lstm_cell(image_embedding, zero_state)
          initial_states.append(initial_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_states[0], name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings[0], axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
        
        lstm_outputs = [lstm_outputs]

      else: # including gradcam
        # Run the batch of sequence embeddings through the LSTM.
        lstm_outputs = []
        for i in range(self.num_parallel_batches):  #looping over input queues
            sequence_length = tf.reduce_sum(self.input_mask[i], 1)
            lstm_output, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=self.seq_embeddings[i],
                                                sequence_length=sequence_length,
                                                initial_state=initial_states[i],
                                                dtype=tf.float32,
                                                scope=lstm_scope)
            lstm_outputs.append(lstm_output)
    with tf.variable_scope("logits", reuse=tf.AUTO_REUSE) as logits_scope:
      logits = []
      tf.get_variable_scope().reuse_variables()
      for lstm_output in lstm_outputs:  #loop over lstm outputs; remember, there can be multiple input queues which means multiple lstm outputs!
          logit = tf.contrib.layers.fully_connected(
              inputs=lstm_output,
              num_outputs=self.config.vocab_size,
              activation_fn=None,
              reuse=tf.AUTO_REUSE,
              weights_initializer=self.initializer,
              scope=logits_scope)
          logits.append(logit)
      print("logits: ", logits[0])
      
    if self.mode == "inference":
      tf.nn.softmax(logits[0], name="softmax")
    elif self.mode == "saliency":
      tf.nn.softmax(logits[0], name="softmax")
      targets_reshape = [tf.reshape(target, [-1]) for target in self.target_seqs]      
      logits_reshape = [tf.reshape(logit, [-1, 12000]) for logit in logits]
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_reshape, logits=logits_reshape)
      print("loss :", loss)
      self.target_cross_entropy_losses = loss
    elif self.mode == "gradcam":
      tf.nn.softmax(logits[0], name="softmax")
      targets_reshape = [tf.reshape(target, [-1]) for target in self.target_seqs]      
      logits_reshape = [tf.reshape(logit, [-1, 12000]) for logit in logits]
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_reshape, logits=logits_reshape)
      print("loss :", loss)
      self.target_cross_entropy_losses = loss
    else:
      targets_reshape = [tf.reshape(target, [-1]) for target in self.target_seqs]
      weights_reshape = [tf.to_float(tf.reshape(weight, [-1])) for weight in self.input_mask]
      logits_reshape = [tf.reshape(logit, [-1, 12000]) for logit in logits]

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_reshape[0],
                                                              logits=logits_reshape[0])
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights_reshape[0])),
                          tf.reduce_sum(weights_reshape[0]),
                          name="batch_loss")
      tf.losses.add_loss(batch_loss)


      #need to compute cross entropy loss for both input queues
      if self.flags['two_input_queues']:
          losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_reshape[1],
                                                                  logits=logits_reshape[1])
          batch_loss2 = tf.div(tf.reduce_sum(tf.multiply(losses, weights_reshape[1])),
                              tf.reduce_sum(weights_reshape[1]),
                              name="batch_loss2")
          tf.losses.add_loss(batch_loss2)

      #code for blocked image loss
      if self.flags['blocked_image']:
          blocked_loss_weight = tf.to_float(tf.constant(self.flags['blocked_loss_weight']))
          #write blocked weight loss
          softmaxes = tf.nn.softmax(logits[1], 2)

          #CHANGE FOR REBUTTAL
          #c0 = tf.gather(softmaxes, confusion_word_idx[0], axis=2)         
          #c1 = tf.gather(softmaxes, confusion_word_idx[1], axis=2)        

          confusion_word_set = confusion_word_idx[0]
          c0 = []
          for idx in confusion_word_set:
              c0.append(tf.gather(softmaxes, idx, axis=2))
          c0 = tf.add_n(c0)

          confusion_word_set = confusion_word_idx[1]
          c1 = []
          for idx in confusion_word_set:
              c1.append(tf.gather(softmaxes, idx, axis=2))
          c1 = tf.add_n(c1)

          diff = tf.abs(tf.subtract(c0, c1))
          blocked_weights = self.input_mask[1]

          if self.flags['blocked_weight_selective']: #select only man woman words
              #CHANGED FOR REBUTTAL
              #for word in confusion_word_idx:
              for word in all_confusion_idx:
                  condition = tf.equal(self.target_seqs[1], tf.constant(word, dtype=tf.int64)) # 0 out weights for confusion words
                  blocked_weights = tf.where(condition, blocked_weights, tf.zeros_like(blocked_weights, dtype=tf.int32)) # 0 out weights for confusion words
              #this value is very low; at least at the start.  Will want to consider a lamda value.
          blocked_weights = tf.to_float(blocked_weights)
          blocked_loss = tf.reduce_sum(tf.multiply(tf.multiply(diff, blocked_weights), 
                                       blocked_loss_weight), 
                                  name="blocked_loss")
          
          tf.losses.add_loss(blocked_loss)
      if self.flags['blocked_image_ce']:
          losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_reshape[1],
                                                               logits=logits_reshape[1])
          #CHANGED FOR REBUTTAL
          #for word in confusion_word_idx:
          for word in all_confusion_idx: #don't want loss on any gender words
              condition = tf.equal(targets_reshape[1], tf.constant(word, dtype=tf.int64)) # 0 out weights for confusion words
              weights_reshape[1] = tf.where(condition, tf.zeros_like(weights_reshape[1], dtype=tf.float32), weights_reshape[1]) # 0 out weights for confusion words
          blocked_image_ce = tf.multiply(tf.div(tf.reduce_sum(tf.multiply(losses, weights_reshape[1])),
                             tf.reduce_sum(weights_reshape[1])),
                             tf.to_float(tf.constant(self.flags['blocked_image_ce_weight'])),
                             name="blocked_image_ce")
          tf.losses.add_loss(blocked_image_ce)

      if self.flags['confusion_word_non_blocked']:
          blocked_weights = self.input_mask[0]
          softmaxes = tf.nn.softmax(logits[0], 2)

          #CHANGE FOR REBUTTAL
          #c0 = tf.gather(softmaxes, confusion_word_idx[0], axis=2)         
          #c1 = tf.gather(softmaxes, confusion_word_idx[1], axis=2)        

          confusion_word_set = confusion_word_idx[0]
          c0 = []
          for idx in confusion_word_set:
              c0.append(tf.gather(softmaxes, idx, axis=2))
          c0 = tf.add_n(c0)

          confusion_word_set = confusion_word_idx[1]
          c1 = []
          for idx in confusion_word_set:
              c1.append(tf.gather(softmaxes, idx, axis=2))
          c1 = tf.add_n(c1)

          if self.flags['confusion_word_non_blocked_type'] == 'subtraction': 
              confusion_losses = [tf.subtract(tf.constant(1.), tf.subtract(c0, c1)), 
                      tf.subtract(tf.constant(1.), tf.subtract(c1, c0))]
          if self.flags['confusion_word_non_blocked_type'] == 'hinge':
              zero = tf.constant(0.)
              bias = tf.constant(0.1) 
              confusion_losses = [tf.maximum(zero, tf.add(tf.subtract(c1, c0), bias)), 
                  tf.maximum(zero, tf.add(tf.subtract(c0, c1), bias))]
 
          if self.flags['confusion_word_non_blocked_type'] == 'quotient':
              epsilon = tf.constant(1e-5)
              confusion_losses = [tf.divide(c1, tf.add(c0, epsilon)), #want c0 to be high, c1 to be low
                                  tf.divide(c0, tf.add(c1, epsilon))]         

          losses = []
          count = 0
          #CHANGED FOR REBUTTAL
          for confusion_set, confusion_loss in zip(confusion_word_idx, confusion_losses):
              #for word, confusion_loss in zip(confusion_word_idx, confusion_losses):
              blocked_weights_word = blocked_weights 
              for word_idx in confusion_set:
                  #only want loss where man/woman *is* present in the image
                  condition = tf.equal(self.target_seqs[0], 
                                       tf.constant(word_idx, dtype=tf.int64)) # 0 out weights for confusion words
                  blocked_weights_word = tf.where(condition, 
                                                  blocked_weights_word, 
                                                  tf.zeros_like(blocked_weights, dtype=tf.int32)) # 0 out weights for non-confusion words
                  
        
              blocked_weights_word = tf.to_float(blocked_weights_word)
              blocked_loss = tf.multiply(tf.reduce_sum(tf.multiply(confusion_loss, 
                                           blocked_weights_word)), 
                                           tf.to_float(tf.constant(self.flags['confusion_word_non_blocked_weight'])), 
                                      name="non-blocked-word-%d" %count)
              count += 1
              losses.append(blocked_loss) 
          tf.losses.add_loss(tf.add(losses[0], losses[1]))
 
      total_loss = tf.losses.get_total_loss() #By default this includes regularization

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      if self.flags['blocked_image']:
          tf.summary.scalar("losses/blocked_loss", blocked_loss) 
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights_reshape[0]  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
