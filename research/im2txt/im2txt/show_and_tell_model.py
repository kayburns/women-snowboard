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


import tensorflow as tf

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops

from inference_utils import vocabulary
vocab_file = 'im2txt/data/word_counts.txt'
vocab = vocabulary.Vocabulary(vocab_file) 
confusion_words = ['man', 'woman']
confusion_word_idx = [vocab.word_to_id(word) for word in confusion_words]
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
    assert mode in ["train", "eval", "saliency", "inference"]
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
    if self.mode == "saliency":
      # In saliency mode, images are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64, shape=[None],
                                  name="input_feed")

      # calculate new dimension with image patches
      images = self.process_image(image_feed)
      width, height = images.get_shape().as_list()[:2]
      patch_dim = 32
      self.num_patches = (width // patch_dim) * (height // patch_dim)

      # construct input and target sequence, duplicate to match num patches
      input_length = tf.shape(input_feed)[0] - 1 # TODO: add start and end characters instead of truncating
      input_seqs = tf.slice(input_feed, [0], [input_length])
      self.target_seqs = tf.slice(input_feed, [1], [input_length])
      input_seqs = tf.tile(tf.expand_dims(input_seqs,0), [self.num_patches, 1])
      self.target_seqs = tf.tile(tf.expand_dims(self.target_seqs, 0), [self.num_patches, 1])
      print("target_seqs: ", self.target_seqs)     

      # Process image and insert batch dimensions.
      images = tf.expand_dims(images, 0)
      ksizes = [1, patch_dim, patch_dim, 1]
      images = tf.extract_image_patches(images, ksizes=ksizes, strides=ksizes,
                                        rates=[1, 1, 1, 1], padding='VALID')
      images = tf.reshape(images,[-1, patch_dim, patch_dim, 3])
      paddings = tf.constant([[0,0],[0,width-patch_dim],[0,height-patch_dim],[0,0]], dtype='int32')
      images = tf.pad(images, paddings)
      print("images: ", images)
      input_mask = None

    elif self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.process_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      # No input mask in saliency mode. Single sentence not padded.
      input_mask = None
        
    else:
      # Prefetch serialized SequenceExample protos.
      input_queues = []
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)
      input_queues.append(input_queue)

      if self.flags['blocked_image']:
          #will have to write some logit to input two input_file_patterns
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
      #self.num_parallel_batches = 2 #for debugging 

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      assert self.config.num_preprocess_threads % 2 == 0
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()

      #Code to read in images: this is where changes for blocked images are done

      images_and_captions_list = [[] for _ in range(len(input_queues))]
      for thread_id in range(self.config.num_preprocess_threads):

        for i, input_queue in enumerate(input_queues): 
            serialized_sequence_example = input_queue.dequeue()
#            encoded_image, caption = input_ops.parse_sequence_example(
#                serialized_sequence_example,
#                image_feature=self.config.image_feature_name, #TODO change this!
#                caption_feature=self.config.caption_feature_name)
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

      #self.target_seqs = [all_target_seqs[0], all_target_seqs[0]] #for debugging
      self.target_seqs = all_target_seqs 
      #self.input_mask = [all_input_masks[0], all_input_masks[0]]
      self.input_mask = all_input_masks 
    #self.images = tf.concat([all_images[0], all_images[0]], 0)
    self.images = tf.concat(all_images, 0)
    #self.input_seqs = [all_input_seqs[0], all_input_seqs[0]] 
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
    if self.mode == "train":  #commented out for debug
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
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")

      elif self.mode == "saliency":
        # Run the batch of sequence embeddings through the LSTM.
        
        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

      else:
        # Run the batch of sequence embeddings through the LSTM.
        lstm_outputs = []
        for i in range(self.num_parallel_batches):
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
      for lstm_output in lstm_outputs:
          logit = tf.contrib.layers.fully_connected(
              inputs=lstm_output,
              num_outputs=self.config.vocab_size,
              activation_fn=None,
              reuse=tf.AUTO_REUSE,
              weights_initializer=self.initializer,
              scope=logits_scope)
          logits.append(logit)
      print("logits: ", logits[0])
#    debug_loss = tf.reduce_sum(tf.subtract(lstm_outputs[0], lstm_outputs[1]), name="debug_loss")
#    tf.losses.add_loss(debug_loss)
      
    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    elif self.mode == "saliency":
      targets = tf.reshape(self.target_seqs, [-1])
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
      print("loss :", loss)
      self.target_cross_entropy_losses = loss
      # self.target_cross_entropy_losses = tf.reshape(loss, [self.num_patches, tf.shape(loss)[0]/self.num_patches]) # Used to generate saliency
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

      if self.flags['blocked_image']:
         blocked_loss_weight = tf.to_float(tf.constant(self.flags['blocked_loss_weight']))
         #write blocked weight loss
         softmaxes = tf.nn.softmax(logits[1], 2)
         c0 = tf.gather(softmaxes, confusion_word_idx[0], axis=2)         
         c1 = tf.gather(softmaxes, confusion_word_idx[1], axis=2)        
         diff = tf.abs(tf.subtract(c0, c1))
         blocked_weights = tf.to_float(self.input_mask[1])
         #this value is very low; at least at the start.  Will want to consider a lamda value.
         blocked_loss = tf.reduce_sum(tf.multiply(tf.multiply(diff, blocked_weights), 
                                      blocked_loss_weight), 
                                 name="blocked_loss")
         
         tf.losses.add_loss(blocked_loss)
 



#      import pdb; pdb.set_trace()
      total_loss = tf.losses.get_total_loss(False)

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
