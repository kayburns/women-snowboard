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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python import debug as tf_debug

from im2txt import configuration
from im2txt import show_and_tell_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("blocked_input_file_pattern", "", #new flag
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_boolean("debug", False,
                        "If the model should be run in debug mode.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("init_from", "", "Initialize entire model from parameters.")   

#to control loss function
tf.flags.DEFINE_integer("loss_weight_value", None, "To increase loss weight on man/woman words.")   
tf.flags.DEFINE_boolean("blocked_image", False, "If blocked images should be included")   
tf.flags.DEFINE_integer("blocked_loss_weight", 100, "How much to weight blocked loss.")
tf.flags.DEFINE_boolean("blocked_image_ce", False, "Flag to include cross entropy loss on blocked images")
tf.flags.DEFINE_boolean("blocked_weight_selective", True, "Turn this flag off if you wonly want to look at differences in probabilities across all words")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  model_config.image_keys = [model_config.image_feature_name]

  #set flags if you are training with blocked image
  if FLAGS.blocked_image:
      assert FLAGS.blocked_input_file_pattern, "--blocked_input_file_pattern is required if you would like to train with blocked images"
      model_config.blocked_input_file_pattern = FLAGS.blocked_input_file_pattern
      model_config.image_keys.append(model_config.blocked_image_feature_name)
  model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  #go from flags to dict
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception,
        flags=FLAGS.__flags) #let's just pass in all the flags bc this is going to get annoying
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    if FLAGS.init_from:
      inception_restore = model.init_fn
      def restore_full_model(sess):
        print("restoring full model")
        inception_restore(sess)
        saver.restore(sess, FLAGS.init_from)
      model.init_fn = restore_full_model

  # Run training.
  if FLAGS.debug:
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver,
        session_wrapper=tf_debug.LocalCLIDebugWrapperSession)
  else:
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver)


if __name__ == "__main__":
  tf.app.run()
