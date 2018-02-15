import tensorflow as tf
from im2txt import configuration
import pdb
from im2txt.ops import inputs as input_ops
from im2txt.ops import image_processing
from im2txt.inference_utils import vocabulary

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
assert FLAGS.input_file_pattern, "--input_file_pattern is required"

vocab_file = 'im2txt/data/word_counts.txt'
vocab_file = 'im2txt/data/word_counts_fine_tune_2.txt'
vocab = vocabulary.Vocabulary(vocab_file)

sess = tf.InteractiveSession()

reader = tf.TFRecordReader()
config = configuration.ModelConfig()
config.input_file_pattern = FLAGS.input_file_pattern
config.batch_size = 1

input_queue = input_ops.prefetch_input_data(
                        reader,
                        config.input_file_pattern,
                        is_training = True,
                        batch_size=config.batch_size,
                        values_per_shard=config.values_per_input_shard,
                        input_queue_capacity_factor=config.input_queue_capacity_factor,
                        num_reader_threads=config.num_input_reader_threads)
                        
assert config.num_preprocess_threads % 2 == 0
images_and_captions = []
for thread_id in range(config.num_preprocess_threads):
  serialized_sequence_example = input_queue.dequeue()
  encoded_image, caption = input_ops.parse_sequence_example(
      serialized_sequence_example,
      image_feature=config.image_feature_name,
      caption_feature=config.caption_feature_name)
  image = image_processing.process_image(encoded_image, is_training=True, height=config.image_height, width=config.image_width, thread_id=thread_id, image_format=config.image_format) 
  images_and_captions.append([image, caption])

# Batch inputs.
queue_capacity = (2 * config.num_preprocess_threads *
                  config.batch_size)
pdb.set_trace()
tf.train.start_queue_runners()
images, input_seqs, target_seqs, input_mask = (
    input_ops.batch_with_dynamic_pad(images_and_captions,
                                     batch_size=config.batch_size,
                                     queue_capacity=queue_capacity,
                                     loss_weight_value=10))
pdb.set_trace()
