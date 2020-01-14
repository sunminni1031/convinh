# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# ==============================================================================
"""Runs a convinh model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.convinh import convinh_model
from official.convinh import convinh_run_loop

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training, data_aug):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training, data_aug)

  return image, label


def preprocess_image(image, is_training, data_aug):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training and data_aug:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn_v(is_training, data_dir, batch_size, data_aug, num_epochs, num_gpus):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    data_aug: whether to use data augmentation
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  return convinh_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train'],
      parse_record_fn= (lambda raw_record,is_training: \
                        parse_record(raw_record, is_training, data_aug=data_aug)),
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )


def input_fn_aug(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  return input_fn_v(is_training, data_dir, batch_size, True, num_epochs, num_gpus)

  
def input_fn_noaug(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  return input_fn_v(is_training, data_dir, batch_size, False, num_epochs, num_gpus)


#def get_synth_input_fn():
#  return convinh_run_loop.get_synth_input_fn(
#      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  learning_rate_fn = convinh_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
      decay_rates=[1, 0.1, 0.01, 0.001])

#  weight_decay = 2e-4

  def loss_filter_fn(_):
    return True

  return convinh_run_loop.convinh_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=convinh_model.Model,
      model_params=params['model_params'],
      weight_decay=params['weight_decay'],
      loss_scale=params['loss_scale'],
      dtype=params['dtype'],
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      loss_filter_fn=loss_filter_fn
  )


def define_cifar_flags():
  convinh_run_loop.define_convinh_flags()
  flags.DEFINE_integer(name="data_aug", default=1,
                help=flags_core.help_wrap('whether to use data augmentation'))
  flags.DEFINE_float(name="data_size", default=1.0,
                help=flags_core.help_wrap('size of the dataset relative to default'))
  flags.adopt_module_key_flags(convinh_run_loop)
  flags_core.set_defaults(train_epochs=250,
                          epochs_between_evals=10,
                          batch_size=128)


def run_cifar(flags_obj):
  """Run convinh CIFAR-10 training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
#  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
#                    or input_fn)
  global _NUM_IMAGES
  _NUM_IMAGES = {
    'train': int(50000*flags_obj.data_size),
    'validation': int(10000*flags_obj.data_size),
    }
  
  input_function = input_fn_aug if flags_obj.data_aug else input_fn_noaug
  print_str = "" if flags_obj.data_aug else "not "
  print_str += "using data augmentation with relative data size {}".format(flags_obj.data_size)
  print(print_str)

  convinh_run_loop.convinh_main(
      flags_obj, cifar10_model_fn, input_function, DATASET_NAME,
      shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_cifar(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_cifar_flags()
  absl_app.run(main)
