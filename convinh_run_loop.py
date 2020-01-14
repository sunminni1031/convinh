# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf

from official.convinh import convinh_model
from official.utils.flags import core as flags_core
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
# pylint: enable=g-bad-import-order


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    examples_per_epoch: The number of examples in an epoch.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  if is_training and num_gpus and examples_per_epoch:
    total_examples = num_epochs * examples_per_epoch
    # Force the number of batches to be divisible by the number of devices.
    # This prevents some devices from receiving batches while others do not,
    # which can lead to a lockup. This case will soon be handled directly by
    # distribution strategies, at which point this .take() operation will no
    # longer be needed.
    total_batches = total_examples // batch_size // num_gpus * num_gpus
    dataset.take(total_batches * batch_size)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):  # pylint: disable=unused-argument
    return model_helpers.generate_synthetic_data(
        input_shape=tf.TensorShape([batch_size, height, width, num_channels]),
        input_dtype=tf.float32,
        label_shape=tf.TensorShape([batch_size]),
        label_dtype=tf.int32)

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = 0.1 * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Reduce the learning rate at certain epochs.
  # CIFAR-10: divide by 10 at epoch 100, 150, and 200
  # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn


def convinh_model_fn(features, labels, mode, model_class, model_params, 
                     weight_decay, learning_rate_fn, momentum, loss_scale,
                     loss_filter_fn=None, dtype=convinh_model.DEFAULT_DTYPE):
  """Shared functionality for different convinh model_fns.

  Initializes the convinhModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of convinhModel.
    model_params: dictionary. hyperparameters for the model
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  features = tf.cast(features, dtype)

  model = model_class(model_params, dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  
  loss = cross_entropy + l2_loss
  
  print("weight_decay={}".format(weight_decay))

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)
    
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum)

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      minimize_op = optimizer.minimize(loss, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  if not tf.contrib.distribute.has_distribution_strategy():
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  else:
    # Metrics are currently not compatible with distribution strategies during
    # training. This does not affect the overall performance of the model.
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    #accuracy = (tf.no_op(), tf.constant(0))

  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def convinh_main(
    flags_obj, model_function, input_function, dataset_name, shape=None):
  """Shared main loop for convinh Models.

  Args:
    flags_obj: An object containing parsed flags. See define_convinh_flags()
      for details.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    dataset_name: the name of the dataset for training and evaluation. This is
      used for logging purpose.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags_obj.
      _dir is passed.
  """

  model_helpers.apply_clean(flags.FLAGS)

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
      allow_soft_placement=True)

  distribution_strategy = distribution_utils.get_distribution_strategy(
      flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)
  
  run_config = tf.estimator.RunConfig(
      tf_random_seed=flags_obj.seed,
      train_distribute=distribution_strategy, 
      session_config=session_config,
      keep_checkpoint_max = flags_obj.num_ckpt
      )

  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
      params={
          'model_params':{
                'data_format':flags_obj.data_format, 
                'filters':list(map(int,flags_obj.filters)),
                'ratio_PV': flags_obj.ratio_PV,
                'ratio_SST': flags_obj.ratio_SST,
                'conv_kernel_size':list(map(int,flags_obj.conv_kernel_size)),
                'conv_kernel_size_inh':list(map(int,flags_obj.conv_kernel_size_inh)),
                'conv_strides':list(map(int,flags_obj.conv_strides)),
                'pool_size':list(map(int,flags_obj.pool_size)),
                'pool_strides':list(map(int,flags_obj.pool_strides)),
                'num_ff_layers':flags_obj.num_ff_layers,
                'num_rnn_layers':flags_obj.num_rnn_layers,
                'connection':flags_obj.connection,
                'n_time':flags_obj.n_time,
                'cell_fn':flags_obj.cell_fn,
                'act_fn':flags_obj.act_fn, 
                'pvsst_circuit':flags_obj.pvsst_circuit,
                'gating':flags_obj.gating,
                'normalize':flags_obj.normalize,
                'num_classes':flags_obj.num_classes
              },
          'batch_size' : flags_obj.batch_size,
          'weight_decay': flags_obj.weight_decay,
          'loss_scale': flags_core.get_loss_scale(flags_obj),
          'dtype': flags_core.get_tf_dtype(flags_obj)
      })

  run_params = {
      'batch_size': flags_obj.batch_size,
      'dtype': flags_core.get_tf_dtype(flags_obj),
      'convinh_size': flags_obj.convinh_size, # deprecated
      'convinh_version': flags_obj.convinh_version, # deprecated
      'synthetic_data': flags_obj.use_synthetic_data, # deprecated
      'train_epochs': flags_obj.train_epochs,
  }
  if flags_obj.use_synthetic_data:
    dataset_name = dataset_name + '-synthetic'

  benchmark_logger = logger.get_benchmark_logger()
  benchmark_logger.log_run_info('convinh', dataset_name, run_params,
                                test_id=flags_obj.benchmark_test_id)

  train_hooks = hooks_helper.get_train_hooks(
      flags_obj.hooks,
      model_dir=flags_obj.model_dir,
      batch_size=flags_obj.batch_size)
  
  class input_fn_train(object):
    def __init__(self,num_epochs):
      self._num_epochs = num_epochs
    def __call__(self):
      return input_function(
          is_training=True, data_dir=flags_obj.data_dir,
          batch_size=distribution_utils.per_device_batch_size(
              flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
          num_epochs=self._num_epochs,
          num_gpus=flags_core.get_num_gpus(flags_obj))

  def input_fn_eval():
    return input_function(
        is_training=False, data_dir=flags_obj.data_dir,
        batch_size=distribution_utils.per_device_batch_size(
            flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
        num_epochs=1)
 
  tf.logging.info('Evaluate the intial model.')                          
  eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=flags_obj.max_train_steps)

  benchmark_logger.log_evaluation_result(eval_results)
  
  # training
  total_training_cycle = (flags_obj.train_epochs //
                          flags_obj.epochs_between_evals) + 1
                          
  for cycle_index in range(total_training_cycle):
    
    cur_train_epochs = flags_obj.epochs_between_evals if cycle_index else 1
    
    tf.logging.info('Starting a training cycle: %d/%d, with %d epochs',
                    cycle_index, total_training_cycle, cur_train_epochs)
    
    classifier.train(input_fn=input_fn_train(cur_train_epochs), 
                     hooks=train_hooks, max_steps=flags_obj.max_train_steps)

    tf.logging.info('Starting to evaluate.')

    # flags_obj.max_train_steps is generally associated with testing and
    # profiling. As a result it is frequently called with synthetic data, which
    # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
    # eval (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=flags_obj.max_train_steps)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags_obj.stop_threshold, eval_results['accuracy']):
      break

    if flags_obj.export_dir is not None:
      # Exports a saved model for the given classifier.
      input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
          shape, batch_size=1)
      if cycle_index==0:
        classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn,
                checkpoint_path='{}/model.ckpt-0'.format(flags_obj.model_dir))
      classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)   
      

def define_convinh_flags(convinh_size_choices=None):
  """Add flags and validators for convinh."""
  flags_core.define_base()
  flags_core.define_performance(num_parallel_calls=False)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_enum(
      name='convinh_version', short_name='rv', default='2',
      enum_values=['1', '2'],
      help=flags_core.help_wrap(
          'Version of convinh. (1 or 2) See README.md for details.'))

  choice_kwargs = dict(
      name='convinh_size', short_name='rs', default='34',
      help=flags_core.help_wrap('The size of the convinh model to use.'))

  if convinh_size_choices is None:
    flags.DEFINE_string(**choice_kwargs)
  else:
    flags.DEFINE_enum(enum_values=convinh_size_choices, **choice_kwargs)
 
  # data_format/batch_size/ defined
  # data_dir/model_dir/export_dir defined
  
  flags.DEFINE_list(name="filters", default=[16, 32, 64, 128],
                    help=flags_core.help_wrap('number of channels for each area'))
  
  flags.DEFINE_float(name="ratio_PV", default=0.25,
                     help=flags_core.help_wrap('ratio PV:EX'))
  
  flags.DEFINE_float(name="ratio_SST", default=0.25,
                     help=flags_core.help_wrap('ratio SST:EX'))
  
  flags.DEFINE_list(name="conv_kernel_size", default=[3, 3, 3, 3],
                    help=flags_core.help_wrap('conv kernel size for each area'))
  
  flags.DEFINE_list(name="conv_kernel_size_inh", default=[3,3,3,3,3,3],
                    help=flags_core.help_wrap('pv, sst, fb'))
  
  flags.DEFINE_list(name="conv_strides", default=[1, 1, 1, 1],
                    help=flags_core.help_wrap('conv strides for each area'))
  
  flags.DEFINE_list(name="pool_size", default=[3, 3, 3, 3],
                    help=flags_core.help_wrap('pooling size for each area'))
  
  flags.DEFINE_list(name="pool_strides", default=[1, 2, 2, 2],
                    help=flags_core.help_wrap('pooling strides for each area'))
  
  flags.DEFINE_integer(name="num_ff_layers", default=2,
                       help=flags_core.help_wrap('number of feedforward areas'))
  
  flags.DEFINE_integer(name="num_rnn_layers", default=2,
                       help=flags_core.help_wrap('number of recurrent areas'))
  
  flags.DEFINE_string(name="connection", 
                      default="normal_ff_without_fb",
                      help=flags_core.help_wrap('connection of areas in time steps'))
  
  flags.DEFINE_integer(name="n_time", default=4,
                       help=flags_core.help_wrap('number of time steps to unroll'))
  
  flags.DEFINE_string(name="cell_fn", default="pvsst",
                      help=flags_core.help_wrap('cell function: pvsst or EI'))
  
  flags.DEFINE_string(name="act_fn", default="gate_relu_cell_relu_kernel_abs",
                      help=flags_core.help_wrap('activation function in cell'))

  flags.DEFINE_string(name="pvsst_circuit", default="",
                      help=flags_core.help_wrap('circuit wiring of cell'))  

  flags.DEFINE_string(name="gating", default="in*_out-",
                      help=flags_core.help_wrap('gating mechanism of cell'))
    
  flags.DEFINE_string(name="normalize", default="inside_batch",
                      help=flags_core.help_wrap('normalization of cell'))
  
  flags.DEFINE_integer(name="num_classes", default=10,
                       help=flags_core.help_wrap('number of classes'))
  
  flags.DEFINE_integer(name="seed", default=None,
                      help=flags_core.help_wrap('random seed'))
  
  flags.DEFINE_integer(name="num_ckpt", default=5,
                      help=flags_core.help_wrap('number of checkpoints to save'))
  
  flags.DEFINE_float(name="weight_decay", default=0.0002,
                      help=flags_core.help_wrap('weight decay'))
  
  # The current implementation of convinh v1 is numerically unstable when run
  # with fp16 and will produce NaN errors soon after training begins.
  msg = ('convinh version 1 is not currently supported with fp16. '
         'Please use version 2 instead.')
  @flags.multi_flags_validator(['dtype', 'convinh_version'], message=msg)
  def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
    return (flags_core.DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
            flag_values['convinh_version'] != '1')

