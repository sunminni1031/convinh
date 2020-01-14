
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.convinh import custom_cell

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,) # not used in convinh model training
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the convinh model.
################################################################################
def batch_norm(inputs, training, data_format='channels_last'):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)


def shape_after_pooling(input_shape, kernel_size, stride_size, 
                        data_format, padding='SAME'):
  """Calculate the tensor's shape after max_pooling""" 
  kernel_size = tf.Dimension(kernel_size)
  stride_size = tf.Dimension(stride_size)
  spatial_size = input_shape[1]
  if padding == 'VALID':
    spatial_size = spatial_size - kernel_size + 1
  spatial_size = (spatial_size + stride_size - 1) // stride_size
  if data_format=='channels_last':
    shape = tf.TensorShape([spatial_size, spatial_size, input_shape[-1]])
  elif data_format=='channels_first':
    shape = tf.TensorShape([input_shape[0], spatial_size, spatial_size])
  return shape


class Model(object):
  """Base class for building the convinh Model."""

  def __init__(self, params, dtype=DEFAULT_DTYPE):
    """Creates a model for classifying an image.

    Args:
      params: dictionary of hyperparameters, they are defined as flags in 
              convinh_run_loop.py
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.params = params
    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))


  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Not used in convinh model training:
      Creates variables in fp32, then casts to fp16 if necessary.
    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model. Note that it is called 'resnet_model'
      in old version of our convinh models
    """
      
    return tf.variable_scope('convinh_model', 
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    with self._model_variable_scope():
      
      data_format = self.params['data_format']
      """
        Hyperparamters in the format of lists specifies the parameter to use 
        for each layer. 
        Hyperparameter 'connection' is a string, and specifies the overall 
        connection of the model: eg: with/without feedback; direct(normal) 
        feedforward or feedforward delayed for a time step
      """
      filters = self.params['filters'] # a list of filter(channel) numbers
      conv_kernel_size = self.params['conv_kernel_size'] # list of kernel sizes
      conv_strides = self.params['conv_strides'] # list of strides for conv
      pool_size = self.params['pool_size'] # list of pooling kernel sizes
      pool_strides = self.params['pool_strides'] # list of pooling strides 
      num_ff_layers= self.params['num_ff_layers'] # number of feedforward layers
      num_rnn_layers = self.params['num_rnn_layers'] # number of recurrent layers
      connection = self.params['connection'] 
      n_time = self.params['n_time'] # number of unrolling time steps
      """
        The following are hyperparamters for the recurrent cell, specifying 
        which recurrent cell, activation function, circuit connection, ratio of
        different neuron types, kernel size of inhibitory neurons,
        gating mechanism and normalization. More details can be found at 
        custom_cell.py
      """
      cell_fn = self.params['cell_fn'] # string, specifying recurrent cell
      if cell_fn=='pvsst':
        cell_fn = custom_cell.PVSSTCell
      elif cell_fn=='EI':
        cell_fn = custom_cell.EICell
      else:
        raise ValueError("cell fn: {} not implemented".format(cell_fn))
      act_fn = self.params['act_fn']  # string, activation func in the cell
      pvsst_circuit = self.params['pvsst_circuit'] # string, specifying circuit
      ratio_PV = self.params['ratio_PV'] # pv:exc
      ratio_SST = self.params['ratio_SST'] # sst:exc
      conv_kernel_size_inh = self.params['conv_kernel_size_inh']
      gating = self.params['gating'] # string, specifying gating mechanism
      normalize = self.params['normalize'] # string, specifying normalization
      num_classes = self.params['num_classes'] # number of classes  
      
      if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2]) # channels last anyway
      
      inputs = tf.identity(inputs, "inputs")
      print("-------------------------------", inputs.shape)
      for i in range(num_ff_layers):       
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters[i], kernel_size=conv_kernel_size[i],
            strides=conv_strides[i], data_format=data_format)
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        inputs = tf.identity(inputs, 'feedforward_layer_{}'.format(i+1))
        if pool_size[i] and pool_strides[i]:
          inputs = tf.layers.max_pooling2d(
              inputs=inputs, pool_size=pool_size[i],
              strides=pool_strides[i], padding='SAME',
              data_format=data_format)       
        inputs = tf.identity(inputs, 'feedforward_layer_{}_after_pooling'.format(i+1))
        print("-------------------------------", inputs.shape)
        
      rnn_cells = {} # dict of rnn cells, rnn_cells[layer]=cell
      input_shape = inputs.shape[1:]
      """
        a for-loop to specify rnn_cells in each layer
      """
      for i in range(num_rnn_layers): 
        i_abs = i + num_ff_layers
        """
          strides_fb: strides used for feedback connection, used in the function
                      _conv2d_transpose. It is the product of two adjacent
                      pool_strides.                     
        """
        if i_abs < num_ff_layers + num_rnn_layers - 1:
          strides_fb = pool_strides[i_abs]*pool_strides[i_abs+1]
        else:
          strides_fb = 0 
        cell_params={
             'input_shape': input_shape,
             'output_channels':filters[i_abs],
             'N_PV':int(filters[i_abs]*ratio_PV),
             'N_SST':int(filters[i_abs]*ratio_SST),
             'kernel_size':conv_kernel_size[i_abs],
             'kernel_size_inh':conv_kernel_size_inh,
             'strides_fb': strides_fb,
             'strides':conv_strides[i_abs],
             'act_fn':act_fn,
             'normalize':normalize,
             'pvsst_circuit':pvsst_circuit,
             'gating':gating,
             'data_format':data_format,
             'name':'convrnn_cell_{}'.format(i+1)
             }
        print("rnn {}------------N_I: {}".format(i, cell_params['N_PV']))
        rnn_cells[i] = cell_fn(cell_params)
        input_shape = shape_after_pooling(
                        input_shape = rnn_cells[i]._output_size, 
                        kernel_size = pool_size[i_abs], 
                        stride_size = pool_strides[i_abs], 
                        data_format = data_format)  
      """
      unrolling:
        rnn_inputs and rnn_states are inputs and states for rnn_layers,
        both are updated through the iteration of time step;
        rnn_nxt_inputs is used only in 'delay_ff' mode (feedforward delayed for 
        a time step), as a temporal storage for rnn_inputs during iteration.
      """
      rnn_inputs = {} # rnn_inputs[i] is a dictionary of inputs to rnn_i 
      rnn_states = {} # rnn_states[i] is a dictionary of states of rnn_i 
      rnn_nxt_inputs = {}
      batch_size = tf.shape(inputs)[0]
      for i in range(num_rnn_layers):
        rnn_inputs[i] = {'ff':rnn_cells[i].zero_input(batch_size, tf.float32)}
        rnn_states[i] = rnn_cells[i].zero_state(batch_size, tf.float32)
        rnn_nxt_inputs[i] = {}
      rnn_inputs[0]['ff'] = inputs
      rnn_nxt_inputs[0]['ff'] = inputs
      # unrolling
      rnn_outputs = [] # rnn_outputs is a list of outputs of the last rnn layer
      for t in range(n_time):
        for i in range(num_rnn_layers):
          i_abs = i + num_ff_layers
          print("-----------time_step{},rnn_layer{}: in_shape:".format(t,i), 
                rnn_inputs[i]['ff'].shape)
          with tf.variable_scope("rnn_{}".format(i+1), reuse=tf.AUTO_REUSE):
            cur_outputs, cur_states = rnn_cells[i](
                                              inputs=rnn_inputs[i], 
                                              state=rnn_states[i], 
                                              training=training, 
                                              rnn_layer=i+1,
                                              scope='time_step_{}'.format(t+1))
            """
              scope='time_step_{}'.format(t+1) is used for batch norm in rnn_cell
              so tf.identity names ('EX','PV','SST') are not reused through 
              time steps. They are in the variable scope 'rnn_(i+1)',
              'rnn_(i+1)_1',...,'rnn_(i+1)_(t-1)'. Weights in cell are reused.
            """                                  
            cur_outputs = tf.identity(cur_outputs, 'EX') 
            cur_outputs = tf.layers.max_pooling2d(cur_outputs,
                                                     pool_size[i_abs],
                                                     pool_strides[i_abs],
                                                     padding='same',
                                                     data_format=data_format)
            cur_outputs = tf.identity(cur_outputs, 'EX_after_pooling') 
          if "normal_ff" in connection:
            if i < num_rnn_layers-1:
              rnn_inputs[i+1]['ff'] = cur_outputs 
              # used for next layer in current time step
            if "with_fb" in connection:
              if i > 0:
                rnn_inputs[i-1]['fb'] = cur_outputs
                # used for previous layer in next time step
          elif "delay_ff" in connection:
            if i < num_rnn_layers-1:
              rnn_nxt_inputs[i+1]['ff'] = cur_outputs
              # used for next layer in next time step
            if "with_fb" in connection:
              if i > 0:
                rnn_nxt_inputs[i-1]['fb'] = cur_outputs
                # used for previous layer in next time step
          else:
            raise ValueError("ff connection:"
                           "{} not implemented".format(connection))
                
          rnn_states[i] = cur_states
          print("-----------time_step{},rnn_layer{}: out_shape:".format(t,i), 
                cur_outputs.shape)
        if "delay_ff" in connection:
          rnn_inputs = rnn_nxt_inputs
        rnn_outputs.append(cur_outputs)
      if "final_sum" in connection:
        inputs = sum(rnn_outputs)
      else:
        inputs = rnn_outputs[-1]

      print("-------------------------------", inputs.shape)

      axes = [2, 3] if data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')      
      print("-------------------------------", inputs.shape)
      inputs = tf.squeeze(inputs, axes)
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)
      inputs = tf.identity(inputs, 'final_dense')

      return inputs

