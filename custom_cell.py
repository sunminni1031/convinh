#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 22:31:00 2018
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.eager import context
import numpy as np
import re
######################################
def convert_data_format(data_format, ndim):
  if data_format == 'channels_last':
    if ndim == 3:
      return 'NWC'
    elif ndim == 4:
      return 'NHWC'
    elif ndim == 5:
      return 'NDHWC'
    else:
      raise ValueError('Input rank not supported:', ndim)
  elif data_format == 'channels_first':
    if ndim == 3:
      return 'NCW'
    elif ndim == 4:
      return 'NCHW'
    elif ndim == 5:
      return 'NCDHW'
    else:
      raise ValueError('Input rank not supported:', ndim)
  else:
    raise ValueError('Invalid data_format:', data_format)


def batch_norm(inputs, training, data_format='channels_last'):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  _BATCH_NORM_DECAY = 0.997
  _BATCH_NORM_EPSILON = 1e-5
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def _conv(args, num_features, kernel_size, strides, 
          use_bias, abs_constraint,
          padding='SAME', separable=True, channel_multiplier=1,  
          kernel_name="kernel", bias_name="biases", 
          initializer=None, bias_start=0.0, data_format="channels_last"):
  """Convolution.

  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    num_features: int, number of features
    kernel_size: int,  kernel height or width.
    strides: int, convolution stride
    use_bias: boolean, whether to add bias in convolution
    abs_constraint: boolean, whether to use absolute value constraint for 
                    convolution weights
    separable: boolean, whether to use depth separable conv
    channel_multiplier: parameter for depth separable conv
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if data_format == "channels_first":
    data_format='NCHW'
    channel_axis=1
  elif data_format == "channels_last":
    data_format='NHWC'
    channel_axis=3
  else:
    raise ValueError("data_format should be one of "
                     "channels_first, channels_last")
  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[channel_axis]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  strides = [strides, strides]
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    default_strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    default_strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    default_strides = shape_length * [1]
  if strides is not None:
      if shape_length != (len(strides) + 2):
          raise ValueError("strides is only valid when using Conv2D")
      else:
          if data_format=='NHWC':
            strides = [1] + strides + [1]
          elif data_format=='NCHW':
            strides = [1, 1] + strides
          else:
            raise ValueError("data_format:{} is not valid".format(data_format))
  else:
      strides = default_strides
  # Now the computation.
  if len(args) == 1:
    inputs = args[0]
  else:
    inputs = array_ops.concat(axis=channel_axis, values=args)

  # depth separable convolution
  if separable: 
      depthwise_filter = vs.get_variable(
              kernel_name+"_depthwise", 
              [kernel_size, kernel_size, total_arg_size_depth, 
                channel_multiplier], dtype=dtype, initializer=initializer)
      if abs_constraint:    
        depthwise_filter = tf.abs(depthwise_filter, 
                                      name=kernel_name
                                      +"_depthwise_abs")
      pointwise_filter = vs.get_variable(
              kernel_name+"_pointwise", [1, 1, channel_multiplier * 
              total_arg_size_depth, num_features], dtype=dtype, 
              initializer=initializer)
      if abs_constraint:
        pointwise_filter = tf.abs(pointwise_filter,
                                      name=kernel_name+
                                      "_pointwise_abs")
      res = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, 
                                   strides, padding=padding, data_format=data_format)
  # normal convolution
  else:     
      kernel = vs.get_variable(
              kernel_name, 
              [kernel_size, kernel_size, total_arg_size_depth, num_features], 
              dtype=dtype, initializer=initializer)
      if abs_constraint:
          kernel = tf.abs(kernel, name=kernel_name+"_abs")
      res = conv_op(inputs, kernel, strides, padding=padding, data_format=data_format)
  # bias for convolution
  if use_bias:
    bias_term = vs.get_variable(
        bias_name, [num_features],
        dtype=dtype,
        initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    res = tf.nn.bias_add(res, bias_term, data_format=data_format)

  return res


def deconv_output_length(input_length, filter_size, padding, stride):
  """Determines output length of a transposed convolution given input length.

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The output length (integer).
  """
  if input_length is None:
    return None
  input_length *= stride
  if padding == 'valid':
    input_length += max(filter_size - stride, 0)
  elif padding == 'full':
    input_length -= (stride + filter_size - 2)
  return input_length


def _conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     strides,
                     use_bias,
                     abs_constraint,
                     padding='same',
                     kernel_name='kernel',
                     bias_name='biases',
                     initializer=None,
                     bias_start=0.0,
                     data_format='channels_last'):
    dtype=inputs.dtype
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    if data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = (kernel_size, kernel_size, filters, input_dim)
    
    kernel = vs.get_variable(
            kernel_name, 
            kernel_shape, 
            dtype=dtype, initializer=initializer)
    if abs_constraint:
        kernel = tf.abs(kernel, name=kernel_name+"_abs")

    if use_bias:
      bias = vs.get_variable(
          bias_name, 
          [filters],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    #tf.layers.conv2d_transpose
    batch_size = tf.shape(inputs)[0]
    if data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = kernel_size, kernel_size
    stride_h, stride_w = strides, strides
    # Infer the dynamic output shape:
    out_height = deconv_output_length(height,
                                     kernel_h,
                                     padding,
                                     stride_h)
    out_width = deconv_output_length(width,
                                    kernel_w,
                                    padding,
                                    stride_w)
    if data_format == 'channels_first':
      output_shape = (batch_size, int(filters), int(out_height), int(out_width))
      strides = (1, 1, stride_h, stride_w)
    else:
      output_shape = (batch_size, int(out_height), int(out_width), int(filters))
      strides = (1, stride_h, stride_w, 1)

#    print("***************", input_shape)
#    print("****************", output_shape)
    outputs = tf.nn.conv2d_transpose(
        inputs,
        kernel,
        output_shape,
        strides,
        padding=padding.upper(),
        data_format=convert_data_format(data_format, ndim=4))

    if use_bias:
      outputs = tf.nn.bias_add(
          outputs,
          bias,
          data_format=convert_data_format(data_format, ndim=4))

    return outputs


def static_rnn_with_training_mode(cell,
               inputs,
               training, 
               initial_state=None,
               dtype=None,
               sequence_length=None,
               scope=None):
  '''
    training mode is included for batch normalization inside the cell
  '''
  outputs = []
  with vs.variable_scope(scope or "rnn"):
    first_input = inputs[0]
    if type(first_input) == list:
      first_input = first_input[0]
    batch_size = tf.shape(first_input)[0]
    state = cell.zero_state(batch_size, tf.float32)
    #####################
    for time, input_ in enumerate(inputs):
      output, state = cell(input_, state, training, 'time_step_{}'.format(time+1)) 
      outputs.append(output)
    return (outputs, state)


######################################
class PVSSTCell(rnn_cell_impl.RNNCell):
  """
  EX: output_channels
  PV: output gate, OG
  SST: input gate, SST
  """

  def __init__(self, params):
    """Construct PVSSTCell.

    Args:
      params: hyperparameters for PVSSTCell

    """
    super(PVSSTCell, self).__init__(params['name'])
    self._input_shape = params['input_shape']
    self._output_channels = params['output_channels']
    self._N_PV = params['N_PV']
    self._N_SST = params['N_SST']
    self._kernel_size = params['kernel_size']
    self._kernel_size_inh = params['kernel_size_inh'] 
    # kernel_size_inh is a list of kernel size for different connections
    # the meaning of each element is defined here:
    self._kernel_size_pv_in = self._kernel_size_inh[0]
    self._kernel_size_sst_in = self._kernel_size_inh[1]
    self._kernel_size_fb = self._kernel_size_inh[2]
    self._kernel_size_hid = self._kernel_size_inh[3]
    # strides_fb is as calculated before in the file convinh_model.py
    self._strides_fb = params['strides_fb']
    self._strides = params['strides']
    # act_fn: string, activation function,eg:'gate_relu_cell_relu_kernel_abs'
    self._act_fn = params['act_fn']
    # normalize: string, specifying batch/layer normalization and its position
    self._normalize = params['normalize']
    # pvsst_circuit: string, eg: '','flip_sign','SstNoFF'
    self._pvsst_circuit = params['pvsst_circuit']
    # gating: string, gating mechanism, eg: 'in_mult_out_subt'
    self._gating = params['gating']
    self._data_format = params['data_format']
    self._skip_connection = False
    self._padding='SAME'
    self._total_output_channels = self._output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]
    if self._skip_connection and (self._strides != 1):
      raise ValueError("stride should be 1 if skip_connection is True")   
    # shape calculation
    kernel_H = tf.Dimension(self._kernel_size)
    strides_H = tf.Dimension(self._strides)
    state_H = self._input_shape[1]
    if self._padding == 'VALID':
      state_H = state_H - kernel_H + 1
    state_H = (state_H + strides_H - 1) // strides_H
    state1_C = self._output_channels
    if ("remove_OG" in self._pvsst_circuit) and ("inside" not in self._normalize):
      state1_C = self._N_SST
    if self._data_format=='channels_last':
      state0_size = tensor_shape.TensorShape(
          [state_H, state_H] + [self._output_channels])
      state1_size = tensor_shape.TensorShape(
          [state_H, state_H] + [state1_C])
      self._state_size = rnn_cell_impl.LSTMStateTuple(state0_size, state1_size)
      self._output_size = tensor_shape.TensorShape(
          [state_H, state_H] + [self._total_output_channels])
    elif self._data_format=='channels_first':
      state0_size = tensor_shape.TensorShape(
          [self._output_channels] + [state_H, state_H])
      state1_size = tensor_shape.TensorShape(
          [state1_C] + [state_H, state_H])
      self._state_size = rnn_cell_impl.LSTMStateTuple(state0_size, state1_size)
      self._output_size = tensor_shape.TensorShape(
          [self._total_output_channels] + [state_H, state_H])
    else:
      raise ValueError("data_format not valid: {}".format(self._data_format)) 
      
  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size
  
  def zero_input(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroInput", values=[batch_size]):
      output = rnn_cell_impl._zero_state_tensors(self._input_shape, batch_size, dtype)
    return output
  
  def __call__(self, inputs, state, training, rnn_layer, scope):
    # see the output of this function:
    # when "remove_OG" and not using "inside" norm
    # cell and hidden are equivalent, and the state is (EX,SST)
    if ("remove_OG" in self._pvsst_circuit) and ("inside" not in self._normalize):
      cell, last_SST = state
      hidden = cell
    else:
      cell, hidden = state
    # specify inputs
    ff_inputs = inputs['ff']
    if 'fb' in inputs:
      fb_inputs = inputs['fb']
    # convolution: depth separable or normal
    if 'norm_conv' in self._pvsst_circuit:
      separable=False
    else:
      separable=True
    # activation function for gate is set to be relu
    if "gate_relu" in self._act_fn:
      gate_act_fn = tf.nn.relu
    elif "gate_sigmoid" in self._act_fn:
      gate_act_fn = tf.sigmoid
    else:
      raise ValueError("gate fn:"
                           "{} not implemented".format(self._act_fn))
    # activation function for cell
    if "cell_relu" in self._act_fn:
      cell_act_fn = tf.nn.relu
    elif "cell_elu" in self._act_fn:
      cell_act_fn = tf.nn.elu
    elif "cell_sigmoid" in self._act_fn:
      cell_act_fn = tf.sigmoid
    elif "cell_tanh" in self._act_fn:
      cell_act_fn = tf.tanh
    elif "cell_softplus" in self._act_fn:
      cell_act_fn = tf.math.softplus
    elif "cell_retanh" in self._act_fn:
      cell_act_fn = tf.nn.relu(tf.tanh)
    else:
      raise ValueError("cell act fn:"
                           "{} not implemented".format(self._act_fn))
    # whether to use absolute constraint
    abs_constraint = True if "kernel_abs" in self._act_fn else False
    # flip_sign: to exchange sign of E and I
    sign_for_hid = -1.0 if 'flip_sign' in self._pvsst_circuit else 1.0
    sign_for_inh = 1.0 if 'flip_sign' in self._pvsst_circuit else -1.0
    # EX as FF connection can be specified in pvsst_circuit with 'FF_XX'
    abs_constraint_ff = abs_constraint
    sign_for_ff = 1.0
    if rnn_layer > 1:
      if 'FF_inh' in self._pvsst_circuit:
        sign_for_ff = -1.0 
      if 'FF_no_abs' in self._pvsst_circuit:
        abs_constraint_ff = False
      elif 'FF_abs' in self._pvsst_circuit:
        abs_constraint_ff = True
    # bias_start specified with "flip_sign_withff"
    bias_start = 0.0
    if (sign_for_hid<0) and (sign_for_ff<0):
      try:
        bias_start = float(re.findall("\d+\.\d+", self._pvsst_circuit)[0])
      except:
        bias_start = 1.0
    # bias_for_IG: the bias of SST->EX conv
    bias_for_IG = True
    if ("simple_subt" in self._pvsst_circuit):
      if "in-" in self._gating:
        bias_for_IG = False

    with tf.variable_scope("_input_gate", reuse=tf.AUTO_REUSE):
      SST = _conv([hidden],
                  self._N_SST, self._kernel_size_sst_in, self._strides,
                  use_bias=True, abs_constraint=abs_constraint,
                  separable=separable, bias_start=bias_start,
                  data_format=self._data_format)*sign_for_hid
      if 'SstNoFF' not in self._pvsst_circuit:
        SST += _conv([ff_inputs],
                  self._N_SST, self._kernel_size_sst_in, self._strides,
                  use_bias=False, abs_constraint=abs_constraint_ff,
                  separable=separable, 
                  kernel_name='kernel_ff', bias_name='biases_ff',
                  data_format=self._data_format)*sign_for_ff
      if 'fb' in inputs:
        SST += _conv2d_transpose(fb_inputs,
                   self._N_SST, self._kernel_size_fb, self._strides_fb,
                   use_bias=False, abs_constraint=abs_constraint,
                   kernel_name='kernel_fb', bias_name='biases_fb',
                   data_format=self._data_format)*sign_for_hid
      if "mem_SST" in self._pvsst_circuit:
      # recurrent SST
        SST += _conv([last_SST],
                  self._N_SST, self._kernel_size_sst_in, self._strides,
                  use_bias=False, abs_constraint=abs_constraint,
                  separable=separable, 
                  kernel_name='kernel_II', bias_name='biases_II',
                  data_format=self._data_format)*sign_for_inh
        # normalization
        if "SST_batch" in self._normalize:
          with tf.variable_scope(scope+"_SST", reuse=False):
            SST = batch_norm(SST, training, data_format=self._data_format)
        elif "SST_layer" in self._normalize:
          SST = tf.contrib.layers.layer_norm(SST)
        # forget gate and cell state
        fg_SST = tf.get_variable("fg_SST", (self._N_SST))
        fg_SST = tf.sigmoid(fg_SST + 1.0)
        SST = fg_SST * last_SST + (1-fg_SST) * cell_act_fn(SST)
        tf.identity(SST, 'SST')
        # gating
        input_gate = _conv([last_SST], 
                           self._output_channels, self._kernel_size_hid, self._strides,
                           use_bias=bias_for_IG, abs_constraint=abs_constraint,
                           separable=separable, 
                           kernel_name='kernel_gate', bias_name='biases_gate',
                           data_format=self._data_format)
      else:
      # default: feedforward SST
        # normalization
        if "SST_batch" in self._normalize:
          with tf.variable_scope(scope+"_SST", reuse=False):
            SST = batch_norm(SST, training, data_format=self._data_format)
        elif "SST_layer" in self._normalize:
          SST = tf.contrib.layers.layer_norm(SST)
        elif "SST_dropout" in self._normalize:
          keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
          SST = tf.nn.dropout(SST, keep_prob=keep_prob)
        # activation
        SST = cell_act_fn(SST)
        if 'drop_SST' in self._pvsst_circuit:
          SST = 0.0*SST
        tf.identity(SST, 'SST')
        # gating
        input_gate = _conv([SST], 
                           self._output_channels, self._kernel_size_hid, self._strides,
                           use_bias=bias_for_IG, abs_constraint=abs_constraint,
                           separable=separable, 
                           kernel_name='kernel_gate', bias_name='biases_gate',
                           data_format=self._data_format)
    if "remove_OG" not in self._pvsst_circuit:
        with tf.variable_scope("_output_gate", reuse=tf.AUTO_REUSE):
            PV = _conv([hidden],
                        self._N_PV, self._kernel_size_pv_in, self._strides,
                        use_bias=True, abs_constraint=abs_constraint,
                        separable=separable, 
                        bias_start=bias_start,
                        data_format=self._data_format)*sign_for_hid
            PV += _conv([ff_inputs],
                        self._N_PV, self._kernel_size_pv_in, self._strides,
                        use_bias=False, abs_constraint=abs_constraint_ff,
                        separable=separable, 
                        kernel_name='kernel_ff', bias_name='biases_ff',
                        data_format=self._data_format)*sign_for_ff
            #normalization
            if "PV_batch" in self._normalize:
              with tf.variable_scope(scope+"_PV", reuse=False):
                PV = batch_norm(PV, training, data_format=self._data_format)
            elif "PV_layer" in self._normalize:
              PV = tf.contrib.layers.layer_norm(PV)
            elif "PV_dropout" in self._normalize:
              keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
              PV = tf.nn.dropout(PV, keep_prob=keep_prob)
            #activation
            PV = cell_act_fn(PV)
            if 'drop_PV' in self._pvsst_circuit:
              PV = 0.0*PV
            tf.identity(PV, 'PV')
            #gating
            output_gate = _conv([PV], 
                               self._output_channels, self._kernel_size_hid, self._strides,
                               use_bias=True, abs_constraint=abs_constraint,
                               separable=separable, 
                               kernel_name='kernel_gate', bias_name='biases_gate',
                               data_format=self._data_format)      
    with tf.variable_scope("_forget_gate", reuse=tf.AUTO_REUSE):
      if "simple_fg" not in self._pvsst_circuit:
        forget_gate = _conv([hidden],
                            self._output_channels, self._kernel_size_hid, self._strides,
                            use_bias=True, abs_constraint=abs_constraint,
                            separable=separable, 
                            bias_start=bias_start,
                            data_format=self._data_format)*sign_for_hid
        forget_gate += _conv([ff_inputs],
                            self._output_channels, self._kernel_size, self._strides,
                            use_bias=False, abs_constraint=abs_constraint_ff,
                            separable=separable, 
                            kernel_name='kernel_ff', bias_name='biases_ff',
                            data_format=self._data_format)*sign_for_ff
      else:
        forget_gate = tf.get_variable("fg_E", (self._output_channels))
      forget_gate = tf.sigmoid(forget_gate + 1.0)
    with tf.variable_scope("_new_input", reuse=tf.AUTO_REUSE):
        new_input = _conv([hidden], 
                          self._output_channels, self._kernel_size_hid, self._strides,
                          use_bias=True, abs_constraint=abs_constraint,
                          separable=separable, 
                          bias_start=bias_start,
                          data_format=self._data_format)*sign_for_hid
        new_input += _conv([ff_inputs], 
                          self._output_channels, self._kernel_size, self._strides,
                          use_bias=False, abs_constraint=abs_constraint_ff,
                          separable=separable, 
                          kernel_name='kernel_ff', bias_name='biases_ff',
                          data_format=self._data_format)*sign_for_ff         

    if "in*" in self._gating:
      # input gate
      if 'flip_sign' in self._pvsst_circuit:
        input_gate = tf.sigmoid(input_gate)
      else:
        input_gate = tf.subtract(tf.constant(1.0), tf.sigmoid(input_gate))
      #input current normalization 
      if "input_batch" in self._normalize:
        with tf.variable_scope(scope, reuse=False):
          new_input = batch_norm(new_input, training, data_format=self._data_format)
      elif "input_layer" in self._normalize:
        new_input = tf.contrib.layers.layer_norm(new_input)
      elif "input_dropout" in self._normalize:
        keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
        new_input = tf.nn.dropout(new_input, keep_prob=keep_prob)
      # cell state
      new_cell = cell * forget_gate + cell_act_fn(new_input) * input_gate
    elif "in-" in self._gating:
      #input current
      if "simple_subt" in self._pvsst_circuit:
        if 'flip_sign' in self._pvsst_circuit:
          new_input = new_input + input_gate
        else:
          new_input = new_input - input_gate
      else:
        if 'flip_sign' in self._pvsst_circuit:
          new_input = new_input + gate_act_fn(input_gate) - 1.0       
        else:
          new_input = new_input - gate_act_fn(input_gate)
      #input current normalization 
      if "input_batch" in self._normalize:
        with tf.variable_scope(scope+"_EX", reuse=False):
          new_input = batch_norm(new_input, training, data_format=self._data_format)
      elif "input_layer" in self._normalize:
        new_input = tf.contrib.layers.layer_norm(new_input)
      elif "input_dropout" in self._normalize:
        keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
        new_input = tf.nn.dropout(new_input, keep_prob=keep_prob)
      #cell state
      if "simple_fg" not in self._pvsst_circuit:
        new_cell = cell * forget_gate + cell_act_fn(new_input)
      else:
        new_cell = forget_gate * cell + (1-forget_gate) * cell_act_fn(new_input)
    else:
      raise ValueError("gating in:"
                           "{} not implemented".format(self._gating))
    #normalization
    if "inside_batch" in self._normalize:
      with tf.variable_scope(scope, reuse=False):
        new_cell = batch_norm(new_cell, training, data_format=self._data_format)
    elif "inside_layer" in self._normalize:
      new_cell = tf.contrib.layers.layer_norm(new_cell)
    elif "inside_dropout" in self._normalize:
      keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
      new_cell = tf.nn.dropout(new_cell, keep_prob=keep_prob)

    if "remove_OG" not in self._pvsst_circuit:
        if "out*" in self._gating:
          if 'flip_sign' in self._pvsst_circuit:
            output_gate = tf.sigmoid(output_gate)
          else:
            output_gate = tf.subtract(tf.constant(1.0), tf.sigmoid(output_gate))    
          output = cell_act_fn(new_cell) * output_gate
        elif "out-" in self._gating:
          if "simple_subt" in self._pvsst_circuit:
            if 'flip_sign' in self._pvsst_circuit:
              output = cell_act_fn(new_cell + output_gate)
            else:
              output = cell_act_fn(new_cell - output_gate)
          else:
            if 'flip_sign' in self._pvsst_circuit:
              output = cell_act_fn(new_cell + gate_act_fn(output_gate) - 1.0)
            else:
              output = cell_act_fn(new_cell - gate_act_fn(output_gate))
        else:
          raise ValueError("gating out:"
                               "{} not implemented".format(self._gating))
        new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    else:
        if "inside" in self._normalize:
          output = cell_act_fn(new_cell)
          new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
        else:
          output = new_cell
          new_state = rnn_cell_impl.LSTMStateTuple(new_cell, SST)
    
    return output, new_state


######################################
class EICell(rnn_cell_impl.RNNCell):
  """
  E, I
  """

  def __init__(self, params):
    """Construct EICell.

    Args:
      params: hyperparameters for EICell

    """
    super(EICell, self).__init__(params['name'])
    self._input_shape = params['input_shape']
    self._output_channels = params['output_channels']
    self._N_PV = params['N_PV']
    self._N_SST = params['N_SST']
    self._kernel_size = params['kernel_size']
    self._kernel_size_inh = params['kernel_size_inh'] 
    # kernel_size_inh is a list of kernel size for different connections
    # the meaning of each element is defined in cell function:
    # strides_fb is as calculated before in the file convinh_model.py
    self._strides_fb = params['strides_fb']
    self._strides = params['strides']
    # act_fn: string, activation function,eg:'gate_relu_cell_relu_kernel_abs'
    self._act_fn = params['act_fn']
    # normalize: string, specifying batch/layer normalization and its position
    self._normalize = params['normalize']
    # pvsst_circuit: string, eg: '','flip_sign','SstNoFF'
    self._pvsst_circuit = params['pvsst_circuit']
    # gating: string, gating mechanism, eg: 'in_mult_out_subt'
    self._gating = params['gating']
    self._data_format = params['data_format']
    self._skip_connection = False
    self._padding='SAME'
    self._total_output_channels = self._output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]
    if self._skip_connection and (self._strides != 1):
      raise ValueError("stride should be 1 if skip_connection is True")   
    # shape calculation
    kernel_H = tf.Dimension(self._kernel_size)
    strides_H = tf.Dimension(self._strides)
    state_H = self._input_shape[1]
    if self._padding == 'VALID':
      state_H = state_H - kernel_H + 1
    state_H = (state_H + strides_H - 1) // strides_H
    if self._data_format=='channels_last':
      state_size_E = tensor_shape.TensorShape(
          [state_H, state_H] + [self._output_channels])
      state_size_I = tensor_shape.TensorShape(
          [state_H, state_H] + [self._N_PV])
      self._state_size = rnn_cell_impl.LSTMStateTuple(state_size_E, state_size_I)
      self._output_size = tensor_shape.TensorShape(
          [state_H, state_H] + [self._total_output_channels])
    elif self._data_format=='channels_first':
      state_size_E = tensor_shape.TensorShape(
          [self._output_channels] + [state_H, state_H])
      state_size_I = tensor_shape.TensorShape(
          [self._N_PV] + [state_H, state_H])
      self._state_size = rnn_cell_impl.LSTMStateTuple(state_size_E, state_size_I)
      self._output_size = tensor_shape.TensorShape(
          [self._total_output_channels] + [state_H, state_H])
    else:
      raise ValueError("data_format not valid: {}".format(self._data_format)) 

      
  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size
  
  def zero_input(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroInput", values=[batch_size]):
      output = rnn_cell_impl._zero_state_tensors(self._input_shape, batch_size, dtype)
    return output
  
  def __call__(self, inputs, state, training, rnn_layer, scope):
    # state: last time step EI
    # inputs: current time step inputs
    ff_inputs = inputs['ff']
    if 'fb' in inputs:
      fb_inputs = inputs['fb']
    if 'norm_conv' in self._pvsst_circuit:
      separable=False
    else:
      separable=True
    E_state, I_state = state
    pre_neurons = [E_state, I_state, ff_inputs]
    pre_names = ['E','I','F']   # presynaptic neuron type
    post_names = ['E','I']  # postsynaptic neuron type
    num_neurons = [self._output_channels,self._N_PV]
    # kernel_sizes
    kernel_sizes = {}
    cnt = 0
    for pre in pre_names:
      for post in post_names:
        cur_w_name = '{}{}'.format(post,pre)
        kernel_sizes[cur_w_name] = self._kernel_size_inh[cnt]
        cnt += 1
    # synaptic currents
    I = {}
    abs_constraint = True if "kernel_abs" in self._act_fn else False
    with tf.variable_scope("_connections", reuse=tf.AUTO_REUSE):
      for i,pre_name in enumerate(pre_names):
        for j,post_name in enumerate(post_names):
          cur_use_bias = False
          if i==0:
            cur_use_bias = True
          cur_name = '{}{}'.format(post_name,pre_name)
          I[cur_name] = _conv([pre_neurons[i]],num_neurons[j], 
                               kernel_sizes[cur_name], self._strides,
                               use_bias=cur_use_bias, abs_constraint=abs_constraint,
                               separable=separable, 
                               kernel_name='kernel_{}'.format(cur_name),
                               bias_name='biases_{}'.format(cur_name),
                               data_format=self._data_format)
    fgs = []
    with tf.variable_scope("_forget_gate", reuse=tf.AUTO_REUSE):
      for i,name in enumerate(post_names):
        cur_fg = tf.get_variable('fg_{}'.format(name), (num_neurons[i]))
        cur_fg = tf.sigmoid(cur_fg+1.0)
        fgs.append(cur_fg)
    # neuron_states
    neurons = []
    if "cell_relu" in self._act_fn:
      cell_act_fn = tf.nn.relu
    elif "cell_elu" in self._act_fn:
      cell_act_fn = tf.nn.elu
    elif "cell_sigmoid" in self._act_fn:
      cell_act_fn = tf.sigmoid
    elif "cell_tanh" in self._act_fn:
      cell_act_fn = tf.tanh
    elif "cell_softplus" in self._act_fn:
      cell_act_fn = tf.math.softplus
    elif "cell_retanh" in self._act_fn:
      cell_act_fn = lambda a: tf.nn.relu(tf.tanh(a))
    elif "cell_power" in self._act_fn:
      power_y = tf.constant(float(re.findall("\d+\.\d+", self._act_fn)[0]))
      cell_act_fn = lambda a: tf.pow(a, power_y)
    elif "cell_repower" in self._act_fn:
      power_y = tf.constant(float(re.findall("\d+\.\d+", self._act_fn)[0]))
      cell_act_fn = lambda a: tf.pow(tf.nn.relu(a), power_y)
    else:
      raise ValueError("cell act fn:"
                           "{} not implemented".format(self._act_fn))
    # signs of currents
    signs = {'E':1.0,'I':-1.0,'F':1.0}
    if 'flip_sign' in self._pvsst_circuit:
      signs['E'] = -1.0
      signs['I'] = 1.0
    proj_neuron = 'I' if 'flip_proj' in self._pvsst_circuit else 'E'
    if rnn_layer > 1:
      signs['F'] = signs[proj_neuron]
      if 'FF_exc' in self._pvsst_circuit:
        signs['F'] = 1.0

    for i,name in enumerate(post_names):
      cur_I = signs['E']*I['{}E'.format(name)]\
              +signs['I']*I['{}I'.format(name)]\
              +signs['F']*I['{}F'.format(name)]
      # normalization
      if "remove_{}".format(name) not in self._normalize:
        if "batch" in self._normalize:
          with tf.variable_scope(scope+'_{}'.format(name), reuse=False):
            cur_I = batch_norm(cur_I, training, data_format=self._data_format)
        elif "layer" in self._normalize:
          cur_I = tf.contrib.layers.layer_norm(cur_I)
        elif "dropout" in self._normalize:
          keep_prob = float(re.findall("\d+\.\d+", self._normalize)[0])
          cur_I = tf.nn.dropout(cur_I, keep_prob=keep_prob)
      # cell state
      cur_neuron = fgs[i]*state[i]+(1-fgs[i])*cell_act_fn(cur_I)
      neurons.append(cur_neuron)
    if 'drop_I' in self._pvsst_circuit:
      neurons[1] = 0.0*neurons[1]
    tf.identity(neurons[0],'E_neuron')
    tf.identity(neurons[1],'I_neuron')
    return neurons[post_names.index(proj_neuron)], \
            rnn_cell_impl.LSTMStateTuple(neurons[0], neurons[1])
    

######################################
