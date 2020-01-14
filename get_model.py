# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 21:22:49 2018

@author: sunminnie
"""

import tensorflow as tf
#import numpy as np
import convinh_model


def get_kernel(graph, name, suffix):
  if 'without_abs' in suffix: 
    depth = graph.get_tensor_by_name('{}_depthwise:0'.format(name))
    point = graph.get_tensor_by_name('{}_pointwise:0'.format(name))
    kernel = tf.abs(depth*tf.squeeze(point))
  else:
    depth = graph.get_tensor_by_name('{}_depthwise_abs:0'.format(name))
    point = graph.get_tensor_by_name('{}_pointwise_abs:0'.format(name))
    kernel = depth*tf.squeeze(point)
  return kernel


tf.reset_default_graph()
model_params={
      'data_format':'channels_last', 
      'filters':[16, 32, 64, 128],
      'ratio_PV': 1,
      'ratio_SST': 1,
      'conv_kernel_size':[3, 3, 3, 3],
      'conv_kernel_size_inh':[3, 3, 3],
      'conv_strides':[1, 1, 1, 1],
      'pool_size':[3, 3, 3, 3],
      'pool_strides':[1, 2, 2, 2],
      'num_ff_layers':2,
      'num_rnn_layers':2,
      'connection':'normal_ff_without_fb',
      'n_time':4,
      'cell_fn':'pvsst',
      'act_fn':'gate_relu_cell_relu_kernel_abs', 
      'pvsst_circuit':'',
      'gating':'in*_out-',
      'normalize':'inside_batch',
      'num_classes':10}
with tf.Session(graph=tf.Graph()) as sess:
  inputs = tf.placeholder(dtype=tf.float32, shape=(1,32,32,3))
  model = convinh_model.Model(model_params, dtype=tf.float32)
  logits = model(inputs,False)

  saver = tf.train.Saver()
  saver.restore(sess, "./ckpt/model.ckpt")
  print("Model restored.")
#  save_path = saver.save(sess, './ckpt/model.ckpt')
#  print("Model saved in path: %s" % save_path)
  
  graph = tf.get_default_graph()
  k1 = get_kernel(graph,'convinh_model/rnn_1/_input_gate/kernel_gate','')
  a1 = k1.eval()
  print(a1.shape)
  
  
#  for t in range(4):
#    for i in range(3):
#      with tf.variable_scope("rnn_{}".format(i), reuse=tf.AUTO_REUSE):
#        with tf.variable_scope("gate", reuse=tf.AUTO_REUSE):
#          kernel = tf.get_variable(name='kernel', shape=(1), dtype=tf.float32)
#          kernel = tf.identity(kernel,'core')
#          a = a + kernel
#  print(tf.global_variables())
#  sess.run(tf.global_variables_initializer())
#  test_0 = graph.get_tensor_by_name('rnn_1/gate/core:0')
#  test_1 = graph.get_tensor_by_name('rnn_1_1/gate/core:0')
#  tt_0 = graph.get_tensor_by_name('rnn_1/gate/kernel:0')
#  print(sess.run(test_0))
#  test_0 = graph.get_tensor_by_name('rnn_1/gate/kernel:0')
#  test_1 = graph.get_tensor_by_name('rnn_1/gate/kernel:0')