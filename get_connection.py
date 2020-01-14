# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:19:20 2019

@author: sunminnie
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import sem
from sklearn.externals import joblib
import os
import platform
import pickle
from get_projs_acts import plot_projections
#from scipy.stats import sem, multivariate_normal
#from sklearn.mixture import GaussianMixture
#from sklearn.externals import joblib
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
rgb_colors = {'EX':(193/256,64/256,61/256),
              'PV':(2/256,148/256,165/256),
              'SST':(3/256,53/256,62/256),
              'FF':(167/256,156/256,147/256),
              'E':(193/256,64/256,61/256),
              'I':(2/256,148/256,165/256),}  

def name_to_params(name):
  params = {}
  params['n_time'] = 4 if 'cifar10' in name else 5
  params['num_rnn_layers'] = 2 if 'cifar10' in name else 3
  params['num_ff_layers'] = 2
  if ('EI' in name):
    params['RNN_Names'] = ['E','I']
    params['projection_names'] = ['EE','EI','II','IE','EF','IF']
    params['kernel_names'] = ['_connections/kernel_{}'.format(name) for name in params['projection_names']]
    params['fg_names'] = ['fg_E','fg_I']
    params['forget_gate_names'] = ['_forget_gate/fg_E', '_forget_gate/fg_I']
  else:
    params['RNN_Names'] = ['EX','PV','SST']
    params['projection_names'] = ['SST-EX','PV-EX','EX-EX','EX-SST','EX-PV','FF-SST','FF-PV','FF-EX']
    params['kernel_names'] = ['_input_gate/kernel_gate',
                              '_output_gate/kernel_gate',
                              '_new_input/kernel',
                              '_input_gate/kernel',
                              '_output_gate/kernel',
                              '_input_gate/kernel_ff',
                              '_output_gate/kernel_ff',
                              '_new_input/kernel_ff']
    if "remove_OG" in name:
      params['RNN_Names'] = ['EX','SST']
      params['projection_names'] = ['SST-EX','EX-EX','EX-SST','FF-SST','FF-EX']
      params['kernel_names'] = ['_input_gate/kernel_gate',
                                '_new_input/kernel',
                                '_input_gate/kernel',
                                '_input_gate/kernel_ff',
                                '_new_input/kernel_ff']
    if "mem_SST" in name:
      params['projection_names'].append('SST-SST')
      params['kernel_names'].append('_input_gate/kernel_II')
    params['fg_names'] = []
    params['forget_gate_names'] = []  
  params['dense_kernel'] = 'dense/kernel:0'
  params['suffix'] = 'without_abs' if 'without_abs' in name else ''
  if platform.system()=='Linux':
    params['export_dir'] = './{}'.format(name)
  else:
    params['export_dir'] = './exports/{}'.format(name)
  params['plot_name'] = name.replace('cifar10_','').replace('_export','')\
                        .replace('EI_','').replace('cell_','')\
                        .replace('Four_normal_relu_','')
  params['model_name'] = name
  return params


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


def get_projs(sess,graph,params):
  """
  get the projections from a single model
  """
  kernels = []
  for layer in range(1,params['num_rnn_layers']+1):
    for i,cur_name in enumerate(params['kernel_names']):
      if (params['k_ff'] is False) and 'kernel_ff' in cur_name:
        cur_name = cur_name.replace('kernel_ff','kernel')
      cur_kernel = get_kernel(graph, 
                                '{}/rnn_{}/{}'.format(params['vs'],layer,cur_name),suffix=params['suffix'])
      kernels.append(cur_kernel)
  kernels = sess.run(kernels)
  cnt = 0
  cur_projections = {}
  num_exc = 0
  for layer in range(1,params['num_rnn_layers']+1):
    for i,proj in enumerate(params['projection_names']):
      cur_kernel = kernels[cnt]
      if proj == 'EX-EX':
        num_exc = cur_kernel.shape[-1]
      if 'EX-' in proj:
        cur_kernel = cur_kernel[...,-num_exc:,:]
      if ('FF-' in proj) and (params['k_ff'] is False):
        cur_kernel = cur_kernel[...,:-num_exc,:]
      print(layer,proj,cur_kernel.shape)
      cur_projections['layer{}:{}'.format(layer, proj)] = cur_kernel
      cnt += 1
  return cur_projections 


def get_projections(model_name, visualize=False, n_init=5):
  """
  the final export for different initializations
  """
  params = name_to_params(model_name)
  projections = []
  dense_kernels = []
  if n_init is None:
    init_export = False
    n_init = 1
  else:
    init_export = True
  for init in range(n_init):
    with tf.Session(graph=tf.Graph()) as sess:
      if init_export:
        cur_export_dir = params['export_dir']+'_init_{}_export'.format(init)
      else:
        cur_export_dir = params['export_dir']+'_export'.format(init)
      if platform.system()=='Linux':
        cycleIDs = sorted(next(os.walk(cur_export_dir))[1])
        cur_export_dir = '{}/{}'.format(cur_export_dir,cycleIDs[-1])
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], cur_export_dir)
      graph = tf.get_default_graph()
      # version of the model
      vs = ''
      k_ff = False
      node_names=[n.name for n in graph.as_graph_def().node]
      for nn in node_names:
        if 'convinh_model' in nn:
          vs = 'convinh_model'
        if 'resnet_model' in nn:
          vs='resnet_model'
        if 'kernel_ff' in nn:
          k_ff = True
      params['vs'] = vs
      params['k_ff'] = k_ff
      print('vs, k_ff:',vs,' ',k_ff)
      # get projections or tensors
      # projections
      cur_projections = get_projs(sess,graph,params)
      projections.append(cur_projections)
      # dense_kernel
      cur_dense_kernel = graph.get_tensor_by_name('{}/dense/kernel:0'.format(params['vs']))
      print(cur_dense_kernel.shape)
      dense_kernels.append(cur_dense_kernel)
  if visualize:
    plot_projections(model_name, projections)
  return projections,dense_kernels  


def projs_to_number_of_connections(model_name, projs, thres,
                                   visualize=True, op="num"):
  """  
  projs: dict[init][projections], the return value of get_projections
  returns:
    dict[init][layer][neuron_type][to/from] is a numpy array shaped (num_neuron_type,)
    if "num" in op:
      "to" is num of effective weights to this neuron 
      "from" is num of effective weights from this neuron
    if "prop" in op:
      "to" is num of effective weights to this neuron / num of weights to this neuron
      "from" is num of effective weights from this neuron / num of weights from this neuron
  """
  n_init = len(projs)
  params = name_to_params(model_name)
  if len(params['RNN_Names'])==3:
    weights = {}
    weights['EX'] = {'to':['PV-EX','SST-EX','EX-EX'], 
                     'from':['EX-PV','EX-SST','EX-EX']} 
    weights['PV'] = {'to':['EX-PV'],
                     'from':['PV-EX']}
    weights['SST'] = {'to':['EX-SST'],
                      'from':['SST-EX']}
  elif len(params['RNN_Names'])==2:
    weights = {}
    weights['E'] = {'to':['EE','IE'],
                    'from':['EE','EI']}
    weights['I'] = {'to':['EI','II'],
                    'from':['IE','II']}
  else:
    raise ValueError("invalid RNN_Names")
  num_conns = {}
  for init in range(n_init):
    num_conns[init] = {}
    for layer in range(1,params['num_rnn_layers']+1):
      num_conns[init][layer] = {}
      for unit in params['RNN_Names']:
        num_conns[init][layer][unit] = {}
        for edge in ['to','from']:
          cur_num_list = []
          for proj_name in weights[unit][edge]:
            cur_proj = projs[init]['layer{}:{}'.format(layer, proj_name)]
            if len(cur_proj.shape)!=4:
              raise ValueError("cur_proj should be 4 dimensional")
            cur_proj = np.mean(cur_proj,(0,1))
            if edge=='to':
              cur_num = np.sum((cur_proj>thres),0)
              if "prop" in op:
                cur_num = cur_num.astype(float)/float(cur_proj.shape[0])
            elif edge=='from':
              cur_num = np.sum((cur_proj>thres),1)
              if "prop" in op:
                cur_num = cur_num.astype(float)/float(cur_proj.shape[1])
            cur_num_list.append(cur_num)
          # filter out
          cur_res = np.array(sum(cur_num_list))
          num_conns[init][layer][unit][edge]=cur_res
  if visualize:
    fig, axes = plt.subplots(2*len(params['RNN_Names']),params['num_rnn_layers'],
                             sharex=True,#sharey=True,
                             figsize=(2*len(params['RNN_Names']),2*params['num_rnn_layers']))
    fig.suptitle(params['plot_name'])
    for layer in range(1,params['num_rnn_layers']+1):
      cnt = 0
      axes[cnt,layer-1].set_title('Area {}'.format(layer+params['num_ff_layers']))
      for unit in params['RNN_Names']:
        for edge in ['to','from']:
          for init in range(n_init):
            x = num_conns[init][layer][unit][edge]
            n_x = x.shape[0]
            bins = np.linspace(x.min(), x.max(), int(n_x/2))
            axes[cnt,layer-1].hist(x, bins=bins, density=True, histtype='step')
          if layer==1:
            axes[cnt,layer-1].set_ylabel("{}:{}".format(unit,edge))
          if cnt==2*len(params['RNN_Names'])-1: 
            axes[cnt,layer-1].set_xlabel('{} of effective weights'.format(op))
          cnt += 1
  return num_conns


def unit_from_proj(proj):
  if 'EX-' in proj:
    pre = 'EX'
  elif 'SST-' in proj:
    pre = 'SST'
  elif 'PV-' in proj:
    pre = 'PV'
  else:
    raise ValueError("not pre neuron in {}".format(proj))
  if '-EX' in proj:
    post = 'EX'
  elif '-SST' in proj:
    post = 'SST'
  elif '-PV' in proj:
    post = 'PV'
  else:
    raise ValueError("not post neuron in {}".format(proj))
  return pre, post


def plot_filtered_projs(model_name, projections, conn_f, feature_f, alive_thres):
  params = name_to_params(model_name)
  fig, axes = plt.subplots(len(params['projection_names']),params['num_rnn_layers'],sharex=True,sharey=True,figsize=(6,4))
  if (conn_f is not None) and (feature_f is not None): 
    fig.suptitle("filtered:"+params['plot_name'])
  else:
    fig.suptitle(params['plot_name'])
  n_init = len(projections)
  gate_names = {'SST-EX':'IG->EX',
                'PV-EX':'OG->EX',
                'EX-EX':'EX->EX',
                'EX-SST':'EX->IG',
                'EX-PV':'EX->OG',
                'EE':'E->E',
                'EI':'I->E',
                'II':'I-I',
                'IE':'E->I'}
  for layer in range(1,params['num_rnn_layers']+1):
    for i,proj in enumerate(params['projection_names']):
      cur_name = 'layer{}:{}'.format(layer, proj)
      pre,post = unit_from_proj(proj)
      ax = axes[i,layer-1]
      for init in range(n_init):
        x = projections[init][cur_name]
        if len(x.shape)!= 4:
          raise ValueError("x should be 4-d")
        x = np.mean(x,(0,1))
        if (conn_f is not None) and (feature_f is not None): 
          pre_f = conn_f[init][layer][pre]*feature_f[init][(layer+1,pre)][alive_thres]
          post_f = conn_f[init][layer][post]*feature_f[init][(layer+1,post)][alive_thres]
          x = x[pre_f,:][:,post_f]
        x = x.flatten()
        x = np.log10(x)
        bins = np.linspace(x.min(), x.max(), 100)
        ax.hist(x, bins=bins, density=True, histtype='step')
      if i==0:
        ax.set_title('Area {}'.format(layer+params['num_ff_layers']))
      if i==len(params['projection_names'])-1: 
        ax.set_xlabel('logarithm to the base 10 of weights')
      if layer==1:
        ax.set_ylabel(gate_names[params['projection_names'][i]])
      ax.set_xlim([-20,0])
  return 
  



get_projections('cifar10_EICell_ratio_1_without_abs', visualize=True, n_init=None)
#projs_dict = joblib.load('projs.dict')
#for model_name in model_names:
#  params = name_to_params(model_name)
#  try:
#    projs = projs_dict[model_name]
#  except:
#    projs = get_projections(model_name, visualize=False, n_init=5)
#    
#  feature_filters = {}
#  n_init = len(projs)
#  for init in range(n_init):
#    ann_response_meta,feature_filter = joblib.load("./ann_features/cifar_actfilter/"\
#                               +"{}_init_{}_export.cifar_actfilter".format(model_name,init))
#    feature_filters[init]=feature_filter
#  
#  num_conns = projs_to_number_of_connections(model_name, projs, np.exp(-10),
#                                   visualize=False, op="num")
#  conn_filters = {}
#  for init in range(n_init):
#    conn_filters[init] = {}
#    for layer in range(1,params['num_rnn_layers']+1):
#      conn_filters[init][layer] = {}
#      for unit in params['RNN_Names']:
#        cur_unit_conns = num_conns[init][layer][unit]
#        cur_connected = np.array((cur_unit_conns['to']*cur_unit_conns['from'])>0)
#        conn_filters[init][layer][unit] = cur_connected
#  plot_filtered_projs(model_name, projs, conn_filters, feature_filters, -10)  
#  plot_filtered_projs(model_name, projs, None, None, -10)         