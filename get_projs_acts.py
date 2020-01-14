# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:23:07 2018

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
    params['projection_names'] = ['EE','EI','II','IE']
    params['kernel_names'] = ['_connections/kernel_{}'.format(name) for name in params['projection_names']]
    params['fg_names'] = ['fg_E','fg_I']
    params['forget_gate_names'] = ['_forget_gate/fg_E', '_forget_gate/fg_I']
  else:
    params['RNN_Names'] = ['EX','PV','SST']
    params['projection_names'] = ['SST-EX','PV-EX','EX-EX','EX-SST','EX-PV']
    params['kernel_names'] = ['_input_gate/kernel_gate',
                              '_output_gate/kernel_gate',
                              '_new_input/kernel',
                              '_input_gate/kernel',
                              '_output_gate/kernel']
    if "remove_OG" in name:
      params['RNN_Names'] = ['EX','SST']
      params['projection_names'] = ['SST-EX','EX-EX','EX-SST']
      params['kernel_names'] = ['_input_gate/kernel_gate',
                                '_new_input/kernel',
                                '_input_gate/kernel']
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
    try:
      depth = graph.get_tensor_by_name('{}_depthwise:0'.format(name))
      point = graph.get_tensor_by_name('{}_pointwise:0'.format(name))
      kernel = tf.abs(depth*tf.squeeze(point))
    except:
      kernel = tf.abs(graph.get_tensor_by_name('{}:0'.format(name)))
  else:
    try:
      depth = graph.get_tensor_by_name('{}_depthwise_abs:0'.format(name))
      point = graph.get_tensor_by_name('{}_pointwise_abs:0'.format(name))
      kernel = depth*tf.squeeze(point)
    except:
      kernel = graph.get_tensor_by_name('{}_abs:0'.format(name))
  return kernel


def get_projs(sess,graph,params):
  kernels = []
  for layer in range(1,params['num_rnn_layers']+1):
    for i,cur_name in enumerate(params['kernel_names']):
      try:
        cur_kernel = get_kernel(graph, 
                                'convinh_model/rnn_{}/{}'.format(layer,cur_name),suffix=params['suffix'])
      except:
        cur_kernel = get_kernel(graph, 
                                'resnet_model/rnn_{}/{}'.format(layer,cur_name),suffix=params['suffix'])
      kernels.append(cur_kernel)
  kernels = sess.run(kernels)
  cnt = 0
  cur_projections = {}
  num_exc = 0
  for layer in range(1,params['num_rnn_layers']+1):
    for i,proj in enumerate(params['projection_names']):
      cur_kernel = kernels[cnt]
#        print(cur_kernel.shape)
      if proj == 'EX-EX':
        num_exc = cur_kernel.shape[-1]
      if 'EX-' in proj:
        cur_kernel = cur_kernel[...,-num_exc:,:]
      cur_projections['layer{}:{}'.format(layer, proj)] = cur_kernel
      cnt += 1
  return cur_projections 
  

def get_forget_gates(sess,graph,params):
  if 'I' not in params['RNN_Names']:
    raise ValueError("Not yet applicable to PV-SST-EX models")
  forget_gates = []
  for layer in range(1,params['num_rnn_layers']+1):
    for i,forget_gate_name in enumerate(params['forget_gate_names']):
      cur_forget_gate = graph.get_tensor_by_name('convinh_model/rnn_{}/{}:0'.format(layer,forget_gate_name))
      cur_forget_gate = tf.sigmoid(cur_forget_gate+1.0)
      forget_gates.append(cur_forget_gate)
  forget_gates = sess.run(forget_gates)
  cnt = 0
  cur_fgs = {}
  for layer in range(1,params['num_rnn_layers']+1):
    for i,fg_name in enumerate(params['fg_names']):
      cur_forget_gate = forget_gates[cnt]
      #print(cur_forget_gate.shape)
      cur_fgs['layer{}:{}'.format(layer, fg_name)] = cur_forget_gate
      cnt += 1
  return cur_fgs
  
  
def get_tensor(sess,graph,params,test_imgs,test_lbls):
  vs='convinh'
  try:
    inputs = graph.get_tensor_by_name("{}_model/inputs:0".format(vs))
  except:
    vs = 'resnet'
    inputs = graph.get_tensor_by_name("{}_model/inputs:0".format(vs))
  Feedforward = [graph.get_tensor_by_name(
        "{}_model/feedforward_layer_{}:0".format(vs,i+1)) 
        for i in range(params['num_ff_layers'])]
  final_dense = graph.get_tensor_by_name("{}_model/final_dense:0".format(vs))
  if 'SST' in params['RNN_Names']:
    PV = []
    SST = []
    EX=[]
    pool_id = ['_{}'.format(i) for i in range(17)]
    pool_id[pool_id=='_0']=''
    layer_pool_id = [2,5,8,11,14]
    for layer in range(params['num_rnn_layers']):
        cur_layer="{}_model/rnn_{}".format(vs,layer+1)
        try:
          cur_PV = [graph.get_tensor_by_name(cur_layer+"/_output_gate/PV:0")]
          cur_PV = cur_PV + [graph.get_tensor_by_name(cur_layer+
                           "_{}/_output_gate/PV:0".format(i+1)) for i in range(params['n_time']-1)]
        except:
          cur_PV = []
        cur_SST = [graph.get_tensor_by_name(cur_layer+"/_input_gate/SST:0")]
        cur_SST = cur_SST + [graph.get_tensor_by_name(cur_layer+
                         "_{}/_input_gate/SST:0".format(i+1)) for i in range(params['n_time']-1)]
        try:
          cur_EX = [graph.get_tensor_by_name(cur_layer+"/EX:0")]
          cur_EX = cur_EX + [graph.get_tensor_by_name(cur_layer+
                         "_{}/EX:0".format(i+1)) for i in range(params['n_time']-1)]
        except:
          cur_EX = [graph.get_tensor_by_name('{}_model/max_pooling2d'+
                                             '{}/MaxPool:0'.format(vs,pool_id[i])) for i in layer_pool_id] 
        layer_pool_id  = [i+1 for i in layer_pool_id]
        PV.append(cur_PV)
        SST.append(cur_SST)
        EX.append(cur_EX)
    neurons = [EX,PV,SST] if 'PV' in params['RNN_Names'] else [EX,SST]
  elif 'I' in params['RNN_Names']:
    E_neuron = []
    I_neuron = []
    for layer in range(params['num_rnn_layers']):
      cur_layer="{}_model/rnn_{}".format(vs,layer+1)
      cur_E = [graph.get_tensor_by_name(cur_layer+"/E_neuron:0")]
      cur_E = cur_E + [graph.get_tensor_by_name(cur_layer+
                       "_{}/E_neuron:0".format(i+1)) for i in range(params['n_time']-1)]
      cur_I = [graph.get_tensor_by_name(cur_layer+"/I_neuron:0")]
      cur_I = cur_I + [graph.get_tensor_by_name(cur_layer+
                       "_{}/I_neuron:0".format(i+1)) for i in range(params['n_time']-1)]
      E_neuron.append(cur_E)
      I_neuron.append(cur_I)
    neurons = [E_neuron,I_neuron]
  else:
    raise ValueError("RNN_Names not valid")
  
  if (test_imgs is not None) and (test_lbls is not None):
    cur_mean = {}
    cur_var = {}
    num_imgs = test_imgs.shape[0]
    for img,lbl in zip(test_imgs,test_lbls):
      logits,FF,RNN_Units = sess.run([final_dense,Feedforward,neurons], 
                                     feed_dict={inputs: np.expand_dims(img, axis=0)})
      print('{}: {} and {}'.format(params['export_dir'],np.argmax(logits),lbl))    
      for i, RNN_name in enumerate(params['RNN_Names']):
        for layer in range(params['num_rnn_layers']):
          for time_step in range(params['n_time']):
            unit_name = '{}_layer{}_time{}'.format(RNN_name, layer, time_step)
            if unit_name not in cur_mean:
              cur_mean[unit_name] = np.mean(RNN_Units[i][layer][time_step])/num_imgs
            else:
              cur_mean[unit_name] += np.mean(RNN_Units[i][layer][time_step])/num_imgs
            if unit_name not in cur_var:
              cur_var[unit_name] = np.var(RNN_Units[i][layer][time_step])/num_imgs
            else:
              cur_var[unit_name] += np.var(RNN_Units[i][layer][time_step])/num_imgs
    return cur_mean,cur_var
  else:
    return neurons
  

def get_projs_acts(model_name, test_imgs, test_lbls, visualize=False, n_init=5, learned=True):
  """
  the final export for different initializations
  """
  params = name_to_params(model_name)
  projections = []
  act_mean = []
  act_var = []
  fgs = []
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
        cur_ID = -1 if learned else 0
        cur_export_dir = '{}/{}'.format(cur_export_dir,cycleIDs[cur_ID])
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], cur_export_dir)
      graph = tf.get_default_graph()
      # projections
      cur_projections = get_projs(sess,graph,params)
      projections.append(cur_projections)
      # forget_gates
      if 'I' in params['RNN_Names']:
        cur_fgs = get_forget_gates(sess,graph,params)
        fgs.append(cur_fgs)
      # activities
      if test_imgs is not None:
        cur_mean,cur_var = get_tensor(sess,graph,params,test_imgs,test_lbls)
        act_mean.append(cur_mean)
        act_var.append(cur_var)
      
  if visualize:
    plot_projections(model_name, projections)
    if 'I' in params['RNN_Names']:
      plot_forget_gates(model_name,fgs)
    if test_imgs is not None:
      plot_end(model_name, act_mean)
  return projections,fgs,act_mean,act_var  


def plot_projections(model_name, projections, op='avg', thres=np.exp(-10),one_plot=True):
  params = name_to_params(model_name)
  n_init = len(projections)
  gate_names = {'SST-EX':'IG->EX',
                'PV-EX':'OG->EX',
                'EX-EX':'EX->EX',
                'EX-SST':'EX->IG',
                'EX-PV':'EX->OG',
                'SST-SST':'IG->IG',
                'FF-SST':'FF->IG',
                'FF-PV':'FF->OG',
                'FF-EX':'FF->EX',
                'EE':'E->E',
                'EI':'I->E',
                'II':'I-I',
                'IE':'E->I',
                'EF':'F->E',
                'IF':'F->I'}
  if one_plot:
    fig, axes = plt.subplots(len(params['projection_names']),params['num_rnn_layers'],sharex=True,sharey=True,figsize=(6,4))
    fig.suptitle(params['plot_name']+":{}".format(op))
    for layer in range(1,params['num_rnn_layers']+1):
      for i,proj in enumerate(params['projection_names']):
        cur_name = 'layer{}:{}'.format(layer, proj)
        ax = axes[i,layer-1]
        prop = []
        for init in range(n_init):
          if 'avg' in op:
            x = np.mean(projections[init][cur_name],(0,1))
          else:
            x = projections[init][cur_name]
          x = x.flatten()
          prop.append(np.mean(x>thres))
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
        ax.plot(-20,0,label=np.mean(np.array(prop)))
        ax.legend(loc='upper left')
  else:
    fig_inits = []
    axes_inits = []
    for init in range(n_init):
      fig, axes = plt.subplots(len(params['projection_names']),params['num_rnn_layers'],sharex=True,sharey=True,figsize=(6,4))
      fig.suptitle(params['plot_name']+"_init_{}:{}".format(init,op)) 
      for layer in range(1,params['num_rnn_layers']+1):
        for i,proj in enumerate(params['projection_names']):
          cur_name = 'layer{}:{}'.format(layer, proj)
          ax = axes[i,layer-1]
          prop = []
          if 'avg' in op:
            x = np.mean(projections[init][cur_name],(0,1))
          else:
            x = projections[init][cur_name]
          x = x.flatten()
          prop.append(np.mean(x>thres))
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
          ax.plot(-20,0,label=np.mean(np.array(prop)))
          ax.legend(loc='upper left')    
      fig_inits.append(fig)
      axes_inits.append(axes)        
  return 


def plot_end(model_name, act_mean, one_plot=True):
  params = name_to_params(model_name)
  if isinstance(act_mean, dict):
    n_init = None
  elif isinstance(act_mean, list):
    n_init = len(act_mean)
  else:
    raise ValueError('act_mean should be dict or list, instead of {}'.format(type(act_mean)))
  if one_plot:
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
    offset_x = 0
    label_plotted = []
    Gate_Names = {'PV':'OG','SST':'IG', 'EX':'EX', 'E':'E', 'I':'I'}
    for layer in range(params['num_rnn_layers']):
      for i, RNN_name in enumerate(params['RNN_Names']):
        if RNN_name in label_plotted:
          label=False
        else:
          label=True
          label_plotted.append(RNN_name)
        for time_step in range(params['n_time']):
          if n_init is None:
            plot_y = act_mean['{}_layer{}_time{}'.format(RNN_name, layer, time_step)]
            plot_y = np.log10(np.abs(plot_y)+1e-10)
            plot_yerr = 0
          elif n_init==1:
            plot_y = act_mean[0]['{}_layer{}_time{}'.format(RNN_name, layer, time_step)]
            plot_y = np.log10(np.abs(plot_y)+1e-10)
            plot_yerr = 0
          else:
            y = []
            for init in range(n_init):
              cur_y = act_mean[init]['{}_layer{}_time{}'.format(RNN_name, layer, time_step)]
              cur_y = np.log10(np.abs(cur_y)+1e-10)
              y.append(cur_y)
            plot_y = np.mean(y)
            plot_yerr = sem(y)
          bar_kw={'width':0.1,'alpha':1.0, 
                  'color':rgb_colors[RNN_name],
                  #'error_kw':{'ecolor': '0.3'},
                  'label':Gate_Names[RNN_name] if ((time_step==0) and label) else ''}
          ax.bar(offset_x, plot_y,yerr=plot_yerr,**bar_kw) 
          offset_x += 0.1
      offset_x += 0.2
    if len(params['RNN_Names'])==3:
      ax.set_xticks([0.5, 2.0])
      ax.set_xticklabels(['Area 3','Area 4'])
    else:
      ax.set_xticks([0.3, 1.3])
      ax.set_xticklabels(['Area 3','Area 4'])
    ax.set_ylabel('base-10 logarithm of average activity')
    ax.legend(bbox_to_anchor=(0,1.3,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
    ax.set_title(params['plot_name']) 
  else:
    for init in range(n_init):
      fig = plt.figure(figsize=(4, 2))
      ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
      offset_x = 0
      label_plotted = []
      Gate_Names = {'PV':'OG','SST':'IG', 'EX':'EX', 'E':'E', 'I':'I'}
      for layer in range(params['num_rnn_layers']):
        for i, RNN_name in enumerate(params['RNN_Names']):
          if RNN_name in label_plotted:
            label=False
          else:
            label=True
            label_plotted.append(RNN_name)
          for time_step in range(params['n_time']):
            plot_y = act_mean[init]['{}_layer{}_time{}'.format(RNN_name, layer, time_step)]
            plot_y = np.log10(np.abs(plot_y)+1e-10)
            plot_yerr = 0
            bar_kw={'width':0.1,'alpha':1.0, 
                    'color':rgb_colors[RNN_name],
                    #'error_kw':{'ecolor': '0.3'},
                    'label':Gate_Names[RNN_name] if ((time_step==0) and label) else ''}
            ax.bar(offset_x, plot_y,yerr=plot_yerr,**bar_kw) 
            offset_x += 0.1
        offset_x += 0.2
      if len(params['RNN_Names'])==3:
        ax.set_xticks([0.5, 2.0])
        ax.set_xticklabels(['Area 3','Area 4'])
      else:
        ax.set_xticks([0.3, 1.3])
        ax.set_xticklabels(['Area 3','Area 4'])
      ax.set_ylabel('base-10 logarithm of average activity')
      ax.legend(bbox_to_anchor=(0,1.3,1,0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=3)
      ax.set_title('{}_init_{}'.format(params['plot_name'],init))     
  return


def plot_forget_gates(model_name, fgs):
  params = name_to_params(model_name)
  fig, axes = plt.subplots(len(params['fg_names']),params['num_rnn_layers'],\
                           sharex=True,sharey=True,\
                           figsize=(3*params['num_rnn_layers'],len(params['fg_names'])))
  n_init = len(fgs)
  for layer in range(1,params['num_rnn_layers']+1):
    for i,fg_name in enumerate(params['fg_names']):
      cur_name = 'layer{}:{}'.format(layer, fg_name)
      ax = axes[i,layer-1]
      for init in range(n_init):
        x = fgs[init][cur_name]
        x = x.flatten() 
        bins = np.linspace(x.min(), x.max(), 10)
        ax.hist(x, bins=bins, density=True, histtype='step')
      if i==0:
        ax.set_title('{} \n Area {}'.format(params['plot_name'],layer+params['num_ff_layers']))
      if i==len(params['fg_names'])-1: 
        ax.set_xlabel('logarithm to the base 10 of weights')
      if layer==1:
        ax.set_ylabel(fg_name)
      ax.set_xlim([0,1])
    
      
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
  
      
def per_image_standardization(image):
  """Linearly scales `image` to have zero mean and unit norm.

  This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
  of all values in image, and
  `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

  """
  image = image.astype(float)
  num_pixels = np.prod(image.shape)
  image_mean = np.mean(image)
  variance = (np.mean(np.square(image))-np.square(image_mean))
  variance = np.amax([variance,0])
  stddev = np.sqrt(variance)
  min_stddev = 1/np.sqrt(num_pixels.astype(float))
  # Apply a minimum normalization that protects us against uniform images.
  pixel_value_scale = np.amax([stddev, min_stddev])
  pixel_value_offset = image_mean

  image = np.subtract(image, pixel_value_offset)
  image = np.divide(image, pixel_value_scale)
  return image    


#TODO
if __name__=='__main__':
#  model_names = ['cifar10_cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_subt_1',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign']
#  model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25']
  model_names = [\
#                 'cifar10_Four_normal_relu_in_mult_out_mult_ratio_0.25',
#                 'cifar10_Four_normal_relu_in_subt_out_mult_ratio_0.25',
#                 'cifar10_Four_normal_relu_in_subt_out_subt_ratio_0.25',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_PV',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST',
#                 'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST_drop_PV',
#                  'cifar10_Four_sum_relu_in_mult_out_mult_ratio_0.25',
#                  'cifar10_Four_sum_relu_in_subt_out_subt_ratio_0.25',
#                  'cifar10_EI_ratio_1_without_abs',
#                  'cifar10_EI_ratio_1_without_abs_flip_proj']
#                  "cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25",
                  "cifar10_standard_pvsst_norm_conv"]
#                  "cifar10_standard_pvsst_64E_16I"]
#                  "cifar10_normal_relu_in_mult_out_subt_ratio_0.25_check",
#                  "cifar10_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh_check",
#                  "cifar10_normal_relu_in_mult_out_subt_ratio_0.25_remove_OG_input_batch",
#                  "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch",
#                  "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST",
#                  "cifar10_EI_ratio_0.25_bias_back",
#                  "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST_SST_batch_init_0",
#                  "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST_SST_batch_init_1"]
  if platform.system()=='Linux':
    test_file_path = '/hdd/allen_dataset'
    #load
    #projs_dict = joblib.load('projs.dict')
    if not os.path.exists('./projs'):
      os.makedirs('./projs')
    activity_result_mean = joblib.load('activity_result_mean.dict')
    activity_result_var = joblib.load('activity_result_var.dict')
    # test_images
    test_file = '{}/cifar-10-python/cifar-10-batches-py/test_batch'.format(test_file_path)
    test_dict = unpickle(test_file)
    images, labels = test_dict[b'data'], test_dict[b'labels']
    images = np.transpose(images.reshape((-1,3,32,32)),(0,2,3,1)).astype(float)
    idx = np.arange(images.shape[0])
    np.random.shuffle(idx)
    num_images = 10   
    test_imgs = np.zeros((num_images,)+images.shape[1:])
    test_lbls = np.zeros(num_images)
    for i in range(num_images):
      test_imgs[i] = per_image_standardization(images[idx[i]])
      test_lbls[i] = labels[idx[i]]
    # calc projs and acts
    if 1:
      for model_name in model_names:
        n_init = None if 'init' in model_name else 5
        if "norm_conv" in model_name:
          n_init = 10
        projections,fgs,act_mean,act_var = get_projs_acts(model_name, test_imgs, test_lbls, visualize=False, n_init=n_init)
        joblib.dump(projections,'./projs/{}.projs_dict'.format(model_name))
        activity_result_mean[model_name] = act_mean
        activity_result_var[model_name] = act_var
    else:
      for model_name in model_names:
        n_init = None if 'init' in model_name else 5
        projections,fgs,act_mean,act_var = get_projs_acts(model_name, test_imgs, test_lbls, visualize=False, n_init=n_init,learned=False)
        joblib.dump(projections,'./projs/{}_initial.projs_dict'.format(model_name))
        activity_result_mean['{}_initial'.format(model_name)] = act_mean
        activity_result_var['{}_initial'.format(model_name)] = act_var
    # dump
    #joblib.dump(projs_dict,'projs.dict')
    joblib.dump(activity_result_mean,'activity_result_mean.dict')
    joblib.dump(activity_result_var,'activity_result_var.dict')
  else:
    #load
    projs_dict = joblib.load('projs.dict')
    activity_result_mean = joblib.load('activity_result_mean.dict')
    activity_result_var = joblib.load('activity_result_var.dict')
    for model_name in model_names:
      if 0:
        model_name += '_initial'
      try: 
        projections = projs_dict[model_name]
      except:
        projections = joblib.load('./projs/{}.projs_dict'.format(model_name))
      plot_projections(model_name, projections, op='flatten all dimensions',one_plot=False)#'avg across space')
      plot_end(model_name, activity_result_mean[model_name],one_plot=False)

      





