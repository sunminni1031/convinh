# -*- coding: utf-8 -*-
"""
Created on Sat May 11 14:54:25 2019

@author: sunminnie
"""
import tensorflow as tf
import os
import platform
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.stats import sem

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['savefig.dpi'] = 480


try:
  performance_dict = joblib.load('performance.dict')
  print("performance_dict loaded")
except:
  performance_dict = {}
  
if platform.system()=='Linux':
  models_dir = '.'
  model_dirs = next(os.walk(models_dir))[1]
  for model_dir in model_dirs:
    if 'cifar10_rd' not in model_dir:
      continue
#  num_list = {}
#  grids = [2,4,8,16,32,64,128,256]
#  for i_E in grids:
#    num_list[i_E] = [i_I/i_E for i_I in grids]
#  for num_EX in grids:
#    for cur in num_list[num_EX]:
#      for i in range(5):
#        model_dir ="cifar10_standard_pvsst_{}E_{}I_init_{}_model".format(num_EX,int(num_EX*cur),i)
        
    model_name = model_dir.replace('_model','')
    model_dir = models_dir+'/{}'.format(model_dir)
    
    if 'cifar10_rd' in model_dir:
      model_files = next(os.walk(model_dir))[2]
      for cur_file in model_files:
        if "events.out.tfevents" in cur_file:
          cur_file_path = model_dir+'/{}'.format(cur_file)
          acc = []
          for e in tf.train.summary_iterator(cur_file_path):
            for v in e.summary.value:
              if v.tag == 'train_accuracy_1':
                acc.append(v.simple_value)
          performance_dict[model_name]=acc
    else:
      eval_dirs = next(os.walk(model_dir))[1]
      if 'eval' in eval_dirs:
        eval_dir = model_dir+'/eval'
        eval_files = next(os.walk(eval_dir))[2]
        if len(eval_files)>0:
          eval_file = eval_files[-1]
          eval_file_path = eval_dir+'/{}'.format(eval_file)
          acc = []
          for e in tf.train.summary_iterator(eval_file_path):
            for v in e.summary.value:
              if v.tag == 'accuracy':
                acc.append(v.simple_value)
          performance_dict[model_name]=acc
    print('{} {} {}'.format(model_dir,len(acc),acc[-1]))
    
  joblib.dump(performance_dict,'performance.dict')      
  print("performance_dict saved")
else:
  plot_learning = 0
  if plot_learning:
    x = [i for i in range(252)]
    n_init = 5
    for model_name in ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_per_epoch']:
      fig = plt.figure(figsize=(3, 1.5))
      ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
      perf = []
      for init in range(n_init):
        cur_init_name = model_name+'_init_{}'.format(init)
        cur_init_perf = performance_dict[cur_init_name]
        print(len(cur_init_perf))
        if len(cur_init_perf) != 252:
          continue
          raise ValueError("number of epoches for model {} is not 252".format(cur_init_name))
        perf.append(cur_init_perf)
      perf = np.array(perf)
      y = np.mean(perf,0)
      yerr = sem(perf,0)
      ax.plot(x,y)
      ax.fill_between(x,y-yerr,y+yerr,alpha=0.2)
      ax.spines["right"].set_visible(False)
      ax.spines["top"].set_visible(False)
      ax.xaxis.set_ticks_position('bottom')
      ax.yaxis.set_ticks_position('left')
      #ax.set_title(model_name)
      ax.set_ylim([0,1])
      ax.set_ylabel("Test accuracy")
      ax.set_xlabel("Epoch")
      ax.set_title("Performance on CIFAR10: standard model")
      fig.savefig('./performance/training.pdf',bbox_inches = "tight")
  check_all = 1
  plot_all = 1
  n_init = 5
  title_name = None
  y_name = "Test accuracy"
  #"batch_remove_I",
  for save_name in [\
                    "rd_0.1"]:
                    #"standard_EI_NoNorm_reg"
                    #"standard_pvsst_number"
                    #"NoNorm_sign"]:
                    #"I=0"
                    #"imagenet_norm_conv"]:
#                    "batch_norm","layer_norm","to_EI","sign","drop","final_sum",\
#                    "gate","final_sum_simple_subt","final_sum_drop","normal_simple_subt"]:
    if save_name=="rd_0.1_reg":
      model_names = ["standard_pvsst_Norm","standard_pvsst_NoNorm",
                     "standard_EI_Norm","standard_EI_NoNorm"]
      model_names = ["cifar10_rd_0.1_"+cur+"_reg" for cur in model_names]
      x_names = ["standard","standard_NoBN","standardEI","standardEI_NoBN"]
      title_name = "Random-labeled 10% of Cifar10, With L2 Loss"
      y_name = "Train accuracy"
    if save_name=="rd_0.1":
      model_names = ["standard_pvsst_Norm","standard_pvsst_NoNorm",
                     "standard_EI_Norm","standard_EI_NoNorm"]
      model_names = ["cifar10_rd_0.1_"+cur for cur in model_names]
      x_names = ["standard","standard_NoBN","standardEI","standardEI_NoBN"]
      title_name = "Random-labeled 10% of Cifar10, No Regularization"
      y_name = "Train accuracy"
    if save_name=="rd_0.03_reg":
      model_names = ["standard_pvsst_Norm","standard_pvsst_NoNorm",
                     "standard_EI_Norm","standard_EI_NoNorm"]
      model_names = ["cifar10_rd_0.03_"+cur+"_reg" for cur in model_names]
      x_names = ["standard","standard_NoBN","standardEI","standardEI_NoBN"]
      title_name = "Random-labeled 3% of Cifar10, With L2 Loss"
      y_name = "Train accuracy"
    if save_name=="rd_0.03":
      model_names = ["standard_pvsst_Norm","standard_pvsst_NoNorm",
                     "standard_EI_Norm","standard_EI_NoNorm"]
      model_names = ["cifar10_rd_0.03_"+cur for cur in model_names]
      x_names = ["standard","standard_NoBN","standardEI","standardEI_NoBN"]
      title_name = "Random-labeled 3% of Cifar10, No Regularization"
      y_name = "Train accuracy"
    if "rd" not in save_name and "_reg" in save_name:#"standard_pvsst_reg":
      suffix_list = ["reg_0","reg_0.25","reg_0.5","reg_0.75",""]
      model_name = save_name.replace("_reg","")
      model_names = ["cifar10_"+model_name+suffix for suffix in suffix_list]
      file2disp = {"standard_pvsst_Norm":"standard",
                   "standard_pvsst_NoNorm":"standard_NoBN",
                   "standard_EI_Norm":"standardEI",
                   "standard_EI_NoNorm":"standardEI_NoBN"}
      x_names = ["RegCoef=0","RegCoef=5e-5","RegCoef=1e-4","RegCoef=1.5e-4",
                     "RegCoef=2e-4"]
      title_name = file2disp[model_name]
#      x_names = [file2disp[model_name]+'_'+suffix for suffix in suffix_disp]
    if save_name == "NoNorm_sign":
      model_names = ["cifar10_standard_pvsst_NoNorm",
                     "cifar10_standard_pvsst_NoNorm_NoAbs",
                     "cifar10_standard_pvsst_NoNorm_FlipSign",
                     "cifar10_standard_EI_NoNorm",
                     "cifar10_standard_EI_NoNorm_NoAbs",
                     "cifar10_standard_EI_NoNorm_FlipSign_FFexc"]
      x_names = ["standard_NoNorm",
                 "standard_NoNorm_NoConstraint",
                 "standard_NoNorm_InhProj",
                 "standard_EI_NoNorm",
                 "standard_EI_NoNorm_NoConstraint",
                 "standard_EI_NoNorm_InhProj"]
      
    if save_name == "I=0":
      model_names = ["cifar10_standard_pvsst_Norm",
                     "cifar10_standard_pvsst_Norm_dropSST",
                     "cifar10_standard_pvsst_Norm_dropPV",
                     "cifar10_standard_pvsst_Norm_dropSST_dropPV",
                     "cifar10_standard_EI_Norm",
                     "cifar10_standard_EI_Norm_dropI"]
      x_names = ["standard","standard_IG=0","standard_OG=0",
                 "standard_IG=0_OG=0",
                 "standardEI","standardEI_I=0"]
    if save_name == "NoNorm_I=0":
      model_names = ["cifar10_standard_pvsst_NoNorm",
                     "cifar10_standard_pvsst_NoNorm_dropSST",
                     "cifar10_standard_pvsst_NoNorm_dropPV",
                     "cifar10_standard_pvsst_NoNorm_dropSST_dropPV",
                     "cifar10_standard_EI_NoNorm",
                     "cifar10_standard_EI_NoNorm_dropI"]
      x_names = ["standard_NoNorm","standard_NoNorm_IG=0","standard_NoNorm_OG=0",
                 "standard_NoNorm_IG=0_OG=0",
                 "standardEI_NoNorm","standardEI_NoNorm_I=0"]
    if save_name=="imagenet_EI":
      model_names = ['imagenet_EI_batch_remove_I']
      x_names = ['imagenet_standard_EI']
    if save_name=="imagenet_norm_conv":
      model_names = ['imagenet_standard_pvsst_norm_conv']
      x_names = ['NormConv']
      n_init=5 
    if save_name=="num_grid":
      E_nums = [1,4,16,64,256]
      I_nums = [1,4,16,64,256]
      num_names = []
      for E_num in E_nums:
        for I_num in I_nums:
          if E_num==64:
            continue
          num_names.append('{}E_{}I'.format(E_num,I_num))
      model_names = ['cifar10_standard_pvsst_'+cur for cur in num_names]
      x_names = num_names
      n_init = 2
    if save_name=="norm_conv":
      model_names = ['cifar10_standard_pvsst_norm_conv']
      x_names = ['NormConv']
      n_init=10
    if save_name=="num_E":
      model_names = ['cifar10_standard_pvsst_16E_4I',
                     'cifar10_standard_pvsst_112E_28I',
                     'cifar10_standard_pvsst_128E_32I',
                     'cifar10_standard_pvsst_256E_64I']
      x_names = ['1/4xE','7/4xE','2xE','4xE']
    if save_name=="kI":
      model_names = ['cifar10_standard_pvsst_64E_16I',
                     'cifar10_standard_pvsst_kI_5',
                     'cifar10_standard_pvsst_kI_7']
      x_names = ['kI_3','kI_5','kI_7']
    if save_name == "conv":
      model_names = ['cifar10_standard_pvsst_norm_conv',
                     'cifar10_standard_pvsst_64E_16I',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25']
      x_names = ['NormConv','DepthConv','Standard']
    if save_name=="standard":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs',
                     'cifar10_cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_subt_1',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I_without_abs',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I_flip_sign']
      x_names = ['Standard','NoConstraint','InhPN','StandardEI','EI_NoConstraint','EI_InhPN']
    if save_name == "FF":
      model_names = ['cifar10_normal_relu_in_mult_out_subt_ratio_0.25_without_abs_FF_abs',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_FF_no_abs',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_FF_no_abs']
      x_names = ['NoConstrt_FFexc','FlipSign_FFNoConstrt','Standard_FFNoConstrt']
    if save_name == "batch_remove_I":
      model_names = ['cifar10_EI_withbias_ratio_0.25_batch_remove_I',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I_without_abs',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I_flip_sign',
                     'cifar10_EI_withbias_ratio_0.25_batch_remove_I_cell_tanh']
      x_names = ['ECurrBN_NoICurrBN','ECurrBN_NoICurrBN_NoConstraint', 
                 'ECurrBN_NoICurrBN_InhProj', 'ECurrBN_NoICurrBN_tanh']
    if save_name == "dropout":
      model_names = ['cifar10_normal_relu_in_mult_out_subt_ratio_0.25_without_norm',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_inside_dropout_0.9',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_inside_dropout_PV_dropout_0.9']
      x_names = ['Standars No Normalization','EXCellDropout0.9','EXCell+PVCurrDropout0.9']
    if save_name == "withff":
      model_names = ['cifar10_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_withff']
      x_names = ['Standard_flip_sign_withff']
    if save_name == "batch_norm":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_SST_batch',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_PV_batch',
                     'cifar10_EI_ratio_0.25_bias_back',
                     'cifar10_EI_withbias_ratio_0.25_remove_E_norm',
                     'cifar10_EI_withbias_ratio_0.25_remove_I_norm',
                     'cifar10_EI_withbias_ratio_0.25_remove_E_and_I_norm']
      x_names = ['EXCellBN','EXCellBN+IGCurrBN','EXCellBN+OGCurrBN',
                 'EICurrBN','ICurrBN','ECurrBN','EINoBN']
    if save_name == "layer_norm":
      model_names = ['cifar10_EI_withbias_ratio_0.25_layer_norm',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_norm',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_PV_layer_norm',
                     'cifar10_normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_SST_layer_norm']
      x_names = ['EICurrLN','EXCellLN','EXCellLN+OGLayerN','EXCellLN+IGLayerN']
    if save_name == "to_EI":
      model_names = ["cifar10_normal_relu_in_mult_out_subt_ratio_0.25_remove_OG_input_batch",
                     "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch",
                     "cifar10_normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST"]
      x_names = ["NoOG_EXCurrBN","NoOG_EXCurrBN_SSub","NoOG_EXCurrBN_SSub_SFg_RecurrIG"]
    if save_name == "sign":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs',
                     'cifar10_cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_subt_1',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh']
      x_names = ['Standard','NoConstraint','InhProj','tanh']
    if save_name == "drop":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_PV',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST',
                     'cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST_drop_PV']
      x_names = ['standard','- OG neuron','- IG neuron','- OG & IG neuron']
    if save_name == "final_sum":
      model_names = ['cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_sum_relu_in_mult_out_mult_ratio_0.25',
                     'cifar10_Four_sum_relu_in_subt_out_mult_ratio_0.25',
                     'cifar10_Four_sum_relu_in_subt_out_subt_ratio_0.25']
      x_names = ['SumT_MulSub','SumT_MulMul','SumT_SubMul','SumT_SubSub']
    if save_name == "gate":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_normal_relu_in_mult_out_mult_ratio_0.25',
                     'cifar10_Four_normal_relu_in_subt_out_mult_ratio_0.25',
                     'cifar10_Four_normal_relu_in_subt_out_subt_ratio_0.25']
      x_names = ['LastT_MulSub','LastT_MulMul','LastT_SubMul','LastT_SubSub']
    if save_name == "final_sum_simple_subt":
      model_names = ['cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25_simple_subt',
                     'cifar10_Four_sum_relu_in_mult_out_mult_ratio_0.25',
                     'cifar10_Four_sum_relu_in_subt_out_mult_ratio_0.25_simple_subt',
                     'cifar10_Four_sum_relu_in_subt_out_subt_ratio_0.25_simple_subt']
      x_names = ['SumT_SSub_MulSub','SumT_SSub_MulMul','SumT_SSub_SubMul','SumT_SSub_SubSub']
    if save_name == "final_sum_drop":
      model_names = ['cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25',
                     'cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25_drop_PV',
                     'cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25_drop_SST',
                     'cifar10_Four_sum_relu_in_mult_out_subt_ratio_0.25_drop_SST_drop_PV']
      x_names = ['SumT_Standard','SumT: - OG neuron','SumT: - IG neuron','SumT: - OG & IG neuron']
    if save_name == "normal_simple_subt":
      model_names = ['cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_simple_subt',
                     'cifar10_Four_normal_relu_in_mult_out_mult_ratio_0.25',
                     'cifar10_Four_normal_relu_in_subt_out_mult_ratio_0.25_simple_subt',
                     'cifar10_Four_normal_relu_in_subt_out_subt_ratio_0.25_simple_subt']
      x_names = ['LastT_SSub_MulSub','LastT_SSub_MulMul','LastT_SSub_SubMul','LastT_SSub_SubSub']
    if "number" in save_name:
      check_all=0
      plot_all=0
      model_name = save_name.replace("_number","")
      if model_name=="standard_EI_NoNorm":
        n_init=10
      grids = [1,4,16,64,256]
      matrx = np.zeros((len(grids),len(grids),n_init))
      for iE,nE in enumerate(grids):
        for iI,nI in enumerate(grids):
          cur_model_perf = []
          for init in range(n_init):
            cur_init_name = "cifar10_{}_{}E_{}I_init_{}".format(model_name,nE,nI,init)
            cur_init_perf = performance_dict[cur_init_name]
            if len(cur_init_perf) < 27:
              cur_init_perf = 0
            else:
              cur_init_perf = cur_init_perf[-1]
            cur_model_perf.append(cur_init_perf)
          matrx[len(grids)-1-iE,iI] = cur_model_perf
      fig,axes = plt.subplots(1,2,figsize=(6, 1.5))
#      matrx[1,0] = 10
      for func,ax in zip([np.max,np.mean],axes):
        img = ax.matshow(func(matrx,2),cmap='hot')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xticks(np.arange(len(grids)))
        ax.set_xticklabels(grids)#,rotation='vertical')
        ax.set_yticks(len(grids)-1-np.arange(len(grids)))
        ax.set_yticklabels(grids)
        ax.set_xlabel("Num of Inh Channels")
        ax.set_ylabel("Num of PN Channels")
        cur_title = "Maximal" if func==np.max else "Average"
        cur_title += " Performance Across {} Trials".format(n_init)
        ax.set_title(cur_title)
        fig.colorbar(img,ax=ax)
      save2fig = {"standard_EI_Norm":"StandardEI Model",
                  "standard_EI_NoNorm":"StandardEI Model Without Batch Normalization",
                  "standard_pvsst_NoNorm":"Standard Model Without Batch Normalization",
                  "standard_pvsst":"Standard Model"}
      fig.suptitle(save2fig[model_name],y=1.2)
      fig.savefig('./performance/performance_{}.pdf'.format(save_name),bbox_inches = "tight")
      
    if check_all:
      for model_name in model_names:
        fig = plt.figure(figsize=(3, 1.5))
        ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
        for init in range(n_init):
          cur_init_name = model_name+'_init_{}'.format(init)
          cur_init_perf = performance_dict[cur_init_name]
          len_x = len(cur_init_perf)
          if len_x <= 27:
            x = [0]+[10*i+1 for i in range(26)]
          else:
            x = [i for i in range(len_x)]
          if (len(cur_init_perf) != 27) and ('cifar' in model_name) and ('rd' not in model_name):
            continue
            raise ValueError("number of epoches for model {} is not 27".format(cur_init_name))
          ax.plot(x,cur_init_perf,label="init_{}:{}%".format(init,cur_init_perf[-1]))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(model_name)
        ax.legend()
        fig.savefig('./performance/performance_curve_{}.pdf'.format(save_name),bbox_inches = "tight")
    if plot_all:
      x = [0]+[10*i+1 for i in range(26)]
      fig = plt.figure(figsize=(3, 1.5))
      ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
      for i,model_name in enumerate(model_names):
        cur_model_perf = []
        for init in range(n_init):
          cur_init_name = model_name+'_init_{}'.format(init)
          cur_init_perf = performance_dict[cur_init_name]
          if len(cur_init_perf) < 27:
            #raise ValueError("number of epoches for model {} is {}, not 27".format(cur_init_name,len(cur_init_perf)))
            cur_init_perf = [0 for i in range(27)]
          cur_model_perf.append(cur_init_perf[-1])
        y_i = np.mean(cur_model_perf)
        p=ax.scatter(i,y_i,s=5)
        ax.annotate(np.around(y_i, decimals=3), (i,y_i))
        p_color=p.get_facecolor()[0]
        for init in range(n_init):
          ax.scatter(i,cur_model_perf[init],color=p_color,alpha=0.2,s=5)
      ax.spines["right"].set_visible(False)
      ax.spines["top"].set_visible(False)
      ax.xaxis.set_ticks_position('bottom')
      ax.yaxis.set_ticks_position('left')
      if save_name=="standard":
        ax.set_title("Performance on CIFAR-10")
      else:
        if title_name is None:
          title_name = save_name
        ax.set_title(title_name)
        ax.set_ylim([0,1])
        base_x = [i for i in range(len(x_names))]
        base_x = [base_x[0]-0.1]+base_x+[base_x[-1]+0.1]
        base_y = [0.8 for i in range(len(x_names)+2)]
        ax.plot(base_x, base_y, color='k',alpha=0.2)
      ax.set_ylabel(y_name)
      ax.set_xticks([i for i in range(len(x_names))])
      ax.set_xticklabels(x_names,rotation='vertical')
      fig.savefig('./performance/performance_{}.pdf'.format(save_name),bbox_inches = "tight")
      
    