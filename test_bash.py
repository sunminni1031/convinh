import os
import re
import time
import _thread
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", help="device", type=int)
parser.add_argument("--act_fn", help="act_fn", type=str)
parser.add_argument("--cell_fn", help="cell_fn", type=str)
parser.add_argument("--gating", help="gating", type=str)
parser.add_argument("--ratio", help="ratio", type=float)
parser.add_argument("--num_EX", help='num_EX', type=int)
parser.add_argument("--k_I", help='kernel size of I', type=int)
parser.add_argument("--pvsst_circuit", help="pvsst_circuit", type=str)
parser.add_argument("--name", help="name", type=str)
parser.add_argument("--store_dir", help="store_dir", type=str)
parser.add_argument("--epoch",help="epoch",type=int)
parser.add_argument("--run_mode",help="run_mode",type=int)
parser.add_argument("--sleep_time",help="sleep_time",type=int)
parser.add_argument("--connection", help="connection", type=str)
parser.add_argument("--normalize", help="normalize", type=str)
parser.add_argument("--n_init",help="n_init", type=int)
parser.add_argument("--slurm",help="slurm",type=int)
parser.add_argument("--dataset",help="dataset",type=str)
args = parser.parse_args()
device = args.device
act_fn = args.act_fn
cell_fn = args.cell_fn
gating = args.gating
ratio = args.ratio
num_EX = args.num_EX
k_I = args.k_I
pvsst_circuit = args.pvsst_circuit
name = args.name
store_dir = args.store_dir
epoch = args.epoch
run_mode = args.run_mode
sleep_time = args.sleep_time
connection = args.connection
normalize = args.normalize
n_init = args.n_init
slurm = bool(args.slurm)
dataset=args.dataset
if dataset is None:
  dataset="cifar10"
if store_dir is None:
  store_dir = "/axsys/scratch/ctn/users/ms5724"
if n_init is None:
  n_init = 5
if num_EX is None:  
  num_EX = 64
if k_I is None:
  k_I = 3


#/axsys/scratch/ctn/users/ms5724
#/home/axsys/ms5724/
    
def cifar10_tests(device,
                     conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
                     num_ff_layers=2,
                     num_rnn_layers=2,
                     connection="normal_ff_without_fb",
                     n_time=4,
                     cell_fn="pvsst",
                     act_fn="gate_relu_cell_relu_kernel_abs",
                     gating="in*_out-",
                     normalize="inside_batch",
                     filters='\'16, 32, 64, 128\'',
                     ratio_PV=0.25,
                     ratio_SST=0.25,
                     pvsst_circuit='', # flip_sign, SstNoFF
                     conv_kernel_size='\'3, 3, 3, 3\'',
                     conv_strides='\'1, 1, 1, 1\'',
                     pool_size='\'3, 3, 3, 3\'',
                     pool_strides='\'1, 2, 2, 2\'',
                     num_classes=10,
                     seed=None,
                     weight_decay=None, #float 0.0002,0.0
                     data_aug=None, #integer 1,0
                     per_epoch=False,
                     store_dir = '/axsys/scratch/ctn/users/ms5724',
                     name="",
                     nohup=False,
                     dataset="cifar10",
                     batch_size=128,
                     slurm=False):
  test_cmd = "CUDA_VISIBLE_DEVICES={} ".format(device) if not slurm else ""
  test_cmd += "nohup " if nohup else ""
  data_size = None #float 1.0, 0.1
  if "cifar10" in dataset:
    datafile = dataset.replace("cifar10","cifar10_data")
    try:
      data_size = float(re.findall("\d+\.\d+", dataset)[0])
    except:
      data_size = None
    test_cmd += "python3 cifar10_main.py "#if epoch is None else " python3 cifar10_main1.py"
    test_cmd += "--data_dir={}/{} ".format(store_dir,datafile)
    test_cmd += "--batch_size={} ".format(batch_size)
    test_cmd += "--num_ckpt=5 "
  elif "imagenet" in dataset:
    test_cmd += "python3 imagenet_main128.py "
    test_cmd += "--data_dir=/mnt/sdb/tfrecord "
    test_cmd += "--batch_size=256 "
    test_cmd += "--num_ckpt=10 "
  else:
    raise ValueError("dataset {} not implemented".format(dataset))
  test_cmd += "--model_dir={}/models/{}_{}_model ".format(store_dir,dataset,name)
  test_cmd += "--export_dir={}/exports/{}_{}_export ".format(store_dir,dataset,name)
  test_cmd += "--data_format=channels_last "
  test_cmd += "--conv_kernel_size_inh={} ".format(conv_kernel_size_inh)
  test_cmd += "--num_ff_layers={} ".format(num_ff_layers)
  test_cmd += "--num_rnn_layers={} ".format(num_rnn_layers)
  test_cmd += "--connection={} ".format(connection)
  test_cmd += "--n_time={} ".format(n_time)
  test_cmd += "--cell_fn={} ".format(cell_fn)
  test_cmd += "--act_fn={} ".format(act_fn)
  test_cmd += "--gating={} ".format(gating)
  test_cmd += "--normalize={} ".format(normalize)
  test_cmd += "--filters={} ".format(filters)
  test_cmd += "--ratio_PV={} ".format(ratio_PV)
  test_cmd += "--ratio_SST={} ".format(ratio_SST)
  test_cmd += "--pvsst_circuit={} ".format(pvsst_circuit)
  test_cmd += "--conv_kernel_size={} ".format(conv_kernel_size)
  test_cmd += "--conv_strides={} ".format(conv_strides)
  test_cmd += "--pool_size={} ".format(pool_size)
  test_cmd += "--pool_strides={} ".format(pool_strides)
  test_cmd += "--num_classes={} ".format(num_classes)
  # tf_random_seed
  if seed is not None:
    test_cmd += "--seed={} ".format(seed)
  if weight_decay is not None:
    test_cmd += "--weight_decay={} ".format(weight_decay)
  if data_aug is not None:
    test_cmd += "--data_aug={} ".format(data_aug)
  if data_size is not None:
    test_cmd += "--data_size={} ".format(data_size)
  if per_epoch:
    test_cmd += "--epochs_between_evals=1 "
  test_cmd += ">{}.out 2>&1 & ".format(name) if nohup else ""
  # model_dir
  cur_dir = '{}/models/{}_{}_model/'.format(store_dir,dataset,name)
  if not os.path.exists(cur_dir):
    os.makedirs(cur_dir)
  # export_dir
  cur_export_dir = '{}/exports/{}_{}_export/'.format(store_dir,dataset,name)
  if not os.path.exists(cur_export_dir):
    os.makedirs(cur_export_dir)
  # run model
  if len(next(os.walk(cur_export_dir))[1]) < 27:
    print(test_cmd)
    if not slurm:
      # a record of hyperparmaters
      with open(cur_dir+'hyperparams.meta', 'w') as f:
        f.write(test_cmd)
    else:
      with open(cur_dir+name+'.sh', 'w') as f:
        f.write('#!/bin/bash\n'
                + '\n'
                + '#SBATCH --nodes=1\n'
                # + '#SBATCH --ntasks-per-node=1\n'
                #+ '#SBATCH --cpus-per-task=1\n'\
                #+ '#SBATCH --mem-per-cpu=7gb\n'\
                + '#SBATCH --partition=ctn\n'
                + '#SBATCH --gres=gpu:gtx1080:1\n'
                + '#SBATCH --output={}{}.out\n'.format(cur_dir,name)
                + '\n'
                + "export PYTHONPATH=$PYTHONPATH:/home/axsys/ms5724/EI_models" + '\n'
                + test_cmd + '\n'
                + '\n'
                + 'exit 0;\n')
      test_cmd = "sbatch {}{}.sh".format(cur_dir,name)
    if nohup:     
      _thread.start_new_thread(os.system, (test_cmd,))
      time.sleep(2)
    else:
      os.system(test_cmd)
  return

 
if run_mode==-2:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   cell_fn=cell_fn,#"EI",
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_init_{}".format(i),
                   nohup=False,
                   slurm=slurm,
                   weight_decay=0.0,################
                   data_aug=0,######################
                   dataset=dataset)
elif run_mode==-4:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   cell_fn=cell_fn,#"EI",
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_init_{}".format(i),
                   nohup=False,
                   slurm=slurm,
                   #weight_decay=0.0, regular weight decay############
                   data_aug=0,##########################
                   dataset=dataset)
elif run_mode==-6:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   cell_fn=cell_fn,#"EI",
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_per_epoch_init_{}".format(i),
                   nohup=False,
                   slurm=slurm,
                   #weight_decay=0.0, regular weight decay############
                   data_aug=0,##################
                   per_epoch=True,#####################
                   dataset=dataset)
elif run_mode==-3:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   cell_fn=cell_fn,#"EI",
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_init_{}".format(i),
                   nohup=False,
                   slurm=slurm,
                   weight_decay=0.0,
                   data_aug=0,
                   dataset="cifar10_rd_0.1",
                   batch_size=12) # don't train as well################
elif run_mode==-5:
  time.sleep(sleep_time)
  for ratio in [0,1/4,2/4,3/4]:
      for i in range(n_init):
        cifar10_tests(device,
                       cell_fn=cell_fn,#"EI",
                       store_dir=store_dir,
                       normalize=normalize,
                       name=name+"reg_{}_init_{}".format(ratio,i),
                       nohup=False,
                       slurm=slurm,
                       weight_decay=2e-4*ratio,#####################
                       dataset=dataset)    
# 30~40% < 40~45%
elif run_mode==0:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
                   connection=connection,
                   cell_fn=cell_fn,#"EI",
                   act_fn=act_fn,
                   gating=gating,
                   filters='\'16, 32, {}, {}\''.format(num_EX, 2*num_EX),
                   ratio_PV=ratio,
                   ratio_SST=ratio,
                   pvsst_circuit=pvsst_circuit, #
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_init_{}".format(i),
                   nohup=False,
                   slurm=slurm,
                   dataset=dataset) 
elif run_mode==1:
  time.sleep(sleep_time)
  for i in range(n_init):
    cifar10_tests(device,
                   conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
                   connection=connection,
                   cell_fn=cell_fn,#"EI",
                   act_fn=act_fn,
                   gating=gating,
                   filters='\'16, 32, {}, {}\''.format(num_EX, 2*num_EX),
                   ratio_PV=ratio,
                   ratio_SST=ratio,
                   pvsst_circuit=pvsst_circuit, #
                   store_dir=store_dir,
                   normalize=normalize,
                   name=name+"_init_{}".format(i+5),
                   nohup=False,
                   slurm=slurm,
                   dataset=dataset) 
else:
  raise ValueError("run mode not implemented")
##################################################### line of 2019.11.6
#elif run_mode==0:
#  time.sleep(sleep_time)
#  for i in range(n_init):
#    cifar10_tests(device,
#                   conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
#                   connection=connection,
#                   cell_fn=cell_fn,#"EI",
#                   act_fn=act_fn,
#                   gating=gating,
#                   filters='\'16, 32, {}, {}\''.format(num_EX, 2*num_EX),
#                   ratio_PV=ratio,
#                   ratio_SST=ratio,
#                   pvsst_circuit=pvsst_circuit, #
#                   store_dir=store_dir,
#                   normalize=normalize,
#                   name=name+"_init_{}".format(i),
#                   nohup=False)
#elif run_mode==1:
#  time.sleep(sleep_time)
#  gates = ["in-_out-","in*_out-","in-_out*"]
#  gate_names = ["in_subt_out_subt","in_mult_out_subt","in_subt_out_mult"]
#  for gate,gate_name in zip(gates,gate_names):
#    for i in range(5):
#      cifar10_tests(device,
#                     connection=connection,
#                     cell_fn=cell_fn,
#                     act_fn=act_fn,
#                     gating=gate,
#                     ratio_PV=ratio,
#                     ratio_SST=ratio,
#                     pvsst_circuit=pvsst_circuit, #
#                     store_dir=store_dir,
#                     name="Four_sum_relu_{}_ratio_0.25_init_{}".format(gate_name,i),
#                     nohup=False)
#elif run_mode==2:
#  time.sleep(sleep_time)
#  cifar10_tests(device,
#                 conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#                 num_ff_layers=2,
#                 num_rnn_layers=3,
#                 connection="normal_ff_without_fb",
#                 n_time=5,
#                 cell_fn=cell_fn,
#                 act_fn=act_fn,#"gate_relu_cell_relu_kernel_abs",
#                 gating="in*_out-",
#                 normalize=normalize,
#                 filters='\'64,128,256,256,512\'',
#                 ratio_PV=0.25,
#                 ratio_SST=0.25,
#                 pvsst_circuit=pvsst_circuit, # flip_sign, SstNoFF
#                 conv_kernel_size='\'7, 3, 3, 3, 3\'',
#                 conv_strides='\'2, 1, 1, 1, 1\'',
#                 pool_size='\'3, 3, 3, 3, 3\'',
#                 pool_strides='\'2, 2, 2, 2, 2\'',
#                 num_classes=1001,
#                 seed=None,
#                 store_dir = '/mnt/sdb/gy2259',
#                 name=name,#"normal_relu_in_mult_out_subt_ratio_0.25_init_{}".format(n_init),
#                 nohup=False,
#                 dataset = 'imagenet')
#elif run_mode==5:
#  time.sleep(sleep_time)
#  for i in range(n_init):
#    cifar10_tests(device,
#                   conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
#                   connection=connection,
#                   cell_fn=cell_fn,#"EI",
#                   act_fn=act_fn,
#                   gating=gating,
#                   filters='\'16, 32, {}, {}\''.format(num_EX, 2*num_EX),
#                   ratio_PV=ratio,
#                   ratio_SST=ratio,
#                   pvsst_circuit=pvsst_circuit, #
#                   store_dir=store_dir,
#                   normalize=normalize,
#                   name=name+"_init_{}".format(i+5),
#                   nohup=False)
#elif run_mode==7:
#  time.sleep(sleep_time)
##  pre_E_I_pair = [(1,1),(1,4),(1,16),(1,64),(1,256),
##                  (4,1/4),(4,1),(4,4),(4,16),(4,64),
##                  (16,1/16),(16,1),(16,4),(16,16),
##                  (256,1/256),(256,4/256),(256,16/256),(256,1)]
###  num_list[1] = [1,4,16,64,256]
###  num_list[4] = [1/4,1,4,16,64]
###  num_list[16] = [1/16,1,4,16]
###  num_list[256] = [1/256,4/256,16/256,1]  
#  
#  num_list = {}
#  grids = [2,4,8,16,32,64,128,256]
#  for i_E in grids:
#    num_list[i_E] = [i_I/i_E for i_I in grids]
#
#  for cur in num_list[num_EX]:
#    for i in range(n_init):
#      cifar10_tests(device,
#                     conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
#                     connection=connection,
#                     cell_fn=cell_fn,#"EI",
#                     act_fn=act_fn,
#                     gating=gating,
#                     filters='\'16, 32, {}, {}\''.format(num_EX, 2*num_EX),
#                     ratio_PV=cur,
#                     ratio_SST=cur,
#                     pvsst_circuit=pvsst_circuit, #
#                     store_dir=store_dir,
#                     normalize=normalize,
#                     name="standard_pvsst_{}E_{}I_init_{}".format(num_EX,int(num_EX*cur),i),
#                     nohup=False)  
#elif run_mode==8:
#  time.sleep(sleep_time)
#  n_Es = [2,4,8,16]
#  n_Is = [128,256,64,256]
#  n_ratios = [i/j for i,j in zip(n_Is,n_Es)]
#  n_inits = [1,3,2,2]
#  cifar10_tests(device,
#                 conv_kernel_size_inh='\'{},{},3,3,3,3\''.format(k_I, k_I), #'pv, sst, fb, hid'
#                 connection=connection,
#                 cell_fn=cell_fn,#"EI",
#                 act_fn=act_fn,
#                 gating=gating,
#                 filters='\'16, 32, {}, {}\''.format(n_Es[device], 2*n_Es[device]),
#                 ratio_PV=n_ratios[device],
#                 ratio_SST=n_ratios[device],
#                 pvsst_circuit=pvsst_circuit, #
#                 store_dir=store_dir,
#                 normalize=normalize,
#                 name="standard_pvsst_{}E_{}I_init_{}".format(n_Es[device],n_Is[device],n_inits[device]),
#                 nohup=False)    
#elif run_mode==12:      ## regularization
#  time.sleep(sleep_time)
#  for i in range(n_init):
#    cifar10_tests(device,
#                 reg=-1,
#                 store_dir = '/home/ms5724',
#                 name="standard_pvsst_noReg_init_{}".format(i))
#elif run_mode==13:
#  time.sleep(sleep_time)
##  for i in range(1):
##    cifar10_tests(device,
##                 reg=-1,
##                 store_dir = '/home/ms5724',
##                 name="standard_pvsst_noReg_check_{}".format(i))  
#  for i in [1.0,0.75,0.5,0.25,0.0]:
#    cifar10_tests(device,
#                 reg=i,
#                 store_dir = '/home/ms5724',
#                 name="standard_pvsst_Reg_{}_0".format(i))
#else:
#  raise ValueError("run mode not implemented")
# 1. act_fn="gate_relu_cell_relu_kernel_abs" or "gate_relu_cell_tanh_kernel_abs" 
#    or without "kernel_abs" 
# 2. pvsst_circuit='' or 'flip_sign'
# 3. ratio_PV, ratio_SST
# 4. gating="in*_out-"
# 5. init
#TODO: for test_init.py to run five models independently  
#for i in range(5):
#  cifar10_tests(device,
#                act_fn=act_fn,
#                gating=gating,
#                ratio_PV=ratio,
#                ratio_SST=ratio,
#                pvsst_circuit=pvsst_circuit,
#                name=name+"_init_{}".format(i),
#                store_dir=store_dir)
#time.sleep(2)  


#cifar10_tests(1,
#               conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#               num_ff_layers=2,
#               num_rnn_layers=2,
#               connection="normal_ff_without_fb",
#               n_time=4,
#               cell_fn="EI",
#               act_fn="gate_relu_cell_relu_kernel_abs",
#               gating="in*_out-",
#               normalize="inside_batch",
#               filters='\'16, 32, 64, 128\'',
#               ratio_PV=0.25,
#               ratio_SST=0.25,
#               pvsst_circuit='', # flip_sign, SstNoFF
#               conv_kernel_size='\'3, 3, 3, 3\'',
#               conv_strides='\'1, 1, 1, 1\'',
#               pool_size='\'3, 3, 3, 3\'',
#               pool_strides='\'1, 2, 2, 2\'',
#               num_classes=10,
#               seed=None,
#               store_dir = '/home/jl5377',
#               name="EIcell_test",
#               nohup=True)
#circs = ['','flip_sign','flip_proj','flip_sign_flip_proj']
#circs = ['sigmoid','tanh','softplus','retanh','power_1.0','power_1.5','power_2.0']
#for i,circ in enumerate(circs):
#  cifar10_tests(i,
#                 conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#                 num_ff_layers=2,
#                 num_rnn_layers=2,
#                 connection="normal_ff_without_fb",
#                 n_time=4,
#                 cell_fn="EI",
#                 act_fn="gate_relu_cell_{}_kernel_abs".format(circ),
#                 gating="in*_out-",
#                 normalize="inside_batch",
#                 filters='\'16, 32, 64, 128\'',
#                 ratio_PV=1,
#                 ratio_SST=1,
#                 pvsst_circuit='', #
#                 conv_kernel_size='\'3, 3, 3, 3\'',
#                 conv_strides='\'1, 1, 1, 1\'',
#                 pool_size='\'3, 3, 3, 3\'',
#                 pool_strides='\'1, 2, 2, 2\'',
#                 num_classes=10,
#                 seed=None,
#                 store_dir = '/home/jl5377',
#                 name="EICell_ratio_1_cell_{}".format(circ),
#                 nohup=True)

#
#if device==0:
#  cifar10_tests(0,
#                 conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#                 num_ff_layers=2,
#                 num_rnn_layers=2,
#                 connection="normal_ff_without_fb",
#                 n_time=4,
#                 cell_fn="EI",
#                 act_fn="gate_relu_cell_power_1.0_kernel_abs",
#                 gating="in*_out-",
#                 normalize="inside_batch",
#                 filters='\'16, 32, 64, 128\'',
#                 ratio_PV=1,
#                 ratio_SST=1,
#                 pvsst_circuit='flip_proj', #
#                 conv_kernel_size='\'3, 3, 3, 3\'',
#                 conv_strides='\'1, 1, 1, 1\'',
#                 pool_size='\'3, 3, 3, 3\'',
#                 pool_strides='\'1, 2, 2, 2\'',
#                 num_classes=10,
#                 seed=None,
#                 store_dir = store_dir,
#                 name="EICell_ratio_1_cell_power_1.0_flip_proj",
#                 nohup=False)
#  cifar10_tests(0,
#                 conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#                 num_ff_layers=2,
#                 num_rnn_layers=2,
#                 connection="normal_ff_without_fb",
#                 n_time=4,
#                 cell_fn="EI",
#                 act_fn="gate_relu_cell_tanh_kernel_abs",
#                 gating="in*_out-",
#                 normalize="inside_batch",
#                 filters='\'16, 32, 64, 128\'',
#                 ratio_PV=1,
#                 ratio_SST=1,
#                 pvsst_circuit='flip_proj', #
#                 conv_kernel_size='\'3, 3, 3, 3\'',
#                 conv_strides='\'1, 1, 1, 1\'',
#                 pool_size='\'3, 3, 3, 3\'',
#                 pool_strides='\'1, 2, 2, 2\'',
#                 num_classes=10,
#                 seed=None,
#                 store_dir = store_dir,
#                 name="EICell_ratio_1_cell_tanh_flip_proj",
#                 nohup=False)
#else:
#  cifar10_tests(1,
#                 conv_kernel_size_inh='\'3,3,3,3,3,3\'', #'pv, sst, fb, hid'
#                 num_ff_layers=2,
#                 num_rnn_layers=2,
#                 connection="normal_ff_without_fb",
#                 n_time=4,
#                 cell_fn="EI",
#                 act_fn="gate_relu_cell_relu",
#                 gating="in*_out-",
#                 normalize="inside_batch",
#                 filters='\'16, 32, 64, 128\'',
#                 ratio_PV=1,
#                 ratio_SST=1,
#                 pvsst_circuit='flip_proj', #
#                 conv_kernel_size='\'3, 3, 3, 3\'',
#                 conv_strides='\'1, 1, 1, 1\'',
#                 pool_size='\'3, 3, 3, 3\'',
#                 pool_strides='\'1, 2, 2, 2\'',
#                 num_classes=10,
#                 seed=None,
#                 store_dir = store_dir,
#                 name="EICell_ratio_1_without_abs_flip_proj",
#                 nohup=False)


#TODO: seed and init



