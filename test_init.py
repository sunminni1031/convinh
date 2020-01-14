import os
import time
import _thread

def run_test_bash(device=0,
                  act_fn="gate_relu_cell_relu_kernel_abs",
                  gating="in*_out-",
                  ratio=0.25,
                  num_EX=64,
                  k_I=3,
                  pvsst_circuit="",
                  name="",
                  cell_fn="pvsst",#'EI'
                  store_dir="/axsys/scratch/ctn/users/ms5724",epoch=None,run_mode=0,sleep_time=0,
                  connection="normal_ff_without_fb",
                  normalize="inside_batch", #"remove_I_batch"
                  n_init=5,
                  slurm=1,
                  dataset="cifar10"):
  test_cmd = "nohup python3 test_bash.py"
  test_cmd += " --device={}".format(device)
  test_cmd += " --act_fn={}".format(act_fn)
  test_cmd += " --gating={}".format(gating)
  test_cmd += " --ratio={}".format(ratio)
  test_cmd += " --num_EX={}".format(num_EX)
  test_cmd += " --k_I={}".format(k_I)
  test_cmd += " --pvsst_circuit={}".format(pvsst_circuit)
  test_cmd += " --name={}".format(name)
  test_cmd += " --cell_fn={}".format(cell_fn)
  test_cmd += " --store_dir={}".format(store_dir)
  test_cmd += " --epoch=1" if epoch is not None else ""
  test_cmd += " --run_mode={}".format(run_mode)
  test_cmd += " --sleep_time={}".format(sleep_time)
  test_cmd += " --connection={}".format(connection)
  test_cmd += " --normalize={}".format(normalize)
  test_cmd += " --n_init={}".format(n_init)
  test_cmd += " --slurm={}".format(slurm)
  test_cmd += " --dataset={}".format(dataset)
  test_cmd += " >{}_{}.out 2>&1 &".format(name,run_mode)
  print(test_cmd)
  _thread.start_new_thread(os.system, (test_cmd,))
  time.sleep(2)


################### 2019.11.14 even smaller dataset
run_test_bash(device=0,
              name="standard_pvsst_Norm_reg",
              cell_fn="pvsst",#'EI'
              run_mode=-6,
              normalize="inside_batch", #"remove_I_batch"
              n_init=5,
              slurm=1,
              dataset="cifar10_rd_0.03")

run_test_bash(device=0,
              name="standard_EI_Norm_reg",
              cell_fn="EI",#'EI'
              run_mode=-6,
              normalize="remove_I_batch", #"remove_I_batch"
              n_init=5,
              slurm=1,
              dataset="cifar10_rd_0.03")  
  
  
run_test_bash(device=0,
              name="standard_pvsst_Norm_reg",
              cell_fn="pvsst",#'EI'
              run_mode=-6,
              normalize="inside_batch", #"remove_I_batch"
              n_init=5,
              slurm=1,
              dataset="cifar10_rd_0.1")

run_test_bash(device=0,
              name="standard_EI_Norm_reg",
              cell_fn="EI",#'EI'
              run_mode=-6,
              normalize="remove_I_batch", #"remove_I_batch"
              n_init=5,
              slurm=1,
              dataset="cifar10_rd_0.1")

#run_test_bash(device=0,
#              name="standard_pvsst_Norm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-2,
#              normalize="inside_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_pvsst_NoNorm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-2,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_EI_Norm",
#              cell_fn="EI",#'EI'
#              run_mode=-2,
#              normalize="remove_I_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_EI_NoNorm",
#              cell_fn="EI",#'EI'
#              run_mode=-2,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")   
#  
#  
#run_test_bash(device=0,
#              name="standard_pvsst_Norm_reg",
#              cell_fn="pvsst",#'EI'
#              run_mode=-4,
#              normalize="inside_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_pvsst_NoNorm_reg",
#              cell_fn="pvsst",#'EI'
#              run_mode=-4,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_EI_Norm_reg",
#              cell_fn="EI",#'EI'
#              run_mode=-4,
#              normalize="remove_I_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")
#run_test_bash(device=0,
#              name="standard_EI_NoNorm_reg",
#              cell_fn="EI",#'EI'
#              run_mode=-4,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.03")  
################### 2019.11.14 regularization coefficient
#run_test_bash(device=0,
#              name="standard_pvsst_Norm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-5,
#              normalize="inside_batch", #"remove_I_batch"
#              slurm=1)
#run_test_bash(device=0,
#              name="standard_pvsst_NoNorm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-5,
#              normalize="", #"remove_I_batch"
#              slurm=1)
#run_test_bash(device=0,
#              name="standard_EI_Norm",
#              cell_fn="EI",#'EI'
#              run_mode=-5,
#              normalize="remove_I_batch", #"remove_I_batch"
#              slurm=1)
#run_test_bash(device=0,
#              name="standard_EI_NoNorm",
#              cell_fn="EI",#'EI'
#              run_mode=-5,
#              normalize="", #"remove_I_batch"
#              slurm=1)
################### 2019.11.11 random
#run_test_bash(device=0,
#              name="standard_pvsst_Norm_reg",
#              cell_fn="pvsst",#'EI'
#              run_mode=-4,
#              normalize="inside_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.1")
#run_test_bash(device=0,
#              name="standard_pvsst_NoNorm_reg",
#              cell_fn="pvsst",#'EI'
#              run_mode=-4,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.1")
#run_test_bash(device=0,
#              name="standard_EI_Norm_reg",
#              cell_fn="EI",#'EI'
#              run_mode=-4,
#              normalize="remove_I_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.1")
#run_test_bash(device=0,
#              name="standard_EI_NoNorm_reg",
#              cell_fn="EI",#'EI'
#              run_mode=-4,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd_0.1")

#run_test_bash(device=0,
#              name="standard_pvsst_Norm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-2,
#              normalize="inside_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd")
#run_test_bash(device=0,
#              name="standard_pvsst_NoNorm",
#              cell_fn="pvsst",#'EI'
#              run_mode=-2,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd")
#run_test_bash(device=0,
#              name="standard_EI_Norm",
#              cell_fn="EI",#'EI'
#              run_mode=-2,
#              normalize="remove_I_batch", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd")
#run_test_bash(device=0,
#              name="standard_EI_NoNorm",
#              cell_fn="EI",#'EI'
#              run_mode=-2,
#              normalize="", #"remove_I_batch"
#              n_init=5,
#              slurm=1,
#              dataset="cifar10_rd")

################### 2019.11.9
#for n_E in [1,4,16,64,256]:
#  for n_I in [1,4,16,64,256]:
#    if n_E==1 or n_I ==1:
#      run_test_bash(ratio=n_I/n_E,
#                    num_EX=n_E,
#                    name="standard_pvsst_{}E_{}I".format(n_E,n_I),
#                    cell_fn="pvsst",#'EI'
#                    normalize="inside_batch")
#for n_E in [1,4,16,64,256]:
#  for n_I in [1,4,16,64,256]:
#    run_test_bash(ratio=n_I/n_E,
#                  num_EX=n_E,
#                  name="standard_EI_NoNorm_{}E_{}I".format(n_E,n_I),
#                  cell_fn="EI",#'EI'
#                  normalize="",
#                  run_mode=1)
  
#run_test_bash(name="standard_EI_NoNorm_FlipSign_FFexc",
#              cell_fn="EI",#'EI'
#              run_mode=1,
#              normalize="",
#              pvsst_circuit="flip_sign_FF_exc")
################### 2019.11.8
#for n_E in [1,4,16,64,256]:
#  for n_I in [1,4,16,64,256]:
#    run_test_bash(ratio=n_I/n_E,
#                  num_EX=n_E,
#                  name="standard_pvsst_NoNorm_{}E_{}I".format(n_E,n_I),
#                  cell_fn="pvsst",#'EI'
#                  normalize="")#"remove_I_batch"
#for n_E in [1,4,16,64,256]:
#  for n_I in [1,4,16,64,256]:
#    run_test_bash(ratio=n_I/n_E,
#                  num_EX=n_E,
#                  name="standard_EI_NoNorm_{}E_{}I".format(n_E,n_I),
#                  cell_fn="EI",#'EI'
#                  normalize="")
################ 2019.11.7/8
#run_test_bash(name="standard_pvsst_NoNorm_NoAbs",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="",
#              act_fn="gate_relu_cell_relu")
#run_test_bash(name="standard_pvsst_NoNorm_FlipSign",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="",
#              pvsst_circuit="flip_sign")
#run_test_bash(name="standard_EI_NoNorm_NoAbs",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="",
#              act_fn="gate_relu_cell_relu")
#run_test_bash(name="standard_EI_NoNorm_FlipSign_FFexc",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="",
#              pvsst_circuit="flip_sign_FF_exc")
#run_test_bash(name="standard_EI_Norm_FlipSign_FFexc",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="remove_I_batch",
#              pvsst_circuit="flip_sign_FF_exc")
################ 2019.11.7
#run_test_bash(name="standard_pvsst_rd_label",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="inside_batch",
#              dataset="cifar10_rd")
#run_test_bash(name="standard_EI_rd_label",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="remove_I_batch",
#              dataset="cifar10_rd")
#run_test_bash(name="standard_pvsst_NoNorm_rd_label",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="",
#              dataset="cifar10_rd")
#run_test_bash(name="standard_EI_NoNorm_rd_label",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="",
#              dataset="cifar10_rd")
##########################################
#run_test_bash(pvsst_circuit="",
#              name="standard_pvsst_Norm",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="inside_batch")
#run_test_bash(pvsst_circuit="drop_SST",
#              name="standard_pvsst_Norm_dropSST",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="inside_batch")
#run_test_bash(pvsst_circuit="drop_PV",
#              name="standard_pvsst_Norm_dropPV",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="inside_batch")
#run_test_bash(pvsst_circuit="drop_SST_drop_PV",
#              name="standard_pvsst_Norm_dropSST_dropPV",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="inside_batch")
#
#run_test_bash(pvsst_circuit="",
#              name="standard_EI_Norm",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="remove_I_batch")
#run_test_bash(pvsst_circuit="drop_I",
#              name="standard_EI_Norm_dropI",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="remove_I_batch")  
###############2019.11.6
#run_test_bash(0,
#              name="standardModel",
#              run_mode=-1)
#run_test_bash(pvsst_circuit="",
#              name="standard_pvsst",
#              cell_fn="pvsst",#'EI'
#              store_dir="/axsys/scratch/ctn/users/ms5724",
#              run_mode=0,
#              normalize="inside_batch",
#              n_init=5,
#              slurm=True)
#run_test_bash(pvsst_circuit="",
#              name="standard_pvsst_NoNorm",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="")
#run_test_bash(pvsst_circuit="drop_SST",
#              name="standard_pvsst_NoNorm_dropSST",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="")
#run_test_bash(pvsst_circuit="drop_PV",
#              name="standard_pvsst_NoNorm_dropPV",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="")
#run_test_bash(pvsst_circuit="drop_SST_drop_PV",
#              name="standard_pvsst_NoNorm_dropSST_dropPV",
#              cell_fn="pvsst",#'EI'
#              run_mode=0,
#              normalize="")

#run_test_bash(pvsst_circuit="",
#              name="standard_EI_NoNorm",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="")
#run_test_bash(pvsst_circuit="drop_I",
#              name="standard_EI_NoNorm_dropI",
#              cell_fn="EI",#'EI'
#              run_mode=0,
#              normalize="")


################ 2019.9.20
#run_test_bash(device=0,
#              run_mode=12,
#              sleep_time=0,
#              n_init=5)
#run_test_bash(device=1,
#              run_mode=13,
#              sleep_time=0,
#              n_init=1)
############### 2019.8.21
#for i in range(4):
#  run_test_bash(i, #13456
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="in*_out-",
#                ratio=0,
#                num_EX=0,
#                k_I=3,
#                pvsst_circuit="",
#                name="standard_pvsst_device{}".format(i),
#                cell_fn="pvsst",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=8,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="inside_batch",
#                n_init=0)   
############## 2019.8.15
#for i in range(5):
#  run_test_bash(i,
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="in*_out-",
#                ratio=0.25,
#                num_EX=64,
#                k_I=3,
#                pvsst_circuit="flip_sign",
#                name="standard_pvsst_flip_sign_init_{}".format(i+1),
#                cell_fn="pvsst",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="inside_batch",
#                n_init=1)
#run_test_bash(5,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="flip_sign",
#              name="standard_pvsst_flip_sign",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
############## 2019.8.15/ 8.10
### num_EX
# 2,4,8,16,32,64,128,256
#for i in range(8):
#  run_test_bash(i, #13456
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="in*_out-",
#                ratio=0,
#                num_EX=2**(i+1),
#                k_I=3,
#                pvsst_circuit="",
#                name="standard_pvsst_{}E".format(2**(i+1)),
#                cell_fn="pvsst",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="inside_batch",
#                n_init=5) 
# 8,16,32
#for i in range(3):
#  run_test_bash(i, #13456
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="in*_out-",
#                ratio=0,
#                num_EX=2**(i+3),
#                k_I=3,
#                pvsst_circuit="",
#                name="standard_pvsst_{}E".format(2**(i+3)),
#                cell_fn="pvsst",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="inside_batch",
#                n_init=5)
############## 2019.8.10
#gpu1
#for i in range(4):
#  run_test_bash(i,
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="",
#                ratio=0.25,
#                num_EX=64,
#                k_I=3,
#                pvsst_circuit="",
#                name="EI_batch_remove_I_init_{}".format(i+1),
#                cell_fn="EI",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="batch_remove_I",
#                n_init=1)
#gpu3
#for i in range(4,8):
#  run_test_bash(i,
#                act_fn="gate_relu_cell_relu",
#                gating="in*_out-",
#                ratio=0.25,
#                num_EX=64,
#                k_I=3,
#                pvsst_circuit="",
#                name="standard_pvsst_without_abs_init_{}".format(i-3),
#                cell_fn="pvsst",#'EI'
#                store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="inside_batch",
#                n_init=1)
############## 2019.7.31
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv_init_1",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv_init_2",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv_init_3",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv_init_4",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
############# 2019.7.29
### num_EX
#run_test_bash(0, #13456
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=1,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_1E",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1) 
#run_test_bash(1, #13456
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=4,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_4E",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(2, #13456
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=16,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_16E",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(3, #13456
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=256,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_256E",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=7,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
### normal convolution
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=5,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)  
  
#############2019.7.28
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=112,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_112E_28I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=128,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_128E_32I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=256,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_256E_64I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=16,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_16E_4I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)

############## 2019.7.27
#run_test_bash(6,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="flip_sign",
#              name="standard_pvsst_flip_sign_init_0",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(7,
#              act_fn="gate_relu_cell_relu",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_without_abs_init_0",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=2,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=5,
#              pvsst_circuit="",
#              name="standard_pvsst_kI_5",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(5,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=7,
#              pvsst_circuit="",
#              name="standard_pvsst_kI_7",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=32,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_32E_8I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=48,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_48E_12I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=80,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_80E_20I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=96,
#              k_I=3,
#              pvsst_circuit="",
#              name="standard_pvsst_96E_24I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
############# 2019.7.26
# kernel size of I
####### gpu1 cluster
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=5,
#              pvsst_circuit="",
#              name="standard_pvsst_kI=5",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=7,
#              pvsst_circuit="",
#              name="standard_pvsst_kI=7",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
##################### normal convolution
#run_test_bash(5,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              k_I=3,
#              pvsst_circuit="norm_conv",
#              name="standard_pvsst_norm_conv",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)
# varying the number of EX
#32 0.5 16.0
#-48 0.3333333333333333 16.0
#-64 0.25 16.0
#-80 0.2 16.0
#96 0.16666666666666666 16.0
#112 0.14285714285714285 16.0
#128 0.125 16.0
#########
#gpu1 cluster
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              num_EX=64,
#              pvsst_circuit="",
#              name="standard_pvsst_64E_16I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5)   
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.5,
#              num_EX=32,
#              pvsst_circuit="",
#              name="standard_pvsst_32E_16I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5) 
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.16666666666666666,
#              num_EX=96,
#              pvsst_circuit="",
#              name="standard_pvsst_96E_16I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/gy2259",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5) 
#########  
#21:30: 129.236.163.48
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.3333333333333333,
#              num_EX=48,
#              pvsst_circuit="",
#              name="standard_pvsst_48E_16I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5) 
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.2,
#              num_EX=80,
#              pvsst_circuit="",
#              name="standard_pvsst_80E_16I",
#              cell_fn="pvsst",#'EI'
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=5) 
##################################### 
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25")
#run_test_bash(1,"gate_relu_cell_relu","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs")
#run_test_bash(2,"gate_relu_cell_tanh_kernel_abs","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh")
#run_test_bash(3,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"flip_sign","Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign")
#run_test_bash(4,"gate_relu_cell_relu_kernel_abs","in*_out-",0.125,"","Four_normal_relu_in_mult_out_subt_ratio_0.125")

#run_test_bash(5,"gate_relu_cell_relu_kernel_abs","in*_out-",0.5,"","Four_normal_relu_in_mult_out_subt_ratio_0.5")
#run_test_bash(6,"gate_relu_cell_relu_kernel_abs","in*_out-",0.0625,"","Four_normal_relu_in_mult_out_subt_ratio_0.0625")
#run_test_bash(7,"gate_relu_cell_relu_kernel_abs","in*_out-",1,"","Four_normal_relu_in_mult_out_subt_ratio_1")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",0.03125,"","Four_normal_relu_in_mult_out_subt_ratio_0.03125","/mnt/sdb/jl5377")
#run_test_bash(1,"gate_relu_cell_relu_kernel_abs","in*_out-",2,"","Four_normal_relu_in_mult_out_subt_ratio_2","/mnt/sdb/jl5377")
#run_test_bash(2,"gate_relu_cell_relu_kernel_abs","in*_out-",4,"","Four_normal_relu_in_mult_out_subt_ratio_4","/mnt/sdb/jl5377")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",0.015625,"","Four_normal_relu_in_mult_out_subt_ratio_0.015625","/mnt/sdb/jl5377")
#run_test_bash(1,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25_per_epoch",epoch=1)
#run_test_bash(0,"","in*_out-",1,"","EI_ratio_power_1.0_tanh_flip_proj",store_dir="/home/ms5724")
#run_test_bash(1,"","in*_out-",1,"","EI_ratio_without_abs_flip_proj",store_dir="/home/ms5724")
  
#run_test_bash(1,"gate_relu_cell_relu_kernel_abs","",1,"","EI_ratio_1",store_dir="/home/jl5377")
#run_test_bash(2,"gate_relu_cell_relu_kernel_abs","",1,"flip_proj","EI_ratio_1_flip_proj",store_dir="/home/jl5377")
#run_test_bash(3,"gate_relu_cell_power_1.0_kernel_abs","",1,"","EI_ratio_1_cell_power_1.0",store_dir="/home/jl5377")
#run_test_bash(4,"gate_relu_cell_tanh_kernel_abs","",1,"","EI_ratio_1_cell_tanh",store_dir="/home/jl5377")
#run_test_bash(5,"gate_relu_cell_relu","",1,"","EI_ratio_1_without_abs",store_dir="/home/jl5377")
#run_test_bash(6,"gate_relu_cell_relu_kernel_abs","",0.5,"","EI_ratio_0.5",store_dir="/home/jl5377")
#run_test_bash(7,"gate_relu_cell_relu_kernel_abs","",0.25,"","EI_ratio_0.25",store_dir="/home/jl5377")
  
#run_test_bash(0,"gate_relu_cell_power_1.0_kernel_abs","",1,"flip_proj","EI_ratio_1_cell_power_1.0_flip_proj",store_dir="/home/jl5377")
#run_test_bash(1,"gate_relu_cell_tanh_kernel_abs","",1,"flip_proj","EI_ratio_1_cell_tanh_flip_proj",store_dir="/home/jl5377")  
#run_test_bash(3,"gate_relu_cell_relu","",1,"flip_proj","EI_ratio_1_without_abs_flip_proj",store_dir="/home/jl5377")
#run_test_bash(5,"gate_relu_cell_relu_kernel_abs","",0.125,"","EI_ratio_0.125",store_dir="/home/jl5377")
#run_test_bash(6,"gate_relu_cell_relu_kernel_abs","",0.0625,"","EI_ratio_0.0625",store_dir="/home/jl5377")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","",0.03125,"","EI_ratio_0.03125",store_dir="/home/ms5724")
#run_test_bash(1,"gate_relu_cell_relu_kernel_abs","",0.015625,"","EI_ratio_0.015625",store_dir="/home/ms5724")


#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","",0.25,"flip_proj","EI_ratio_0.25_flip_proj",store_dir="/home/ms5724")
#run_test_bash(1,"gate_relu_cell_tanh_kernel_abs","",0.25,"","EI_ratio_0.25_cell_tanh",store_dir="/home/ms5724")
#run_test_bash(0,"gate_relu_cell_relu","",0.25,"","EI_ratio_0.25_without_abs",store_dir="/home/jl5377")
#run_test_bash(4,"gate_relu_cell_relu","",0.25,"flip_proj","EI_ratio_0.25_without_abs_flip_proj",store_dir="/home/jl5377")
#run_test_bash(7,"gate_relu_cell_tanh_kernel_abs","",0.25,"flip_proj","EI_ratio_0.25_cell_tanh_flip_proj",store_dir="/home/jl5377")
#run_test_bash(0,"gate_relu_cell_power_1.0_kernel_abs","",0.25,"","EI_ratio_0.25_cell_power_1.0",store_dir="/home/jl5377")
#run_test_bash(7,"gate_relu_cell_power_1.0_kernel_abs","",0.25,"flip_proj","EI_ratio_0.25_cell_power_1.0_flip_proj",store_dir="/home/jl5377")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","",0.25,"flip_sign_subt_1",\
#              "cifar10_Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_subt_1",cell_fn="pvsst",store_dir="/home/ms5724")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in-_out-",0.25,"",\
#              "Four_normal_relu_in_subt_out_subt_ratio_0.25",cell_fn="pvsst",store_dir="/home/ms5724")
#run_test_bash(4,"gate_relu_cell_relu_kernel_abs","in*_out*",0.25,"",\
#              "Four_normal_relu_in_mult_out_mult_ratio_0.25",cell_fn="pvsst",store_dir="/home/jl5377")
#run_test_bash(5,"gate_relu_cell_relu_kernel_abs","in-_out*",0.25,"",\
#              "Four_normal_relu_in_subt_out_mult_ratio_0.25",cell_fn="pvsst",store_dir="/home/jl5377")
#run_test_bash(2,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"drop_SST",\
#              "Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST",cell_fn="pvsst",store_dir="/home/jl5377")
#run_test_bash(7,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"drop_PV",\
#              "Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_PV",cell_fn="pvsst",store_dir="/home/jl5377")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",-1,"","Four_normal_relu_in_mult_out_subt",cell_fn="pvsst",\
#                  store_dir="/home/ms5724",epoch=None,run_mode=5,sleep_time=0)
#run_test_bash(2,"gate_relu_cell_relu_kernel_abs","in*_out-",-1,"","Four_normal_relu_in_mult_out_subt",cell_fn="pvsst",\
#                  store_dir="/home/jl5377",epoch=None,run_mode=6,sleep_time=0)
#run_test_bash(4,"gate_relu_cell_relu_kernel_abs","in*_out-",-1,"","Four_normal_relu_in_mult_out_subt",cell_fn="pvsst",\
#                  store_dir="/home/jl5377",epoch=None,run_mode=7,sleep_time=0)
#run_test_bash(5,"gate_relu_cell_relu_kernel_abs","in*_out-",-1,"","Four_normal_relu_in_mult_out_subt",cell_fn="pvsst",\
#                  store_dir="/home/jl5377",epoch=None,run_mode=8,sleep_time=0)
#run_test_bash(7,"gate_relu_cell_relu_kernel_abs","in*_out-",-1,"","Four_normal_relu_in_mult_out_subt",cell_fn="pvsst",\
#                  store_dir="/home/jl5377",epoch=None,run_mode=9,sleep_time=0)
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"drop_SST_drop_PV",\
#                  "Four_normal_relu_in_mult_out_subt_ratio_0.25_drop_SST_drop_PV",\
#                  cell_fn="pvsst",store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0)  
#run_test_bash(0,
#              gating="in*_out*",
#              name="Four_sum_relu_in_mult_out_mult_ratio_0.25",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb_final_sum")
#run_test_bash(1,
#              gating="",
#              name="Four_sum_relu_gates_ratio_0.25",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=1,sleep_time=0,
#              connection="normal_ff_without_fb_final_sum")
#devices = [1,2,4]
#circs = ['drop_PV','drop_SST','drop_SST_drop_PV']
#for device,circ in zip(devices,circs):
#  run_test_bash(device,
#                gating="in*_out-",
#                pvsst_circuit=circ,
#                name="Four_sum_relu_in_mult_out_subt_ratio_0.25_{}".format(circ),
#                cell_fn="pvsst",
#                store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#                connection="normal_ff_without_fb_final_sum")
#devices = [0,2,4]#[0,1]
#gates = ["in*_out-","in-_out-","in-_out*"]
#gate_names = ["in_mult_out_subt","in_subt_out_subt","in_subt_out_mult"]
#for device,gate,gate_name in zip(devices,gates,gate_names):
#  run_test_bash(device,
#                gating=gate,
#                pvsst_circuit="simple_subt",
#                name="Four_normal_relu_{}_ratio_0.25_simple_subt".format(gate_name),
#                cell_fn="pvsst",
#                store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#                connection="normal_ff_without_fb")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_0.25_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="flip_sign_bias_back",
#              name="EI_ratio_0.25_flip_sign_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(5,
#              act_fn="gate_relu_cell_relu",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_0.25_without_abs_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(4,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_0.25_cell_tanh_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=1,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_1_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=1,
#              pvsst_circuit="flip_sign_bias_back",
#              name="EI_ratio_1_flip_sign_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu",
#              gating="",
#              ratio=1,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_1_without_abs_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(5,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="",
#              ratio=1,
#              pvsst_circuit="bias_back",
#              name="EI_ratio_1_cell_tanh_bias_back",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="Four_normal_relu_in_mult_out_subt_ratio_0.25",
#              cell_fn="pvsst", #
#              store_dir="/home/ms5724",epoch=None,run_mode=5,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs",
#              cell_fn="pvsst", #
#              store_dir="/home/ms5724",epoch=None,run_mode=5,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(0,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="Four_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh",
#              cell_fn="pvsst", #
#              store_dir="/home/ms5724",epoch=None,run_mode=5,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="flip_sign",
#              name="Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign",
#              cell_fn="pvsst", #
#              store_dir="/home/ms5724",epoch=None,run_mode=5,sleep_time=0,
#              connection="normal_ff_without_fb")
#run_test_bash(0,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25")
#run_test_bash(1,"gate_relu_cell_relu","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25_without_abs")
#run_test_bash(2,"gate_relu_cell_tanh_kernel_abs","in*_out-",0.25,"","Four_normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh")
#run_test_bash(3,"gate_relu_cell_relu_kernel_abs","in*_out-",0.25,"flip_sign","Four_normal_relu_in_mult_out_subt_ratio_0.25_flip_sign")
#run_test_bash(2,
#              run_mode=2,
#              sleep_time=0,
#              name="normal_relu_in_mult_out_subt_ratio_0.25")
  
  
  
  
  
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="remove_OG",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_remove_OG_input_batch",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="input_batch")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in-_out-",
#              ratio=0.25,
#              pvsst_circuit="remove_OG_simple_subt",
#              name="normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="input_batch")
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in-_out-",
#              ratio=0.25,
#              pvsst_circuit="remove_OG_simple_subt_simple_fg_mem_SST",
#              name="normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="input_batch")
#run_test_bash(5,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_check",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(6,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_cell_tanh_check",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in-_out-",
#              ratio=0.25,
#              pvsst_circuit="remove_OG_simple_subt_simple_fg_mem_SST",
#              name="normal_relu_in_simple_subt_out_subt_ratio_0.25_remove_OG_input_batch_simple_fg_mem_SST_SST_batch",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="input_batch_SST_batch")

#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_remove_E_norm",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="remove_E")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_remove_I_norm",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="remove_I")
  
#run_test_bash(3,
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="",
#                ratio=0.25,
#                pvsst_circuit="",
#                name="EI_withbias_ratio_0.25_remove_E_and_I_norm",
#                cell_fn='EI', #"pvsst"
#                store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="remove_I_remove_E")
#run_test_bash(5,
#                act_fn="gate_relu_cell_relu_kernel_abs",
#                gating="",
#                ratio=0.25,
#                pvsst_circuit="",
#                name="EI_withbias_ratio_0.25_check",
#                cell_fn='EI', #"pvsst"
#                store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#                connection="normal_ff_without_fb",
#                normalize="")
#run_test_bash(6,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_SST_batch",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch_SST_batch")
#run_test_bash(7,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_PV_batch",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch_PV_batch")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_layer_norm",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="layer")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_norm",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_layer")
#run_test_bash(6,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_SST_layer_norm",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_layer_SST_layer")
#run_test_bash(7,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_layer_PV_layer_norm",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_layer_PV_layer")
# 5.20
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="flip_sign",
#              name="EI_withbias_ratio_0.25_remove_I_norm_flip_sign",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=18000,
#              connection="normal_ff_without_fb",
#              normalize="remove_I")
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_remove_I_norm_without_abs",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="remove_I")
#run_test_bash(1,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_remove_I_norm_cell_tanh",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="remove_I")
# 5.20
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="flip_sign_withff",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_withff",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_dropout",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="dropout")

#5.20
#run_test_bash(3,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_batch_remove_I",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="batch_remove_I")
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="flip_sign",
#              name="EI_withbias_ratio_0.25_batch_remove_I_flip_sign",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="batch_remove_I")
#run_test_bash(6,
#              act_fn="gate_relu_cell_relu",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_batch_remove_I_without_abs",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="batch_remove_I")
#run_test_bash(7,
#              act_fn="gate_relu_cell_tanh_kernel_abs",
#              gating="",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="EI_withbias_ratio_0.25_batch_remove_I_cell_tanh",
#              cell_fn='EI', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="batch_remove_I")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_dropout_0.9",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_dropout_0.9")
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_without_norm",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="")
#run_test_bash(2,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_dropout_PV_dropout_0.9",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_dropout_PV_dropout_0.9")
# 5.21
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="flip_sign_FF_inh_2.0",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_FF_inh_2.0",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="flip_sign_FF_inh_5.0",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_FF_inh_5.0",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#[0,1,2,5][3,4,6,7]
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="flip_sign_FF_no_abs",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_flip_sign_FF_no_abs",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=6,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(4,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="FF_inh",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_FF_inh",
#              cell_fn="pvsst",
#              store_dir="/home/jl5377",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(1,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="FF_no_abs",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_FF_no_abs",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=6,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="FF_abs",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_without_abs_FF_abs",
#              cell_fn="pvsst",
#              store_dir="/home/jl5377",epoch=None,run_mode=6,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch")
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="FF_abs_FF_inh",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_without_abs_FF_abs_FF_inh",
#              cell_fn="pvsst",
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_batch",
#              n_init=1)
#run_test_bash(0,
#              act_fn="gate_relu_cell_relu_kernel_abs",
#              gating="in*_out-",
#              ratio=0.25,
#              pvsst_circuit="",
#              name="normal_relu_in_mult_out_subt_ratio_0.25_inside_dropout_SST_dropout_0.9",
#              cell_fn='pvsst', #"pvsst"
#              store_dir="/home/ms5724",epoch=None,run_mode=0,sleep_time=0,
#              connection="normal_ff_without_fb",
#              normalize="inside_dropout_SST_dropout_0.9")
#run_test_bash(device=0,run_mode=2,sleep_time=0,name="imagenet_standard_model")
#run_test_bash(device=1,run_mode=2,sleep_time=0,name="imagenet_standard_model_init_1",n_init=1)
#run_test_bash(device=2,run_mode=2,sleep_time=0,name="imagenet_standard_model_init_2",n_init=2)
#run_test_bash(device=3,run_mode=2,sleep_time=0,name="imagenet_standard_model_init_3",n_init=3)
#run_test_bash(device=4,run_mode=2,sleep_time=0,name="imagenet_standard_model_init_4",n_init=4)
#run_test_bash(device=0,run_mode=2,sleep_time=0,name="normal_relu_in_mult_out_subt_ratio_0.25_without_norm_init_0",normalize='')
#run_test_bash(device=1,run_mode=2,sleep_time=0,name="normal_relu_in_mult_out_subt_ratio_0.25_add_SST_batch_init_0",normalize="inside_batch_SST_batch")
#run_test_bash(device=2,run_mode=2,sleep_time=0,name="EI_ratio_0.25_both_batch_init_0",cell_fn='EI',normalize='batch')
#run_test_bash(device=3,run_mode=2,sleep_time=0,name="EI_ratio_0.25_without_norm_init_0",cell_fn='EI',normalize='')
#run_test_bash(device=4,run_mode=2,sleep_time=0,name="EI_ratio_0.25_batch_remove_I_init_0",cell_fn='EI',normalize='batch_remove_I')
