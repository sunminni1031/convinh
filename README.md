# convinh folder

To train the model:

1. python path:  export PYTHONPATH="$PYTHONPATH:/path/to/EI_models" 
2. Dependencies: pip3 install --user -r official/requirements.txt 
3. go to convinh/:  python3 cifar10_main.py (or imagenet_main128.py)



cifar10_main.py (or imagenet_main128.py) is the main file, it is a wrapped version of convinh_run_loop.py 

convinh_run_loop.py is the training protocol 

convinh_model.py builds the model 

custom_cell.py defines the recurrent cells 



example training command:

python3  cifar10_main.py  --data_dir=/home/cifar10_data  --batch_size=128  --model_dir=/home/convinh_model  --export_dir=/home/convinh_export  --cell_fn=pvsst

exported models are saved in export_dir,  and checkpoints are saved in model_dir