#!/bin/bash

## SET THE FOLLOWING VARIABLES ACCORDING TO YOUR SYSTEM ##
CUDA_HOME=/scratch_net/biwidl210/peifferp/apps/cuda-9.0

# Otherwise the detail shell would be used
#$ -S /bin/bash

## Pass env. vars of the workstation to the GPU node
##$-V

## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q gpu.48h.q@*
##$ -q gpu.2h.q


## The maximum memory usage of this job (below 4G does not make much sense)
#$ -l gpu
#$ -l h_vmem=40G


## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory, preferably on your scratch
#$ -o /scratch_net/biwidl210/peifferp/thesis/master-thesis/logs
#
## send mail on job's end or abort
#$ -m a


# cuda paths
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# call your calculation exec.
source /scratch_net/biwidl210/peifferp/anaconda3/bin/activate
echo $SGE_GPU
echo $PATH
echo $LD_LIBRARY_PATH

# Execute this file with config
# python main.py --model EDSR_6 --scale 2 --patch_size 96 --save edsr_6_3_res_5_ind_load_weights_all_in_one_opt --reset --n_resblocks 1 --epochs 200 --load_weights False --custom_loss True &>log.log


# ===== MASKED SETUP HERE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess mask --model_name Masked_VAE_200EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/conf.yaml

# ===== SLICED SETUP HERE ========
python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess slice --model_name Sliced_VAE_1500EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/conf_sliced.yaml

# ===== SLICED STRAIGHETNED SETUP HERE ========
python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess straighten --model_name Sliced_Straightened_VAE_200EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_straightened.yaml

# ===== WINDOWS SETUP HERE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/visualize_velocities.py --model_name Implement_VAE --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/conf.yaml



# python main.py --preprocess slice --model_name CVAE_FIRST_TEST --config config/conf_windows.yaml   