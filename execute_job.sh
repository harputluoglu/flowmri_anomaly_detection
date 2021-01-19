#!/bin/bash

## SET THE FOLLOWING VARIABLES ACCORDING TO YOUR SYSTEM ##
CUDA_HOME=/scratch_net/biwidl210/peifferp/apps/cuda-9.0

# Otherwise the detail shell would be used
#$ -S /bin/bash

## Pass env. vars of the workstation to the GPU node
##$-V

## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q gpu.48h.q@*
##$ -q gpu.24h.q@*


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


# ===== MASKED SETUP HERE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess mask --model_name Masked_VAE_200EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/conf.yaml

# ===== SLICED SETUP HERE ========
#python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess slice --model_name Sliced_ConditonalVAE_300EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced.yaml

# ===== SLICED STRAIGHETNED SETUP HERE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main.py --preprocess straighten --model_name Sliced_Straightened_Normalized_2000EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_straightened.yaml

# ===== WINDOWS SETUP HERE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/visualize_velocities.py --model_name Implement_VAE --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/conf.yaml

# ===== SLICED REDUCED CVAE ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main_cvae_reduced.py --preprocess slice --model_name Sliced_ConditonalReducedVAE_2000EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced_cvae_reduced.yaml


# ===============================================================
# Masking everything but the aorta
# ===============================================================

# reduced CVAE
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main_cvae_reduced.py --preprocess masked_slice --model_name Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced_cvae_reduced.yaml

# normal VAE
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main_vae.py --preprocess masked_slice --model_name Masked_Sliced_VAE_2000EP_no_aug --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced.yaml

# ===============================================================
# Masking everything but the aorta, with data_augmentation
# ===============================================================
# normal VAE
#python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main_vae.py \
#--preprocess masked_slice \
#--model_name Masked_Sliced_VAE_2000EP_augmented \
#--config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced.yaml \
#--continue_training True

# reduced CVAE
python /scratch_net/biwidl210/peifferp/thesis/master-thesis/main_cvae_reduced.py \
--preprocess masked_slice \
--model_name Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled \
--config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/conf_sliced_cvae_reduced.yaml \
--continue_training True
