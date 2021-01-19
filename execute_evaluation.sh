#!/bin/bash

## SET THE FOLLOWING VARIABLES ACCORDING TO YOUR SYSTEM ##
CUDA_HOME=/scratch_net/biwidl210/peifferp/apps/cuda-9.0

# Otherwise the detail shell would be used
#$ -S /bin/bash

## Pass env. vars of the workstation to the GPU node
##$-V

## <= 1h is short queue, <= 6h is middle queue, <= 48h is long queue
#$ -q gpu.24h.q@*
##$ -q gpu.2h.q
##$ -P projdevel -now n


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


# ===== VAE EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_vae_model.py --preprocess slice --model_name Sliced_VAE_2000EP_Normalized --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta.yaml

# ===== VAE EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_vae_model.py --preprocess slice --model_name Sliced_VAE_1500EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_total_aorta.yaml

# ===== CVAE EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_cvae_model.py --preprocess slice --model_name Sliced_ConditonalVAE_Test --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta.yaml

# ===== REDUCED CVAE EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_cvae_model_reduced.py --preprocess slice --model_name Sliced_ConditonalReducedVAE_2000EP --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced.yaml

# ===== REDUCED CVAE MASKED EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_cvae_model_reduced.py --preprocess masked_slice --model_name Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced_masked.yaml

# ===== VAE MASKED EVALUATION =================
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_vae_model.py --preprocess masked_slice --model_name Masked_Sliced_VAE_2000EP_no_aug --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta_masked.yaml

# ===== REDUCED CVAE MASKED AUGMENTED EVALUATION ========
python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_cvae_model_reduced.py --preprocess masked_slice --model_name Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced_masked_augmented.yaml

# ===== VAE MASKED AUGMENTED EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_vae_model.py --preprocess masked_slice --model_name Masked_Sliced_VAE_2000EP_augmented --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta_masked_augmented.yaml





# =================================================================================
# ============== AUC EVALUATIONS ==================================================
# =================================================================================

# ===== REDUCED CVAE MASKED AUGMENTED EVALUATION ========
python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_cvae_reduced.py \
--preprocess masked_slice \
--model_name Masked_Sliced_ConditonalReducedVAE_2000EP_augmented_enabled \
--config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced_masked_augmented.yaml

# ===== REDUCED CVAE MASKED NOT AUGMENTED EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_cvae_reduced.py \
# --preprocess masked_slice \
# --model_name Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced_masked.yaml
#
# # ===== REDUCED CVAE  ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_cvae_reduced.py \
# --preprocess slice \
# --model_name Sliced_ConditonalReducedVAE_2000EP \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_cvae_ascending_aorta_reduced.yaml
#
# # ===== VAE MASKED AUGMENTED EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_vae.py \
# --preprocess masked_slice \
# --model_name Masked_Sliced_VAE_2000EP_augmented \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta_masked_augmented.yaml

# # ===== VAE MASKED NOT AUGMENTED EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_vae.py \
# --preprocess masked_slice \
# --model_name Masked_Sliced_VAE_2000EP_no_aug \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta_masked.yaml
#
# # ===== Non masked VAE EVALUATION ========
# python /scratch_net/biwidl210/peifferp/thesis/master-thesis/compute_auc_vae.py \
# --preprocess slice \
# --model_name Sliced_VAE_2000EP_Normalized \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta.yaml



# =================================================================================
# ============== ANOMALOUS EVALUATION =============================================
# =================================================================================

#python /scratch_net/biwidl210/peifferp/thesis/master-thesis/evaluate_vae_model.py \
# --preprocess masked_slice_anomalous \
# --model_name Masked_Sliced_VAE_2000EP_no_aug \
# --config /scratch_net/biwidl210/peifferp/thesis/master-thesis/config/evaluation/conf_sliced_vae_ascending_aorta_masked_anomalous.yaml
