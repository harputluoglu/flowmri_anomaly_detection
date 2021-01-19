# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import os
import tensorflow as tf
import yaml
import argparse
import logging
import numpy as np
import datetime
import h5py

from matplotlib import pyplot as plt

# =================================================================================
# ============== IMPORT HELPER FUNCTIONS ==========================================
# =================================================================================
from helpers.batches import iterate_minibatches
from dataset_processing import data_freiburg_numpy_to_hdf5
from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5
from helpers.metrics import dice_score

# =================================================================================
# ============== IMPORT NETWORK STRUCTURES ========================================
# =================================================================================
from networks.variational_autoencoder import VariationalAutoencoder
from networks.conditional_variational_autoencoder import ConditionalVariationalAutoencoder
from networks.conditional_variational_autoencoder_reduced import ConditionalVariationalAutoencoderReduced


# =================================================================================
# ============== IMPORT MODELS ====================================================
# =================================================================================
from models.vae import VAEModel
from models.conditional_vae import ConditionalVAEModel
from models.conditional_vae_reduced import ConditionalVAEReducedModel



# =================================================================================
# ============== MAIN FUNCTION ====================================================
# =================================================================================
if __name__ == "__main__":

    # Parse the parameters given
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default="logs", help="path to checkpoint to restore")
    parser.add_argument("--train", type=str)
    parser.add_argument("--val", type=str)
    parser.add_argument("--preprocess", type=str)

    parameters = parser.parse_args()

    with open(parameters.config) as f:
        config = yaml.load(f)


    # ========= LOAD MODEL CONFIGURATION ===========
    model_name = parameters.model_name
    train_set_name = parameters.train
    val_set_name = parameters.val
    preprocess_enabled = parameters.preprocess

    # ========= TRAINING PARAMETER CONFIGURATION ===========
    epochs = config['lr_decay_end']
    batch_size = config["batch_size"]
    LR = config["lr"]

    # =================================================================================
    # ============== FOLDER AND LOGDIR CONFIGURATION ==================================
    # =================================================================================

    project_data_root = config["project_data_root"]
    project_code_root = config["project_code_root"]

    log_dir = os.path.join(project_code_root, "logs/" + model_name + 'EVAL' + '_' + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        os.mkdir(log_dir)
        logging.info('============================================================')
        logging.info(' Logging Directory: %s' %log_dir)
        logging.info('============================================================\n')

    except OSError:
        pass


    # ==================================================================================================================================================================
    # ============== LOAD THE MODELS ===================================================================================================================================
    # ==================================================================================================================================================================

    cvae_network = ConditionalVariationalAutoencoderReduced
    model = ConditionalVAEReducedModel(cvae_network, config, model_name, log_dir)

    # Load the vae model as our baseline
    path = os.path.join(project_code_root, config["model_directory"])
    model.load_from_path(path, config["model_name"] , config["latest_model_epoch"])


    # ==================================================================================================================================================================
    # ============== LOAD THE DATA =====================================================================================================================================
    # ==================================================================================================================================================================

    if preprocess_enabled == "slice":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                        idx_start = config['train_data_start_idx'],
                                                        idx_end = config['train_data_end_idx'],
                                                        train_test='train')
        images_tr_sl = data_tr['sliced_images_train']
        logging.info(type(images_tr_sl))
        logging.info('Shape of training images: %s' %str(images_tr_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                        idx_start = config['validation_data_start_idx'],
                                                        idx_end = config['validation_data_end_idx'],
                                                        train_test='validation')
        images_vl_sl = data_vl['sliced_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('============================================================\n')


    # =================================================================================
    # ==== If masked slicing preprocessing is enabled, do preprocessing of masked the data now
    # =================================================================================

    if preprocess_enabled == "masked_slice":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                        idx_start = config['train_data_start_idx'],
                                                        idx_end = config['train_data_end_idx'],
                                                        train_test='train')
        images_tr_sl = data_tr['sliced_images_train']
        logging.info(type(images_tr_sl))
        logging.info('Shape of training images: %s' %str(images_tr_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                        idx_start = config['validation_data_start_idx'],
                                                        idx_end = config['validation_data_end_idx'],
                                                        train_test='validation')
        images_vl_sl = data_vl['sliced_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('============================================================\n')


    # ==================================================================================================================================================================
    # ============== START EVALUATION ==================================================================================================================================
    # ==================================================================================================================================================================

    from helpers.batches import plot_vector_field_3d, plot_batch_3d, plot_batch_3d_complete
    from helpers.visualization import create_gif_of_velocities
    from helpers.metrics import rmse, compute_tpr_fpr

    # Create the needed directories for evaluation if they do not exist already
    try:
        output_dir_train = 'Results/Evaluation/' + model_name + '/train'
        output_dir_validation = 'Results/Evaluation/' + model_name + '/validation'

        if not os.path.exists(os.path.join(project_code_root, output_dir_train)):
            os.makedirs(os.path.join(project_code_root, output_dir_train))

        if not os.path.exists(os.path.join(project_code_root, output_dir_validation)):
            os.makedirs(os.path.join(project_code_root, output_dir_validation))

        if not os.path.exists(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/generated')):
            os.makedirs(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/generated'))

        if not os.path.exists(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/noise')):
            os.makedirs(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/noise'))

        if not os.path.exists(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/AUC')):
            os.makedirs(os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/AUC'))

    except OSError:
        print("Could not create output directories for evaluation")
        pass


    # Start by writing a summary of the most important config elements
    logging.info('============================================================')
    logging.info('=============== EVALUATION SETTINGS ========================')
    logging.info('============================================================')

    logging.info('Model name: {}'.format(config['model_name']))
    logging.info('Model directory: {}'.format(config['model_directory']))
    logging.info('Number of epochs: {}'.format(str(config['latest_model_epoch'])))
    logging.info('')
    logging.info('Evaluation on {} subjects and {} slices'.format(config['subject_mode'],config['slice_mode']))

    if config['subject_mode'] != 'all':
        logging.info('Custom train subjects: {}'.format(str(config['subjects_train'])))
        logging.info('Custom validation subjects: {}'.format(str(config['subjects_validation'])))

    if config['slice_mode'] != 'all':
        logging.info('Custom slices selected: {}'.format(str(config['which_slices'])))

    logging.info('============================================================')
    logging.info('============================================================')


    # =================================================================================
    # ============== ADD RANDOM NOISE AND EVALUATE ANOMALY DETECTION ==================
    # =================================================================================

    noise_parameters = [(0.05, 0.05), (0.1, 0.1), (0.1, 0.3) , (0.3, 0.1)] # additive noise to be tested, each tuple has mean and stdev for the normal distribution
    which_dataset = 'validation'

    total_storage_for_plotting = [] # will hold the tpr/fpr for each level of noise

    # loop over the noise parameters
    for (mean, stdev) in noise_parameters:

        _images_vl_sl = np.copy(images_vl_sl)

        # Create some random noise to the data in a small area
        part_noise = np.random.normal(mean,
                                        stdev,
                                        (8, 5, 5, 48, 4))

        full_noise = np.zeros([8,32,32,48,4])
        full_noise[:,14:19,14:19,:, :] = part_noise

        # Binary mask for ROC and AUC curve computation
        mask = np.zeros([batch_size,32,32,48,4])
        mask[:,14:19,14:19,:,:] =  1.


        # =================================================================================
        # ============== COMPUTE THE DECODER OUTPUTS GIVEN THE LEVEL OF NOISE =============
        # =================================================================================

        # Initialize variables for looping
        start_idx = 0
        end_idx = config["batch_size"]

        # Create a "dataset" to store all the decoder outputs to avoid recomputations
        out_mu_total = np.zeros(_images_vl_sl.shape)

        # loop over all the slices in the dataset
        while end_idx <= _images_vl_sl.shape[0]:

            # Noise is added on each channel for each slice of the subject
            _images_vl_sl[start_idx:end_idx,:,:,:,:] += full_noise

            bins = np.linspace(0, 64, 9) # 64 slices, 8 classes + 1 since outer limit excluded
            batch_slice_info = [x % 64 for x in range(start_idx,end_idx)]
            batch_slice_info = np.digitize(batch_slice_info, bins)-1

            feed_dict = {model.image_matrix: _images_vl_sl[start_idx:end_idx], model.batch_slice_info: batch_slice_info}
            out_mu = model.sess.run(model.decoder_output, feed_dict)

            # save it to the total array
            out_mu_total[start_idx:end_idx,:,:,:,:] = out_mu

            # update loop variables
            start_idx += config["batch_size"]
            end_idx += config["batch_size"]

        # =================================================================================
        # ============== COMPUTE THE AUC PARAMETERS BASED ON THE DATASET THAT WAS COMPUTED
        # =================================================================================

        # Setup storage arrays for tpr/fpr
        one_threshold_tpr_storage = []
        one_threshold_fpr_storage = []

        tpr_storage = []
        fpr_storage = []

        for threshold in config['dice_score_thresholds']:

            # Initialize variables for looping
            start_idx = 0
            end_idx = config["batch_size"]

            # loop over all the slices in the dataset
            while end_idx <= _images_vl_sl.shape[0]:

                # compute the tpr / fpr
                difference = np.abs(_images_vl_sl[start_idx:end_idx]- out_mu_total[start_idx:end_idx])

                tpr, fpr = compute_tpr_fpr(mask,difference,threshold)

                one_threshold_tpr_storage.append(tpr)
                one_threshold_fpr_storage.append(fpr)

                # update loop variables
                start_idx += config["batch_size"]
                end_idx += config["batch_size"]

            # Average full dataset tpr/fprs for one threshold
            tpr_avg = np.mean(one_threshold_tpr_storage)
            fpr_avg = np.mean(one_threshold_fpr_storage)

            # Append it to the total
            tpr_storage.append(tpr_avg)
            fpr_storage.append(fpr_avg)

        # The storage arrays now hold averaged tpr/fpr for each threshold on the dataset with the given noise.
        # compute AUC
        auc = 1. + np.trapz(np.array(fpr_storage), np.array(tpr_storage))

        logging.info('Model: {} || Noise Mean {}, Stdev {} --- AUC VALUE: {}'.format(model_name, str(mean), str(stdev), str(auc)))

        # Save the fpr/tpr information to text file so we can grab it later to create plots
        fpr_file_path = os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/AUC/' + 'FPR_NoiseMean_' + str(mean) +'_Std_' + str(stdev) +'.txt')
        tpr_file_path = os.path.join(project_code_root, 'Results/Evaluation/' + model_name + '/AUC/' + 'TPR_NoiseMean_' + str(mean) +'_Std_' + str(stdev) +'.txt')

        np.savetxt(fpr_file_path, np.array(fpr_storage))
        np.savetxt(tpr_file_path, np.array(tpr_storage))

        # save to total storage
        total_storage_for_plotting.append((np.array(tpr_storage), np.array(fpr_storage)))



    # =================================================================================
    # ============== PLOTTING =========================================================
    # =================================================================================

    # We have now looped over all the different noise/threshold combinations.
    # Create a summary plot
    plt.figure()
    plt.title("ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    i = 0
    colors = ['blue', 'purple', 'red', 'orange', 'magenta']
    for (tpr, fpr) in total_storage_for_plotting:

        tpr = tpr / tpr.max()
        fpr = fpr / fpr.max()

        noise_mean, noise_std = noise_parameters[i]
        plt.plot(fpr, tpr, color=colors[i], label="Noise: mean=" + str(noise_mean) + ", stdev=" + str(noise_std))
        i += 1

    plt.legend()
    out_path = os.path.join(project_code_root,'Results/Evaluation/' + model_name + '/AUC/' + 'ROC_CURVES.png')
    plt.savefig(out_path)
