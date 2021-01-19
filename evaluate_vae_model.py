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


# =================================================================================
# ============== IMPORT MODELS ====================================================
# =================================================================================
from models.vae import VAEModel
from models.conditional_vae import ConditionalVAEModel


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

    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)

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
    
    if preprocess_enabled == "masked_slice_anomalous":

        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 1,
                                                        train_test='validation',
                                                        load_anomalous=True)
                                                        
        images_vl_sl = data_vl['sliced_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('============================================================\n')


    # ==================================================================================================================================================================
    # ============== START EVALUATION ==================================================================================================================================
    # ==================================================================================================================================================================

    from helpers.batches import plot_vector_field_3d, plot_batch_3d, plot_batch_3d_complete
    from helpers.visualization import create_gif_of_velocities
    from helpers.metrics import rmse

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
    # ============== RMSE EVALUATION AND VISUAL INSPECTION ============================
    # =================================================================================

    # Loop over the datasets that should be evaluated
    for which_dataset in config['which_datasets']:

        logging.info('============ DATASET {} ============='.format(which_dataset))

        # initialize storage array for all the subject RMSEs in the dataset
        mean_dataset_rmse = []

        # Select the subjects, either all or custom selection of subjects
        if config['subject_mode'] == 'all':
            if which_dataset == 'train':
                subject_indexes = range(config['train_data_end_idx']+1)

            elif which_dataset == 'validation':
                subject_indexes = range(config['validation_data_end_idx'] - config['validation_data_start_idx'] + 1)

        else:
            if which_dataset == 'train':
                subject_indexes = config['subjects_train']

            elif which_dataset == 'validation':
                subject_indexes = config['subjects_validation']

        # If option to create a hdf5 file is enabled
        if config["save_hdf5"]:

            if which_dataset == 'train':
                filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_' + str(config['train_data_start_idx']) + 'to' + str(config['train_data_end_idx']) + '.hdf5')

            elif which_dataset == 'validation':
                if preprocess_enabled == 'masked_slice_anomalous':
                    filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_' + 'anomalous_' + str(config['validation_data_start_idx']) + 'to' + str(config['validation_data_end_idx']) + '.hdf5')
                else:    
                    filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_' + str(config['validation_data_start_idx']) + 'to' + str(config['validation_data_end_idx']) + '.hdf5')

            # create a hdf5 file
            dataset = {}
            hdf5_file = h5py.File(filepath_output, "w")
            dataset['reconstruction'] = hdf5_file.create_dataset("reconstruction", images_vl_sl.shape, dtype='float32')


        # Loop over the selected subject indexes
        for i in subject_indexes:

            # grab the full data for that one subject
            if which_dataset == 'train':
                subject_sliced = images_tr_sl[i*64:(i+1)*64,:,:,:,:]

            elif which_dataset == 'validation':
                subject_sliced = images_vl_sl[i*64:(i+1)*64,:,:,:,:]


            # Initialize the while loop variables
            start_idx = 0
            end_idx = config["batch_size"]
            subject_rmse = []

            while end_idx <= config["spatial_size_z"]:

                # Selected slices for batch and run through the model
                feed_dict = {model.image_matrix: subject_sliced[start_idx:end_idx]}
                out_mu = model.sess.run(model.decoder_output, feed_dict)

                # Compute rmse of these slices and append it to the subject error
                error = rmse(subject_sliced[start_idx:end_idx], out_mu)
                subject_rmse.append(error)

                # Visualization
                if config["visualization_mode"] == 'all':
                    out_path = os.path.join(project_code_root,'Results/Evaluation/' + model_name + '/' + which_dataset + '/' + 'Subject_' + str(i) + '_' + str(start_idx) + '_' + str(end_idx) + '.png')
                    plot_batch_3d_complete(subject_sliced[start_idx:end_idx], out_mu, every_x_time_step=1, out_path= out_path)

                # Save it to the hdf5 file
                dataset['reconstruction'][start_idx+(i*config["spatial_size_z"]):end_idx+(i*config["spatial_size_z"]), :, :, :, :] = out_mu

                # update vars for next loop
                start_idx += config["batch_size"]
                end_idx += config["batch_size"]

            # Now that we have the rmse of the different slices, compute total RMSE and stdev for this subject
            to_numpy = np.array(subject_rmse)
            mean_subject_rmse = np.mean(to_numpy)

            # Append it to the dataset rmse
            mean_dataset_rmse.append(mean_subject_rmse)

            # Compute stdev
            std_subject_rmse = np.std(to_numpy)

            logging.info('RMSE: Subject {} from {} ; {} ; {}'.format(i, which_dataset, mean_subject_rmse, std_subject_rmse))

        # end of for loop over subjects

        to_numpy = np.array(mean_dataset_rmse)
        total_mean_dataset_rmse = np.mean(to_numpy)
        total_std_dataset_rmse = np.std(to_numpy)

        logging.info('==== TOTAL MEAN RMSE ====: {}'.format(total_mean_dataset_rmse))
        logging.info('==== TOTAL STD RMSE  ====: {}'.format(total_std_dataset_rmse))

        hdf5_file.close()
        logging.info('=================================\n')

    # end of for loop over data_sets

    # =================================================================================
    # ============== GENERATE SAMPLES FROM THE NETWORK ================================
    # =================================================================================

    # Create some samples from the VAE.
    # So we need to sample the latent space and then give it a category, then we generate images and look at what they look like.
    # We could plot the same latent vector with the different categories and see how it changes.

    """
    generated_samples = model.sample()
    path = os.path.join(project_code_root,'Results/Evaluation/' + model_name + '/generated/')

    plot_batch_3d(generated_samples, channel=0, every_x_time_step=1, out_path= path + "Generated_Channel_0" + ".png")
    plot_batch_3d(generated_samples, channel=1, every_x_time_step=1, out_path= path + "Generated_Channel_1" + ".png")
    plot_batch_3d(generated_samples, channel=2, every_x_time_step=1, out_path= path + "Generated_Channel_2" + ".png")
    plot_batch_3d(generated_samples, channel=3, every_x_time_step=1, out_path= path + "Generated_Channel_3" + ".png")

    """


    # =================================================================================
    # ============== ADD RANDOM NOISE AND EVALUATE ANOMALY DETECTION ==================
    # =================================================================================

    # Noise evaluation and dice score
    # In the below section we add artificial noise to some subjects and check the reconstructed images
    # to see how the model performs at removing these artificial anomalies.

    """

    noise_parameters = [(0.05, 0.01), (0.1, 0.02), (0.1, 0.1) ,(0.2, 0.4), (0.3, 0.1)] # additive noise to be tested, each tuple has mean and stdev for the normal distribution
    which_dataset = 'validation'

    # create a hdf5 file
    dataset = {}
    filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_noisy_' + str(config['validation_data_start_idx']) + 'to' + str(config['validation_data_end_idx']) + '.hdf5')
    hdf5_file = h5py.File(filepath_output, "w")
    dataset['noisy'] = hdf5_file.create_dataset("noisy", images_vl_sl.shape, dtype='float32')
    dataset['noisy_reconstruction'] = hdf5_file.create_dataset("noisy_reconstruction", images_vl_sl.shape, dtype='float32')

    logging.info('======== NOISE COMPUTATION START ===========')
    # Use validation subjects to add noise
    for idx in config['noise_subjects_validation']:
        subject_sliced = images_vl_sl[idx*64:(idx+1)*64,:,:,:,:]

        logging.info('===== SUBJECT {} :'.format(str(idx)))

        for (mean, stdev) in noise_parameters:

            # Add some random noise to the data in a small area
            part_noise = np.random.normal(mean,
                                          stdev,
                                          (8, 5, 5, 48, 4))

            full_noise = np.zeros([8,32,32,48,4])
            full_noise[:,14:19,14:19,:, :] = part_noise

            # Binary mask for dice score computation
            mask = np.zeros([batch_size,32,32,48])
            mask[:,14:19,14:19,:] =  1.

            # Initialize variables for looping
            start_idx = 0
            end_idx = config["batch_size"]
            rmse_list = [] # temp variable to store the rmses for the subject before taking the average

            while end_idx <= config["spatial_size_z"]:

                # Noise is added on each channel for each slice of the subject
                subject_sliced[start_idx:end_idx,:,:,:,:] += full_noise

                # Save it to the hdf5 file if the selected noise level is hit
                if (mean == 0.1) and (stdev == 0.1):
                    dataset['noisy'][start_idx+(idx*config["spatial_size_z"]):end_idx+(idx*config["spatial_size_z"]), :, :, :, :] = subject_sliced[start_idx:end_idx]

                feed_dict = {model.image_matrix: subject_sliced[start_idx:end_idx]}
                out_mu = model.sess.run(model.decoder_output, feed_dict)

                # Save it to the hdf5 file if the selected noise level is hit
                if (mean == 0.1) and (stdev == 0.1):
                    dataset['noisy_reconstruction'][start_idx+(idx*config["spatial_size_z"]):end_idx+(idx*config["spatial_size_z"]), :, :, :, :] = out_mu

                # Compute dice score
                difference_image = subject_sliced[start_idx:end_idx] - out_mu

                # TODO: Compute the rmse between the difference image and the noise that we used as input
                # Think about: we could either save all the reconstructions separately for each model which would take a lot of space
                # Or we do not save them at all but just save the results as a plot and then grab it manually instead of with the UI

                noise_reconstruction_rmse = rmse(full_noise, difference_image)
                rmse_list.append(noise_reconstruction_rmse)

                # Visualize it
                if config["visualization_mode"] == 'all':
                    out_path = os.path.join(project_code_root,'Results/Evaluation/' + model_name + '/noise/' + 'Subject_' + which_dataset + '_' + str(idx) + '_noise' + '_' + str(mean) +'_' + str(stdev) + '_slice_' + str(start_idx) + '_to_' + str(end_idx) +'.png')
                    plot_batch_3d_complete(subject_sliced[start_idx:end_idx], out_mu, every_x_time_step=1, out_path= out_path)

                # update loop variables
                start_idx += config["batch_size"]
                end_idx += config["batch_size"]

            mean_rmse = np.mean(rmse_list)
            stdev_rmse = np.std(rmse_list)
            logging.info('Mean noise {}, Stdev noise {} --> RMSE (mean and stdev); {} ; {}'.format(str(mean), str(stdev), str(mean_rmse), str(stdev_rmse)))

        # end of for with different noises loop

    # end of for loop for subjects
    hdf5_file.close()"""
