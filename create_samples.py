# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
# tf.enable_eager_execution()
import yaml
import argparse
import logging

import numpy as np

import datetime

# =================================================================================
# ============== IMPORT HELPER FUNCTIONS ==========================================
# =================================================================================
from helpers.batches import iterate_minibatches
from dataset_processing import data_freiburg_numpy_to_hdf5
from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5


# =================================================================================
# ============== IMPORT NETWORK STRUCTURES ========================================
# =================================================================================
from networks.variational_autoencoder import VariationalAutoencoder


# =================================================================================
# ============== IMPORT MODELS ====================================================
# =================================================================================
from models.vae import VAEModel




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



    # ========= MODEL CONFIGURATION ===========
    model_name = parameters.model_name
    train_set_name = parameters.train
    val_set_name = parameters.val

    preprocess_enabled = parameters.preprocess


    # ========= TRAINING PARAMETER CONFIGURATION ===========
    epochs = config['lr_decay_end']
    batch_size = config["batch_size"]
    box_factor = config["box_factor"]
    data_index = config["data_index"]

    z_dim =config["z_dim"]
    LR = config["lr"]
    image_original_size = 200


    # =================================================================================
    # ============== LOGGING CONFIGURATION ============================================
    # =================================================================================

    project_data_root = config["project_data_root"]
    project_code_root = config["project_code_root"]

    log_dir = os.path.join(project_code_root, "logs/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        os.mkdir(log_dir)
        logging.info('============================================================')
        logging.info(' Logging Directory: %s' %log_dir)
        logging.info('============================================================\n')

    except OSError:
        pass

    validation_frequency = config["validation_frequency"]


    # =================================================================================
    # ============== LOAD DATA ========================================================
    # =================================================================================


    logging.info('============================================================')
    logging.info('Loading training data from: ' + project_data_root)
    data_tr = data_freiburg_numpy_to_hdf5.load_data(basepath = project_data_root,
                                                    idx_start = 0,
                                                    idx_end = 19,
                                                    train_test='train')
    images_tr = data_tr['images_train']
    labels_tr = data_tr['labels_train']
    logging.info(type(images_tr))
    logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]


    # =================================================================================
    # ==== If mask preprocessing is enabled, do preprocessing of masked the data now
    # =================================================================================

    if preprocess_enabled == "mask":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data(basepath = project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 19,
                                                        train_test='train')
        images_tr = data_tr['masked_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data(basepath = project_data_root,
                                                        idx_start = 20,
                                                        idx_end = 24,
                                                        train_test='validation')
        images_vl = data_vl['masked_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('============================================================\n')


    # =================================================================================
    # ==== If slicing preprocessing is enabled, do preprocessing of masked the data now
    # =================================================================================
    
    logging.info('============================================================')
    logging.info('Loading training data from: ' + project_data_root)
    data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                    idx_start = 0,
                                                    idx_end = 10,
                                                    train_test='train')
    images_tr_sl = data_tr['sliced_images_train']
    logging.info(type(images_tr_sl))
    logging.info('Shape of training images: %s' %str(images_tr_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


    logging.info('============================================================')
    logging.info('Loading validation data from: ' + project_data_root)
    data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                    idx_start = 10,
                                                    idx_end = 13,
                                                    train_test='validation')
    images_vl_sl = data_vl['sliced_images_validation']
    logging.info('Shape of validation images: %s' %str(images_vl_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('============================================================\n')


    # =================================================================================
    # ==== If straighten preprocessing is enabled, do preprocessing of masked the data now
    # =================================================================================
    if preprocess_enabled == "straighten":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 10,
                                                        train_test='train')
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                        idx_start = 10,
                                                        idx_end = 13,
                                                        train_test='validation')
        images_vl = data_vl['sliced_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('============================================================\n')

        # Reshape the sliced images to have only side views of them
        # Only select part of the image to allow for greater uniformity in the images
        num_images = int(images_tr.shape[0]/64) #64 is the dimension along z
        num_of_y_slices = 10 #we want to select this number of slices from the middle of the image, has to be divisible by 2
        assert(num_of_y_slices % 2 == 0)

        new_shape = [num_images*num_of_y_slices,images_tr.shape[2],64,48,4]
        temp_reshape = np.zeros(new_shape)

        for i in range(num_images):

            one_subject = images_tr[i*64:(i+1)*64,:,:,:,:]

            one_subject = np.moveaxis(one_subject, 2,0) #move y axis to front
            one_subject = np.moveaxis(one_subject, 2,1) #move x axis before z axis

            start = 0 + (one_subject.shape[0]-num_of_y_slices)/2
            end = one_subject.shape[0] - (one_subject.shape[0]-num_of_y_slices)/2

            temp_reshape[i*num_of_y_slices:(i+1)*num_of_y_slices,:,:,:,:] = one_subject[start:end,:,:,:,:]

        images_tr = temp_reshape

        # Reshape the sliced images to have only side views of them
        # Only select part of the image to allow for greater uniformity in the images
        num_images = int(images_vl.shape[0]/64) #64 is the dimension along z
        num_of_y_slices = 10 #we want to select this number of slices from the middle of the image, has to be divisible by 2
        assert(num_of_y_slices % 2 == 0)

        new_shape = [num_images*num_of_y_slices,images_vl.shape[2],64,48,4]
        temp_reshape = np.zeros(new_shape)

        for i in range(num_images):

            one_subject = images_vl[i*64:(i+1)*64,:,:,:,:]

            one_subject = np.moveaxis(one_subject, 2,0) #move y axis to front
            one_subject = np.moveaxis(one_subject, 2,1) #move x axis before z axis

            start = 0 + (one_subject.shape[0]-num_of_y_slices)/2
            end = one_subject.shape[0] - (one_subject.shape[0]-num_of_y_slices)/2

            temp_reshape[i*num_of_y_slices:(i+1)*num_of_y_slices,:,:,:,:] = one_subject[int(start):int(end),:,:,:,:]

        images_vl = temp_reshape



    # ====================================================================================
    # Initialize the network architecture, training parameters, model_name, and logging directory
    # ====================================================================================


    # test new visualization function
    from helpers.batches import plot_vector_field_3d, plot_batch_3d, plot_batch_3d_complete
    from helpers.visualization import create_gif_of_velocities


    #Select subject 4
    i = 3

    #subject_normal = images_vl[i*32:(i+1)*32,:,:,:,:]
    subject_sliced = images_vl_sl[i*64:(i+1)*64,:,:,:,:]


    #plot the subject normally
    plot_batch_3d(subject_sliced, channel=0, every_x_time_step=1, out_path= "VECTOR_TEST_PLOTS/Subjects/Validation_Subject_Channel0_" + str(i) + ".png")
    plot_batch_3d(subject_sliced, channel=1, every_x_time_step=1, out_path= "VECTOR_TEST_PLOTS/Subjects/Validation_Subject_Channel1_" + str(i) + ".png")
    plot_batch_3d(subject_sliced, channel=2, every_x_time_step=1, out_path= "VECTOR_TEST_PLOTS/Subjects/Validation_Subject_Channel2_" + str(i) + ".png")
    plot_batch_3d(subject_sliced, channel=3, every_x_time_step=1, out_path= "VECTOR_TEST_PLOTS/Subjects/Validation_Subject_Channel3_" + str(i) + ".png")

    subject_sliced = np.moveaxis(subject_sliced,0,2)
    create_gif_of_velocities(input_data=subject_sliced, z_slice=10, output_folder="VECTOR_TEST_PLOTS/Subjects", output_file_name="test.gif")

    """
    plot_batch_3d(subject_normal, channel=1, every_x_time_step=1, out_path= "testing_slice_normal" + str(i) + ".png")
    plot_batch_3d(subject_sliced, channel=1, every_x_time_step=1, out_path= "testing_slice_sliced" + str(i) + ".png")

    plot_vector_field_3d(subject_sliced[55], timestep=8)
    """

    #Load the network
    """vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)

    path = os.path.join(project_code_root, "logs/Sliced_VAE_1500EP_20191126-142746/Sliced_VAE_1500EP")
    model.load_from_path(path, "Sliced_VAE_1500EP", 1490)

    # Add some random noise to the data in a small area

    part_noise = np.random.normal(0.8,0.1,(8,
                                  4,
                                  4,
                                  48))

    full_noise = np.ones([8,32,32,48])

    full_noise[:,4:8,16:20,:] = part_noise


    subject_sliced[50:58,:,:,:,1] *= full_noise
    #plot_vector_field_3d(subject_sliced[50], timestep=7)



    # Pass something through the network
    # Run a decoder output from the current input images
    feed_dict = {model.image_matrix: subject_sliced[56:64]}
    out_mu = model.sess.run(model.decoder_output, feed_dict)

    plot_batch_3d_complete(subject_sliced[56:64], out_mu, every_x_time_step=1, out_path= "VECTOR_TEST_PLOTS/TestFulloutput" + str(i) + ".png")


    plot_vector_field_3d(subject_sliced[60], timestep=7)
    plot_vector_field_3d(out_mu[3], timestep=7)
    """
