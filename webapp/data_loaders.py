import streamlit as st
import numpy as np
import logging
from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5, data_freiburg_numpy_to_hdf5
import h5py
import os


@st.cache
def load_data_base():
    project_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/processed_data'


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

    logging.info('============================================================')
    logging.info('Loading validation data from: ' + project_data_root)
    data_vl = data_freiburg_numpy_to_hdf5.load_data(basepath = project_data_root,
                                                    idx_start = 20,
                                                    idx_end = 24,
                                                    train_test='validation')
    images_vl = data_vl['images_validation']
    labels_vl = data_vl['labels_validation']
    logging.info(type(images_tr))
    logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('Shape of training labels: %s' %str(labels_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]

    logging.info('============================================================')
    logging.info('Loading sliced training data from: ' + project_data_root)
    data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                    idx_start = 0,
                                                    idx_end = 19,
                                                    train_test='train')
    images_tr_sl = data_tr['sliced_images_train']
    logging.info(type(images_tr_sl))
    logging.info('Shape of training images: %s' %str(images_tr_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


    logging.info('============================================================')
    logging.info('Loading sliced validation data from: ' + project_data_root)
    data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                    idx_start = 20,
                                                    idx_end = 26,
                                                    train_test='validation')
    images_vl_sl = data_vl['sliced_images_validation']
    logging.info('Shape of validation images: %s' %str(images_vl_sl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
    logging.info('============================================================\n')

    return np.array(images_tr), np.array(labels_tr), np.array(images_vl), np.array(labels_vl), np.array(images_tr_sl), np.array(images_vl_sl)

@st.cache(allow_output_mutation=True)
def load_data_reconstructions():
    project_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/processed_data'

    # Load the reconstructions to visualize in the webapp
    # Save them in a dictionary?
    reconstructions = {}

    # filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_' + str(config['train_data_start_idx']) + 'to' + str(config['train_data_end_idx']) + '.hdf5')
    model_names = ['Sliced_VAE_2000EP_Normalized', 'Masked_Sliced_VAE_2000EP_no_aug', 'Sliced_ConditonalVAE_Test', 'Sliced_ConditonalReducedVAE_2000EP', 'Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug']

    for model_name in model_names:

        dataset_filepath_train = os.path.join(project_data_root, 'model_reconstructions/' + model_name + '_' + 'train' + '_' + str(0) + 'to' + str(19) + '.hdf5')
        dataset_filepath_validation = os.path.join(project_data_root, 'model_reconstructions/' + model_name + '_' + 'validation' + '_' + str(20) + 'to' + str(26) + '.hdf5' )

        if os.path.exists(dataset_filepath_train):
            reconstruction = h5py.File(dataset_filepath_train, 'r')
            data = reconstruction['reconstruction']

            dict_string = model_name + '_' + 'train'
            reconstructions[dict_string] = np.array(data)

        else:
            print("Reconstruction training data for model {} not found".format(model_name))


        if os.path.exists(dataset_filepath_validation):
            reconstruction = h5py.File(dataset_filepath_validation, 'r')
            data = reconstruction['reconstruction']

            dict_string = model_name + '_' + 'validation'
            reconstructions[dict_string] = np.array(data)

        else:
            print("Reconstruction testing data for model {} not found".format(model_name))

    return reconstructions

@st.cache(allow_output_mutation=True)
def load_data_reconstructions_noise():
    project_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/processed_data'

    # Load the reconstructions to visualize in the webapp
    # Save them in a dictionary?
    reconstructions = {}

    # filepath_output = os.path.join(config["project_data_root"], 'model_reconstructions/' + model_name + '_' + which_dataset + '_' + str(config['train_data_start_idx']) + 'to' + str(config['train_data_end_idx']) + '.hdf5')
    # model_names = ['Sliced_VAE_2000EP_Normalized', 'Sliced_ConditonalVAE_Test', 'Sliced_ConditonalReducedVAE_2000EP']
    model_names = ['Masked_Sliced_VAE_2000EP_no_aug','Sliced_ConditonalReducedVAE_2000EP', 'Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug']

    for model_name in model_names:

        dataset_filepath_validation = os.path.join(project_data_root, 'model_reconstructions/' + model_name + '_' + 'validation' + '_noisy_' + str(20) + 'to' + str(26) + '.hdf5')

        if os.path.exists(dataset_filepath_validation):
            reconstruction = h5py.File(dataset_filepath_validation, 'r')
            reconstruction_data = reconstruction['noisy_reconstruction']
            noisy_input = reconstruction['noisy']

            dict_string = model_name + '_' + 'validation'
            reconstructions[dict_string] = np.array(reconstruction_data)

        else:
            print("Reconstruction testing data for model {} not found".format(model_name))

    return np.array(noisy_input), reconstructions
