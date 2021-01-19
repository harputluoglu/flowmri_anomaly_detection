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
from helpers.batches import iterate_minibatches, iterate_minibatches_with_sliceinfo
from dataset_processing import data_freiburg_numpy_to_hdf5
from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5


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
    parser.add_argument("--continue_training", default=False, help="set to true and give path to model that should be continued in training")
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
    continue_training = parameters.continue_training


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

    if preprocess_enabled == "none":

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
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
        logging.info('Shape of validation labels: %s' %str(labels_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t]
        logging.info('============================================================\n')


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

    if preprocess_enabled == "slice":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_cropped_data_sliced(basepath = project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 10,
                                                        train_test='train')
        images_tr = data_tr['sliced_images_train']
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


    # =================================================================================
    # ==== If masked slicing preprocessing is enabled, do preprocessing of masked the data now
    # =================================================================================

    if preprocess_enabled == "masked_slice":

        logging.info('============================================================')
        logging.info('Loading training data from: ' + project_data_root)
        data_tr = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                        idx_start = 0,
                                                        idx_end = 19,
                                                        train_test='train')
        images_tr = data_tr['sliced_images_train']
        logging.info(type(images_tr))
        logging.info('Shape of training images: %s' %str(images_tr.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]


        logging.info('============================================================')
        logging.info('Loading validation data from: ' + project_data_root)
        data_vl = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(basepath = project_data_root,
                                                        idx_start = 20,
                                                        idx_end = 26,
                                                        train_test='validation')
        images_vl = data_vl['sliced_images_validation']
        logging.info('Shape of validation images: %s' %str(images_vl.shape)) # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]
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

    #Initialize the VAE
    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)
    model.initialize()
    model.summarize()

    # Should we continue training an existing model or is it a new model
    if continue_training:
        path = os.path.join(project_code_root, config["model_directory"])
        model.load_from_path(path, config["model_name"] , config["latest_model_epoch"])

        already_completed_epochs = config["latest_model_epoch"]

    else:
        model.summarize()
        already_completed_epochs = 0


    logging.info('============================================================')
    logging.info('Starting iterating through minibatches ....')
    for ep in range(already_completed_epochs,epochs):

        weight = 1.

        logging.info("Processing epoch:" + str(ep))

        for batch in iterate_minibatches(images_tr, batch_size = batch_size, data_augmentation=config["do_data_augmentation"]):

            input_images = batch
            model.train(input_images, weight)

        if ep % 10 == 0:

            input_images = input_images.astype("float32")


            #model.validate(validate_images,weight)
            #model.visualize(model_name, ep)
            #samples = model.sample()
            gen_loss, res_loss, lat_loss = model.sess.run([model.autoencoder_loss,
                                                           model.autoencoder_res_loss,
                                                           model.latent_loss], {model.image_matrix: input_images})


            # Write summary for tensorboard
            summary_str = model.sess.run(model.summary_op, {model.image_matrix: input_images,model.weight: weight})
            model.writer.add_summary(summary_str, ep)
            model.writer.flush()


            logging.info(("Epoch %d: train_gen_loss %f train_lat_loss %f train_res_loss %f total train_loss %f") % (
                ep, gen_loss.mean(), lat_loss.mean(), res_loss.mean(), 100.*gen_loss.mean()+lat_loss.mean()))

            model.save(model_name, ep)


        # =================================================================================
        # =========== FOR EPOCHS MATCHING VALIDATION FREQUENCY, VALDIATE MODEL ============
        # =================================================================================

        if ep % validation_frequency == 0:

            batchcount = 0
            for batch in iterate_minibatches(images_vl, batch_size = batch_size):
                input_images = batch

                gen_loss_valid, res_loss_valid, lat_loss_valid = model.sess.run([model.autoencoder_loss_test,
                                                               model.autoencoder_res_loss_test,
                                                               model.latent_loss_test], {model.image_matrix: input_images})

                print(("epoch %d: test_gen_loss %f test_lat_loss %f res_loss %f total loss %f") % (
                    ep, gen_loss_valid.mean(), lat_loss_valid.mean(), res_loss.mean(),
                    100.*gen_loss_valid.mean()+lat_loss_valid.mean()))

                batchcount +=1

            model.validate(input_images, weight)
            model.visualize(model_name, ep, project_code_root)
