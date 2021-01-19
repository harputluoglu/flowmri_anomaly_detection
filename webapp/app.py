# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================
import sys
sys.path.append("../")

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import logging
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize_3d
from scipy import interpolate
import math
import h5py
import os

from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5, data_freiburg_numpy_to_hdf5


# =================================================================================
# ============== Streamlit Components =============================================
# =================================================================================
from webapp.data_loaders import *
from webapp.views import *

# =================================================================================
# This function renders the side menue of the app
# and returns the variables that the user can select within that menu
# =================================================================================
def side_menu(data, dataset_sizes):

    # Title of the side menu
    st.sidebar.markdown("# Menu")

    # Create the radio buttons that let you choose the dataset
    train_or_validation = st.sidebar.radio("Select training, validation or test set",
     ('train', 'validation', 'test', 'anomalous'))

    # Initialize subject to 0 and create a dropdown menu to select a
    # specific subject from the dataset
    subject = 0
    subject = st.sidebar.selectbox("Select Subject",
            range(dataset_sizes[train_or_validation]))

    # Create the sliders to slide through the z_slices as well as time
    time_step = st.sidebar.slider('Select Timestep', 0, data.shape[2], 7)
    z_slice = st.sidebar.slider('Select Slice', 0, data.shape[0], 32)


    # Select which model should be used to load in reconstructions
    st.sidebar.markdown("** Reconstruction Menu **")

    model_names = ['Masked_Sliced_VAE_2000EP_no_aug', 'Sliced_VAE_2000EP_Normalized', 'Sliced_ConditonalVAE_Test', 'Sliced_ConditonalReducedVAE_2000EP', 'Masked_Sliced_ConditonalReducedVAE_2000EP_no_aug']
    model_name = 'Sliced_VAE_2000EP_Normalized'
    model_name = st.sidebar.selectbox("Select model", model_names)

    return (train_or_validation, subject, time_step, z_slice, model_name)

# =================================================================================
# This is the main function to load the layout and functionalities of the web view
# =================================================================================
def load_layout():

    # Load and cache the data that we will use
    images_tr, labels_tr, images_vl, labels_vl, images_tr_sl, images_vl_sl = load_data_base()
    reconstructions = load_data_reconstructions()

    # Initialize to subject 0
    subject = 0

    # Compute dataset sizes and put them into a dictionary for easy lookup
    dataset_sizes = {
        'train': int(images_tr_sl.shape[0]/64),
        'validation': int(images_vl_sl.shape[0]/64),
        'test': 0,
        'anomalous': 0
    }

    # Initialize the subject to 0 and load the side menu
    data = images_tr_sl[subject*64:(subject+1)*64,:,:,:,:]
    train_or_validation, subject, time_step, z_slice, model_name = side_menu(data, dataset_sizes)

    # Change dataset based on radio buttons in the side menu
    if train_or_validation == 'train':
        data = images_tr_sl[subject*64:(subject+1)*64,:,:,:,:]
    elif train_or_validation == 'validation':
        data = images_vl_sl[subject*64:(subject+1)*64,:,:,:,:]


    SPACES = '&nbsp;' * 10
    st.title("4D Flow MRI Visualization")

    # If this is set to true, show the image before any processing
    show_original = st.checkbox("Show original image before preprocessing.")
    if show_original:
        if train_or_validation == 'train':
            label = labels_tr[subject*32:(subject+1)*32,:,:,:]
            original = images_tr[subject*32:(subject+1)*32,:,:,:,:]

        elif train_or_validation == 'validation':
            label = labels_vl[subject*32:(subject+1)*32,:,:,:]
            original = images_vl[subject*32:(subject+1)*32,:,:,:,:]

        load_original_view(subject, original, label)

    # Write some explanatory text describing the processing and data
    st.write("Below we show the 3 velocity channels, VX, VY, VZ for a slice and a timestep.."
            )
    st.markdown("Things to note: ".format(SPACES))
    st.markdown("{}🔹 Data is acquired by **slicing along the centerline** of the aorta.".format(SPACES))
    st.markdown("{}🔹 Images are then **normalized** to -1 and 1. ".format(SPACES))

    load_sliced_view(data, time_step, z_slice)
    load_vector_plot(data, time_step, z_slice)

    st.markdown("## Reconstructions")

    # Change dataset based on radio buttons in the side menu
    reconstruction_data = reconstructions[model_name + '_' + train_or_validation][subject*64:(subject+1)*64,:,:,:,:]

    # Threshold slider
    threshold = st.slider('Select Threshold', 0., 0.3, 0.1)
    load_reconstruction_view(data, reconstruction_data, time_step, z_slice, threshold)

    # Noisy Reconstructions
    st.markdown("## Noise Reconstructions")
    noisy_threshold = st.slider('Select Threshold', 0., 0.3, 0.1, key="noisy_threshold")

    noisy_input, noisy_reconstructions = load_data_reconstructions_noise()
    noisy_data = noisy_input[subject*64:(subject+1)*64,:,:,:,:]
    noisy_reconstruction_data = noisy_reconstructions[model_name + '_' + train_or_validation][subject*64:(subject+1)*64,:,:,:,:]

    load_reconstruction_view(noisy_data, noisy_reconstruction_data, time_step, z_slice, noisy_threshold)

# Main for webapp. This is run when running "streamlit run app.py"
if __name__ == "__main__":
    load_layout()
