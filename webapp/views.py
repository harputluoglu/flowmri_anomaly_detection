# =================================================================================
# ============== GENERAL PACKAGE IMPORTS ==========================================
# =================================================================================

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import logging
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import skeletonize_3d
from scipy import interpolate
import os
import imageio

from helpers.metrics import rmse


# =================================================================================
# Visalization helper functions below
#
# =================================================================================
def tile_3d(X, rows, cols, every_x_time_step):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            img = X[i,:,:,j*every_x_time_step]
            tiling[
                    i*X.shape[1]:(i+1)*X.shape[1],
                    j*X.shape[2]:(j+1)*X.shape[2]] = img
    return tiling

def plot_batch_3d(X, channel, every_x_time_step):

    """
    This method creates a plot of a batch

    param: X - input of dimensions (batches, x, y, t,  channels)
    param: channel - which channel of the images should be plotted (0-3):(intensity,vx,vy,vz)
    param: every_x_time_step - for 1, all timesteps are plotted, for 2, every second timestep is plotted etc..
    param: out_path - path of the folder where the plots should be saved
    """

    X = np.stack(X)
    X = X[:,:,:,:,channel]

    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d(X, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)

    return canvas

def tile_model_reconstruction_comparison(data, reconstruction_data, time_step, z_slice, threshold):

    rows = 4
    cols = 3
    tiling = np.zeros((rows * data.shape[1], cols * data.shape[2]), dtype = data.dtype)

    rmses = []

    for j in range(cols):
        img = data[z_slice,:,:,time_step,j+1]
        reconstruction = reconstruction_data[z_slice,:,:,time_step,j+1]
        difference = np.abs(img-reconstruction)


        # Input
        i = 0
        tiling[
            i*data.shape[1]:(i+1)*data.shape[1],
            j*data.shape[2]:(j+1)*data.shape[2]] = img

        # Reconstruction
        i = 1
        tiling[
            i*data.shape[1]:(i+1)*data.shape[1],
            j*data.shape[2]:(j+1)*data.shape[2]] = reconstruction

        # Difference
        i = 2
        tiling[
            i*data.shape[1]:(i+1)*data.shape[1],
            j*data.shape[2]:(j+1)*data.shape[2]] = difference

        # Thresholded difference
        difference[difference <= threshold] = 0.
        i = 3
        tiling[
            i*data.shape[1]:(i+1)*data.shape[1],
            j*data.shape[2]:(j+1)*data.shape[2]] = difference

        error = rmse(img, reconstruction)
        rmses.append(error)

    difference = np.abs(data[z_slice,:,:,time_step,:] - reconstruction_data[z_slice,:,:,time_step,:])
    difference[difference <= threshold] = 0.

    return tiling, difference, rmses

def create_gif(data, reconstruction_data, z_slice, threshold, model_name):

    # Create the needed directories for the gif if they don't exist already
    try:
        temp_dir = 'webapp/gifs/temp'
        completed_dir = 'webapp/gifs/completed/' + model_name

        if not os.path.exists(os.path.join(project_code_root, temp_dir)):
            os.makedirs(os.path.join(project_code_root, temp_dir))

        if not os.path.exists(os.path.join(project_code_root, completed_dir)):
            os.makedirs(os.path.join(project_code_root, completed_dir))

    except OSError:
        print("Could not create output directories for evaluation")
        pass


    difference = np.abs(data[z_slice] - reconstruction_data[z_slice])
    difference[difference <= threshold] = 0.

    # Create the individual images that will be assembled into a gif
    for time_step in data.shape[3]:

        plt.figure(figsize=[10,5])
        plot1 = plt.subplot(1,4,1)
        plt.imshow(data[z_slice,:,:,time_step,1], cmap='gray')
        plt.imshow(difference[:,:,1], cmap='jet', alpha=0.35)
        plot1.set_title("VX")

        plot2 = plt.subplot(1,4,2)
        plt.imshow(data[z_slice,:,:,time_step,2], cmap='gray')
        plt.imshow(difference[:,:,2], cmap='jet', alpha=0.35)
        plot2.set_title("VY")

        plot3 = plt.subplot(1,4,3)
        plt.imshow(data[z_slice,:,:,time_step,3], cmap='gray')
        plt.imshow(difference[:,:,3], cmap='jet', alpha=0.35)
        plot3.set_title("VZ")

        plot4 = plt.subplot(1,4,4)
        plt.imshow(np.linalg.norm(data[z_slice,:,:,time_step,:], axis=-1), cmap='gray')
        plot4.set_title("Norm of Velocities")

        plt.savefig(temp_dir + '/' + 'Image_{:02d}'.format(time_step))
        plt.close()

    # Now assemble the frames into a gif
    anim_file = 'cvae.gif'

    with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
      filenames = glob.glob('Test/Image_*.png')
      filenames = sorted(filenames)
      for i,filename in enumerate(filenames):
        print(filename)
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)



# =================================================================================
# Views below
# =================================================================================
def load_original_view(subject, data, label):

    SPACES = '&nbsp;' * 10

    st.write("Below the original image, before slicing:")

    time_step = st.slider('Select Timestep for Original', 0, data.shape[3], 7)
    z_slice = st.slider('Select z-Slice for Original', 0, data.shape[0], 15)
    show_mask = st.checkbox("Show segmentation")

    plt.figure(figsize=[15,8])

    plot0 = plt.subplot(1,4,1)
    plt.imshow(data[z_slice,:,:,time_step,0], cmap='gray')
    plt.colorbar()
    plot0.set_title("Intensity")

    plot1 = plt.subplot(1,4,2)
    plt.imshow(data[z_slice,:,:,time_step,1], cmap='gray')
    plt.colorbar()
    plot1.set_title("VX")

    plot2 = plt.subplot(1,4,3)
    plt.imshow(data[z_slice,:,:,time_step,2], cmap='gray')
    plt.colorbar()
    plot2.set_title("VY")

    plot3 = plt.subplot(1,4,4)
    plt.imshow(data[z_slice,:,:,time_step,3], cmap='gray')
    plt.colorbar()
    plot3.set_title("VZ")

    st.pyplot()

    # load_vector_plot_original(data, time_step, z_slice)

    # If show_mask is set to true, show the segmentation and compute the centerline
    if show_mask:
        plt.figure(figsize=[15,8])


        centerline_indexes = [
            [115, 81, 43, 7, 52, 120, 160],
            [87, 44, 11, 19, 89, 119, 131],
            [72, 30, 7, 49, 89, 119, 150],
            [94, 31, 0, 34, 81, 120, 151],
            [74, 34, 15, 35, 79, 110],
            [73, 14, 5, 34, 74, 119],
            [94, 54, 0, 55, 121, 151],
            [100, 52, 12, 63, 112, 134],
            [95, 55, 15, 40, 94, 118, 157],
            [105, 70, 16, 32, 93, 129, 162],
            [78, 34, 21, 35, 104, 134, 164],
            [74, 34, 6, 35, 100, 120, 153],
            [109, 71, 9, 26, 92, 131, 142, 174],
            [88, 21, 15, 66, 111, 141, 167]
        ]

        # Average the segmentation over time (the geometry should be the same over time)
        avg = np.average(label, axis = 3)

        # Compute the centerline points of the skeleton
        skeleton = skeletonize_3d(avg[:,:,:])

        # Get the points of the centerline as an array
        points = np.array(np.where(skeleton != 0)).transpose([1,0])
        points = points[np.where(points[:,2]<60)]
        points = points[np.where(points[:,1]<100)]

        # Order the points in ascending order with x
        points = points[points[:,1].argsort()[::-1]]

        temp = []
        for index, element in enumerate(points[5:]):
            if (index%10)==0:
                temp.append(element)

        coords = np.array(temp)

        # spline parametrization
        size = [32,32,64]
        params = [i / (size[2] - 1) for i in range(size[2])]
        tck, _ = interpolate.splprep(np.swapaxes(coords, 0, 1), k=3, s=200)

        # derivative is tangent to the curve
        spline_points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)


        plt.figure(figsize=[15,8])

        plot0 = plt.subplot(1,2,1)
        plt.imshow(label[z_slice,:,:,time_step], cmap='gray')
        plt.colorbar()
        plot0.set_title("Segmentation Mask")

        plot1 = plt.subplot(1,2,2)
        plt.imshow(data[z_slice,:,:,time_step,1], cmap='gray')
        plt.scatter(spline_points[:,2],spline_points[:,1], s=5, c='red', marker='o')
        #plt.scatter(points[:,2],points[:,1], s=2, c='blue', marker='o')
        plot1.set_title("Centerline Points")

        st.pyplot()

def load_sliced_view(data, time_step, z_slice):

    plt.figure(figsize=[10,5])
    plot1 = plt.subplot(1,3,1)
    plt.imshow(data[z_slice,:,:,time_step,1], cmap='gray')
    plt.colorbar()
    plot1.set_title("VX")

    plot2 = plt.subplot(1,3,2)
    plt.imshow(data[z_slice,:,:,time_step,2], cmap='gray')
    plt.colorbar()
    plot2.set_title("VY")

    plot3 = plt.subplot(1,3,3)
    plt.imshow(data[z_slice,:,:,time_step,3], cmap='gray')
    plt.colorbar()
    plot3.set_title("VZ")

    st.pyplot()

def load_reconstruction_view(data, reconstruction_data, time_step, z_slice, threshold):

    SPACES = '&nbsp;' * 10

    tiled, difference, rmses = tile_model_reconstruction_comparison(data, reconstruction_data, time_step, z_slice, threshold)

    plt.figure(figsize=[10,5])
    plot1 = plt.subplot(1,3,1)
    plt.imshow(data[z_slice,:,:,time_step,1], cmap='gray')
    plt.imshow(difference[:,:,1], cmap='jet', alpha=0.35)
    plot1.set_title("VX")

    plot2 = plt.subplot(1,3,2)
    plt.imshow(data[z_slice,:,:,time_step,2], cmap='gray')
    plt.imshow(difference[:,:,2], cmap='jet', alpha=0.35)
    plot2.set_title("VY")

    plot3 = plt.subplot(1,3,3)
    plt.imshow(data[z_slice,:,:,time_step,3], cmap='gray')
    plt.imshow(difference[:,:,3], cmap='jet', alpha=0.35)
    plot3.set_title("VZ")
    st.pyplot()

    plt.figure(figsize=[20,20])
    plt.imshow(tiled, cmap='gray')
    st.pyplot()

    st.markdown("**Legend**: ".format(SPACES))
    st.markdown("{} **Row 1**: input images".format(SPACES))
    st.markdown("{} **Row 2**: model reconstructions".format(SPACES))
    st.markdown("{} **Row 3**: difference between input and reconstructions".format(SPACES))
    st.markdown("{} **Row 4**: difference between input and reconstructions, threshold {}".format(SPACES, str(threshold)))

    st.markdown("{} **RMSEs**: Column 1 VX {}, Column 2 VY {}, Column 3 VZ {}".format(SPACES,str(rmses[0]), str(rmses[1]), str(rmses[2])))

def load_vector_plot_original(data, time_step, z_slice):
    """
    Loads the vector visualization in plotly
    """

    # Make the grid
    x = []
    y = []
    z = []

    u = []
    v = []
    w = []

    selector = 1
    for i in range(144):
        for j in range(112):
            if (i % selector == 0 and j % selector == 0):

                x.append(i)
                y.append(j)
                z.append(0)

                u.append(data[z_slice,i,j, time_step, 1])
                v.append(data[z_slice,i,j, time_step, 2])
                w.append(data[z_slice,i,j, time_step, 3])

    fig = go.Figure(data = go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        colorscale='magenta',
        sizemode='absolute',
        sizeref=2,
        showscale=True,
        colorbar=dict(thickness=20, ticklen=4),
        ))

    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                                camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    st.write("Below we show the 3D Vector visualization of the above 3 channels. Color according to norm of the vectors."
        )
    st.plotly_chart(fig, width = 1, height= 800)

def load_vector_plot(data, time_step, z_slice):
    """
    Loads the vector visualization in plotly
    """

    # Make the grid
    x = []
    y = []
    z = []

    u = []
    v = []
    w = []

    selector = 1
    for i in range(32):
        for j in range(32):
            if (i % selector == 0 and j % selector == 0):

                x.append(i)
                y.append(j)
                z.append(0)

                u.append(data[z_slice,i,j, time_step, 3])
                v.append(-data[z_slice,i,j, time_step, 2])
                w.append(-data[z_slice,i,j, time_step, 1])

    fig = go.Figure(data = go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        colorscale='magenta',
        sizemode='absolute',
        sizeref=25,
        showscale=True,
        colorbar=dict(thickness=20, ticklen=4),
        ))

    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                                camera_eye=dict(x=1.2, y=1.2, z=0.6)))

    st.write("Below we show the 3D Vector visualization of the above 3 channels. Color according to norm of the vectors."
        )
    st.plotly_chart(fig, width = 1, height= 800)

def load_gif_view(data, z_slice):

    return 0
