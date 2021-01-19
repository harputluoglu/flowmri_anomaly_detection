import numpy as np
import math

from matplotlib import pyplot as plt

# needed for data augmentation
from helpers.data_augmentation import do_data_augmentation

def iterate_minibatches(images
                        ,batch_size
                        ,data_augmentation=False):
    '''
    Author: Neerav Kharani, extended by Pol Peiffer

    Function to create mini batches from the dataset of a certain batch size
    :param images: numpy dataset
    :param labels: numpy dataset (same as images/volumes)
    :param batch_size: batch size
    :return: mini batches
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]

        # ===========================
        # augment the batch
        # ===========================

        if data_augmentation:
            X = do_data_augmentation(images=X,
                                     data_aug_ratio=0.5, 
                                     trans_min=-10,
                                     trans_max=10,
                                     rot_min=-10,
                                     rot_max=10,
                                     scale_min=0.9,
                                     scale_max=1.1)
                                     
        yield X

def iterate_minibatches_with_sliceinfo(images,
                        batch_size,
                        img_size_z,
                        data_augmentation=False):
    '''
    Author: Pol Peiffer

    Function to create mini batches from the dataset of a certain batch size and in addition return the slice
    information as we use it for the condition of the conditional VAE

    :param images: numpy dataset
    :param batch_size: batch size
    :param img_size_z: the number of slices we have determines the categories
    :return: mini batches and slice info along with it
    '''

    # So... what does our data look like?
    # expected: [img_size_z*num_images, img_size_x, vol_size_y, img_size_t, n_channels]

    # 63 -> 63
    # 256 -> 


    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]

        # information on where in the aorta this slice is from
        batch_slice_info = [x % img_size_z for x in batch_indices]
        
        # ===========================
        # augment the batch
        # ===========================
        if data_augmentation:
            X = do_data_augmentation(images=X,
                                     data_aug_ratio=0.5, 
                                     trans_min=-10,
                                     trans_max=10,
                                     rot_min=-10,
                                     rot_max=10,
                                     scale_min=0.9,
                                     scale_max=1.1)

        yield (X, np.array(batch_slice_info))

def iterate_minibatches_with_class(images,
                        batch_size,
                        number_of_classes,
                        data_augmentation=False):
    '''
    Author: Pol Peiffer

    Function to create mini batches from the dataset of a certain batch size and in addition return the class
    information as we use it for the condition of the conditional VAE

    :param images: numpy dataset
    :param batch_size: batch size
    :param number_of_classes: the number of classes that we want to return, should be able to create equal distances with 64, so use either 2/4/8/16/32/64!!
    :return: mini batches and class info along with it
    '''

    # ===========================
    # generate indices to randomly select slices in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.arange(n_images)
    np.random.shuffle(random_indices)

    # ===========================
    # using only a fraction of the batches in each epoch
    # ===========================
    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]

        # information on where in the aorta this slice is from
        batch_slice_info = [x % 64 for x in batch_indices]

        # now we need to digitize this info into the number of categories
        # Create equally spaced bins of the 64 slices
        bins = np.linspace(0, 64, number_of_classes + 1)

        # digitize and subtract 1 to get classes starting from 0 (digitize starts counting at 1 for the first bin)
        inds = np.digitize(np.array(batch_slice_info), bins)-1
        
        # ===========================
        # augment the batch
        # ===========================
        if data_augmentation:
            X = do_data_augmentation(images=X,
                                     data_aug_ratio=0.5, 
                                     trans_min=-10,
                                     trans_max=10,
                                     rot_min=-10,
                                     rot_max=10,
                                     scale_min=0.9,
                                     scale_max=1.1)

        yield (X, np.array(inds))

def tile(X, rows, cols):
    """Tile images for display."""
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype = X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx,...]
                tiling[
                        i*X.shape[1]:(i+1)*X.shape[1],
                        j*X.shape[2]:(j+1)*X.shape[2],
                        :] = img
    return tiling

def plot_batch(X, out_path):
    """
    Author: Chen
    Save batch of images tiled."""

    X=np.stack(X)
    n_channels = X.shape[3]
    if n_channels > 3:
        X = X[:,:,:,np.random.choice(n_channels, size = 3)]
    #X = postprocess(X)
    rc = math.sqrt(X.shape[0])
    rows = cols = math.ceil(rc)
    canvas = tile(X, rows, cols)
    canvas = np.squeeze(canvas)
    #plt.imsave(out_path, canvas)
    plt.imsave(out_path, canvas)


# ============================================
# Batch plotting helper functions
# ============================================

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

def plot_batch_3d(X, channel, every_x_time_step, out_path):

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
    plt.imsave(out_path, canvas, cmap='gray')

def tile_3d_complete(X, Out_Mu, rows, cols, every_x_time_step):
    """
    Tile images for display.

    Each patient slice in a batch has the following:
    ----------------------------------------------
    1. Row: original image for channel
    2. Out_Mu
    3. Difference
    ----------------------------------------------
    """
    row_separator_counter = 0
    row_seperator_width = 1

    tiling = np.zeros((rows * X.shape[1] * 3 * 4 + rows * row_seperator_width * 3 * 4, cols * X.shape[2] + cols), dtype = X.dtype)

    #Loop through all the channels
    i = 0
    subject = 0

    # Rows is the number of samples in a batch, 3 comes from the fact that we draw 3 rows per subject. We have 4 channels.
    while i < (rows*3*4-1):
        for channel in range(4):
            for j in range(cols):
                img = X[subject,:,:,j*every_x_time_step,channel]
                out_mu = Out_Mu[subject,:,:,j*every_x_time_step,channel]
                difference = np.absolute(img - out_mu)

                separator_offset= row_separator_counter*row_seperator_width

                # Original input image
                tiling[
                        i*X.shape[1] + separator_offset:(i+1)*X.shape[1] + separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = img

                # Autoencoder prediction
                tiling[
                        (i+1)*X.shape[1]+ separator_offset:(i+2)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = out_mu

                # Difference of the images
                tiling[
                        (i+2)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset,
                        j*X.shape[2]:(j+1)*X.shape[2]] = difference

            # White line to separate this from the next channel
            tiling[
                    (i+3)*X.shape[1]+ separator_offset:(i+3)*X.shape[1]+ separator_offset + row_seperator_width,
                    0:(cols-1)*X.shape[2]] = 1

            # Increase row separator count
            row_separator_counter += 1

            #One channel is now complete, so increase i by 3 (three rows are done)
            i += 3

        #One subject is now complete, so move to the next subject in the batch
        subject += 1

    return tiling

def plot_batch_3d_complete(batch, Out_Mu, every_x_time_step, out_path):
    X = np.stack(batch)
    Out_Mu = np.stack(Out_Mu)
    rows = X.shape[0]
    cols = math.ceil(X.shape[3] // every_x_time_step)
    canvas = tile_3d_complete(X, Out_Mu, rows, cols, every_x_time_step)
    canvas = np.squeeze(canvas)
    plt.imsave(out_path, canvas, cmap='gray')


# ============================================
# Vector visualization helpers
# ============================================

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def plot_vector_field_3d(sliced_crosssection, timestep):
    """
     param: sliced_crossection - input of dimensions (x, y, t, channels) - dimensions 32x32x48x4
     param: timestep - the timestep at which we want to plot the vectorfield

    """
    plt.figure(figsize=[5,25])
    plot1 = plt.subplot(1,3,1)
    plt.imshow(sliced_crosssection[:,:,timestep,1], cmap='gray')
    plt.colorbar()
    plot1.set_title("VX")

    plot2 = plt.subplot(1,3,2)
    plt.imshow(sliced_crosssection[:,:,timestep,2], cmap='gray')
    plt.colorbar()
    plot2.set_title("VY")

    plot3 = plt.subplot(1,3,3)
    plt.imshow(sliced_crosssection[:,:,timestep,3], cmap='gray')
    plt.colorbar()
    plot3.set_title("VZ")

    fig = plt.figure(figsize=[12,12])
    ax = fig.gca(projection='3d')

    # Make the grid
    x = np.arange(0,1,0.032)
    y = np.arange(0,1,0.032)
    z = np.arange(0,1,1)

    #Create the mesh for matplotlib
    X, Y, Z = np.meshgrid(x, y, z)

    #select every xth point to draw a vector
    selector = 1

    u = np.zeros([32,32,1])
    v = np.zeros([32,32,1])
    w = np.zeros([32,32,1])


    for i in range(32):
        for j in range(32):
            if (i % selector == 0 and j % selector == 0):
                u[i,j, 0] = sliced_crosssection[i,j, timestep, 1]  #channel vx
                v[i,j, 0] = sliced_crosssection[i,j, timestep, 2]  #channel vy
                w[i,j, 0] = sliced_crosssection[i,j, timestep, 3]  #channel vz

    #plt.imshow(sliced_crosssection[:,:,10,1], cmap='gray')
    ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=False)
    ax.set_xlabel('$Velocity-X$', fontsize=20, rotation=150)
    ax.set_ylabel('$Velocity-Y$', fontsize=20)
    ax.set_zlabel('$Velocity-Z$', fontsize=20, rotation=60)

    plt.show()
