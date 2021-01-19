import os
import glob
import numpy as np
import logging
import gc
import h5py

import SimpleITK as sitk

import math
from scipy import interpolate
import scipy
from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt
import utils
from skimage.morphology import erosion, skeletonize_3d, skeletonize, dilation, binary_dilation

def normalize_image(image):

    # ===============
    # initialize with zeros
    # ===============
    normalized_image = np.zeros((image.shape))

    # ===============
    # normalize magnitude channel
    # ===============
    normalized_image[...,0] = image[...,0] / np.amax(image[...,0])

    # ===============
    # normalize velocities
    # ===============

    # extract the velocities in the 3 directions
    velocity_image = np.array(image[...,1:4])

    # denoise the velocity vectors
    velocity_image_denoised = gaussian_filter(velocity_image, 0.5)

    # compute per-pixel velocity magnitude
    velocity_mag_image = np.linalg.norm(velocity_image_denoised, axis=-1)

    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    #vpercentile_min = np.percentile(velocity_mag_image, 5)
    #vpercentile_max = np.percentile(velocity_mag_image, 95)

    normalized_image[...,1] = 2.*(velocity_image_denoised[...,0] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,2] = 2.*(velocity_image_denoised[...,1] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1
    normalized_image[...,3] = 2.*(velocity_image_denoised[...,2] - np.min(velocity_image_denoised))/ np.ptp(velocity_image_denoised)-1


    # normalized = 2.*(velocity_image_denoised - np.min(velocity_image_denoised))/np.ptp(velocity_image_denoised)-1
    # print('normalized arrays: max=' + str(np.amax(normalized_arrays)) + ' min:' + str(np.amin(normalized_arrays)))

    return normalized_image

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k', '1', '2', '3', '4','h','l'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.timestep = 7
    ax.channel = 0
    ax.index = volume.shape[3] // 2
    image = ax.imshow(ax.volume[:,:,ax.index,ax.timestep,ax.channel], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.colorbar(image)
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    elif event.key == '1' or event.key == '2' or event.key == '3':
        swap_channel(ax, event.key)
    elif event.key == 'l':
        next_time_step(ax)
    elif event.key == 'h':
        previous_time_step(ax)
    fig.canvas.draw()

def swap_channel(ax, channel):
    ax.channel = int(channel) -1
    volume = ax.volume
    ax.images[0].set_array(volume[:,:,ax.index,ax.timestep,ax.channel])
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index,ax.timestep,ax.channel])
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))

    #ax.imshow(ax.volume[:,:,ax.index,ax.timestep,ax.channel], cmap='gray')

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[:,:,ax.index,ax.timestep,ax.channel])
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))
    # ax.imshow(ax.volume[:,:,ax.index,ax.timestep,ax.channel], cmap='gray')

def previous_time_step(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.timestep = (ax.timestep - 1) % volume.shape[3]  # wrap around using %
    ax.images[0].set_array(volume[:,:,ax.index,ax.timestep,ax.channel])
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))
    #ax.imshow(ax.volume[:,:,ax.index,ax.timestep,ax.channel], cmap='gray')

def next_time_step(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.timestep = (ax.timestep + 1) % volume.shape[3]
    ax.images[0].set_array(volume[:,:,ax.index,ax.timestep,ax.channel])
    ax.set_title('Z-Index {}, Timestep {}, Channel {}'.format(ax.index, ax.timestep, ax.channel+1))

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


# ===========================================================================
# Load numpy image: [x,y,z,t, channel (intensity, vx, vy,vz) ]
# ===========================================================================

im = np.load('../../experiments/sample_data/image.npy')
normal = normalize_image(im)
multi_slice_viewer(normal[...,1:4])
