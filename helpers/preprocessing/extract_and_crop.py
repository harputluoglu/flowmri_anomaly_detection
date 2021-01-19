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
from skimage.morphology import erosion, skeletonize_3d, skeletonize



def extract_slice_from_sitk_image(sitk_image, point, Z, X, new_size, fill_value=0):
    """
    Extract oblique slice from SimpleITK image. Efficient, because it rotates the grid and
    only samples the desired slice.

    """
    num_dim = sitk_image.GetDimension()

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())

    new_size = [int(el) for el in new_size]  # SimpleITK expects lists, not ndarrays
    point = [float(el) for el in point]

    rotation_center = sitk_image.TransformContinuousIndexToPhysicalPoint(point)

    X = X / np.linalg.norm(X)
    Z = Z / np.linalg.norm(Z)
    assert np.dot(X, Z) < 1e-12, 'the two input vectors are not perpendicular!'
    Y = np.cross(Z, X)

    orig_frame = np.array(orig_direction).reshape(num_dim, num_dim)
    new_frame = np.array([X, Y, Z])

    # important: when resampling images, the transform is used to map points from the output image space into the input image space
    rot_matrix = np.dot(orig_frame, np.linalg.pinv(new_frame))
    transform = sitk.AffineTransform(rot_matrix.flatten(), np.zeros(num_dim), rotation_center)

    phys_size = new_size * orig_spacing
    new_origin = rotation_center - phys_size / 2

    resample_filter = sitk.ResampleImageFilter()
    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   transform,
                                                   sitk.sitkLinear,
                                                   new_origin,
                                                   orig_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)
    return resampled_sitk_image



# ==========================================
# loads the numpy array saved from the dicom files of the Freiburg dataset
# ==========================================
def load_npy_data(subject):
    img_path = os.path.join(os.getcwd(), '../../data/freiburg/')

    npy_files_list = []

    for _, _, file_names in os.walk(img_path):

        for file in file_names:

            if '.npy' in file:
                npy_files_list.append(file)

    # use passed subject numer to index into files list
    path = img_path + '{}'.format(npy_files_list[subject])
    array = np.load(path)

    return array


# ==========================================
# function to normalize the input arrays (intensity and velocity) to a range between 0 to 1 and -1 to 1
# magnitude normalization is a simple division by the largest value
# velocity normalization first calculates the largest magnitude and then uses the components of this vector to normalize the x,y and z directions seperately
# ==========================================
def normalize_arrays(arrays):

    # dimension of normalized_arrays: 128 x 128 x 20 x 25 x 4
    normalized_arrays = np.zeros((arrays.shape))

    # normalize magnitude channel
    normalized_arrays[...,0] = arrays[...,0]/np.amax(arrays[...,0])

    # normalize velocities
    # extract the velocities in the 3 directions
    velocity_arrays = np.array(arrays[...,1:4])
    # denoise the velocity vectors
    velocity_arrays_denoised = gaussian_filter(velocity_arrays, 0.5)
    # compute per-pixel velocity magnitude
    velocity_mag_array = np.linalg.norm(velocity_arrays_denoised, axis=-1)
    # velocity_mag_array = np.sqrt(np.square(velocity_arrays[...,0])+np.square(velocity_arrays[...,1])+np.square(velocity_arrays[...,2]))
    # find max value of 95th percentile (to minimize effect of outliers) of magnitude array and its index
    vpercentile =  np.percentile(velocity_mag_array, 95)
    normalized_arrays[...,1] = velocity_arrays_denoised[...,0] / vpercentile
    normalized_arrays[...,2] = velocity_arrays_denoised[...,1] / vpercentile
    normalized_arrays[...,3] = velocity_arrays_denoised[...,2] / vpercentile
    # print('normalized arrays: max=' + str(np.amax(normalized_arrays)) + ' min:' + str(np.amin(normalized_arrays)))

    return normalized_arrays
