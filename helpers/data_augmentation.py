# ===============================================================
# import packages
# ===============================================================
import numpy as np
import scipy.ndimage.interpolation
from skimage import transform

# ===============================================================
# ===============================================================
def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

# ===========================      
# Data augmentation: (2d) translations, (2d) rotations, scaling
# One slice has dimensions x-y-t-channels
# We need to perform the 2d translations on each of the timesteps and each channel
# ===========================        
def do_data_augmentation(images, # [batch_size, x, y, t, channels]
                         data_aug_ratio, # between 0.0 and 1.0
                         trans_min, # -10
                         trans_max, # 10
                         rot_min, # -10
                         rot_max, # 10
                         scale_min, # 0.9
                         scale_max): # 1.1

    """
    Augments data by applying rotations, translations and scaling to the data.
    :param: data_aug_ratio: between 0 and 1, percent of images to be augmented
    :param: trans_min: translation minimum (for example -10 pixels)
    :param: trans_max: translation maximum (for example 10 pixels)
    :param: rot_min: rotation minimum (for example -10 degrees)
    :param: rot_max: rotation maximum (for example +10 degrees)
    :param: scale_min: scaling factor minimum, for example 0.9
    :param: scale_max: scaling factor maximum, for example 1.1

    return: images. Size (batch_size, x, y, t, channels)
    """

    images_ = np.copy(images)
    
    # iterate over each slice in the batch
    for i in range(images.shape[0]):

        # ===========
        # translation
        # ===========
        if np.random.rand() < data_aug_ratio:
            
            random_shift_x = np.random.uniform(trans_min, trans_max)
            random_shift_y = np.random.uniform(trans_min, trans_max)
            
            # Apply for each time step and for each channel
            for t in range(images_.shape[3]):
                for channel in range(images_.shape[4]):
                    images_[i,:,:,t,channel] = scipy.ndimage.interpolation.shift(images_[i,:,:,t,channel],
                                                                    shift = (random_shift_x, random_shift_y),
                                                                    order = 1)
            
        # ========
        # rotation
        # ========
        if np.random.rand() < data_aug_ratio:
            
            random_angle = np.random.uniform(rot_min, rot_max)

            # Apply for each time step and for each channel
            for t in range(images_.shape[3]):
                for channel in range(images_.shape[4]):
                    images_[i,:,:,t,channel] = scipy.ndimage.interpolation.rotate(images_[i,:,:,t,channel],
                                                                        reshape = False,
                                                                        angle = random_angle,
                                                                        axes = (1, 0),
                                                                        order = 1)
            
        # ========
        # scaling
        # ========
        if np.random.rand() < data_aug_ratio:
            
            n_x, n_y = images_.shape[1], images_.shape[2]
            
            scale_val = np.round(np.random.uniform(scale_min, scale_max), 2)
            
            # Apply for each time step and for each channel
            for t in range(images_.shape[3]):
                for channel in range(images_.shape[4]):
                    images_i_tmp = transform.rescale(images_[i,:,:,t,channel], 
                                                    scale_val,
                                                    order = 1,
                                                    preserve_range = True,
                                                    mode = 'constant')
            
                    images_[i,:,:,t,channel] = crop_or_pad_slice_to_size(images_i_tmp, n_x, n_y)
                        
    return images_