# ========================================================================================
# Imports of needed libraries
# ========================================================================================
import numpy as np
import math




# ========================================================================================
# RMSE
# ========================================================================================

def rmse(x,y):
    """
    Returns the root mean squared error between an original and predicted tensor
    :parameter: x - original
    :parameter: y - prediction
    """    
    result = np.sqrt(np.mean((x-y)**2))
    return result


# ========================================================================================
# DICE score
# ========================================================================================

def to_binary(img, lower, upper):
    """
    Return binary image based on upper and lower thresholds.
    """
    return (lower < img) & (img < upper)

def dice_score(ground_truth_mask, image, threshold_value):
    """
    Computes the dice score between a ground truth mask and a target image.
    Takes threshold value to create a binary image from an input image (in our case we would input the difference image)
    """
    # Create a binary image using threshold value
    seg = to_binary(image, -threshold_value, threshold_value)

    # Compute dice score
    dice = np.sum(seg[ground_truth_mask==1.0])*2.0 / (np.sum(seg) + np.sum(ground_truth_mask))

    return dice

# ========================================================================================
# ROC and AUC
# ========================================================================================
from sklearn.metrics import confusion_matrix

def compute_tpr_fpr(added_noise_mask, reconstruction_difference, threshold):
    """
        added_noise_mask: shape of (batch_size,x,y,t,channels). A mask of where in the image we added the noise patch
        reconstruction_difference: abs(noisy_input_image - decoder_output)
        threshold: the threshold at which we will set the difference to 0 to get rid of only small reconstruction errors
    """
    
    # create a copy to not overwrite the original data
    difference_ = np.copy(reconstruction_difference)

    # where the difference is below threshold, set it to 0
    difference_[difference_ <= threshold] = 0.
    difference_[difference_  > threshold] = 1.

    # Flatten the matrices into 1d
    added_noise_mask = np.ravel(added_noise_mask)
    difference_ = np.ravel(difference_)

    # compute true negatives, false positives, false negatives and true positives using confusion matrix
    tn, fp, fn, tp = confusion_matrix(added_noise_mask, difference_).ravel()

    # compute true positive rate and false positive rate
    tpr = tp /(tp + fn)
    fpr = fp /(fp + tn)

    return (tpr, fpr)




# TODO get rid of this function or complete it, currently unfinished
def compute_auc(added_noise_mask, reconstruction_difference, thresholds):

    """
        added_noise_mask: shape of (x,y,t,channels). A mask of where in the image we added the noise patch
        reconstruction_difference: abs(noisy_input_image - decoder_output)
        thresholds: the thresholds at which we will set the difference to 0 to get rid of only small reconstruction errors
    """

    # compute total number of positives and negatives in the noise mask
    total_p = np.sum(added_noise_mask == 1.)
    total_n = np.sum(added_noise_mask == 0.)

    true_positives = []
    false_positives = []

    for threshold in thresholds:

        # create a copy to not overwrite the original data
        difference_ = np.copy(reconstruction_difference)

        # where the difference is below threshold, set it to 0
        difference_[difference_ <= threshold] = 0.
        difference_[difference_  > threshold] = 1.

        # logical operations will help us find the true positives and false positives





