# This file is mainly for experimentation and doing quick tests, delete once everything works as intended


import numpy as np
# import tensorflow as tf
"""
batch_slice_info = [0,1,2,3,4,5,6,7]

temp = []
for z_slice_location in batch_slice_info:
    one_hot = np.zeros(64)
    one_hot[z_slice_location] = 1.

    reshaped = np.reshape(one_hot, (1,1,1,1,64))
    broadcast = np.broadcast_to(reshaped, (1,32,32,48,64))
    temp.append(broadcast)

condition_matrix = np.concatenate(temp)

# ========================================================================================
# Above is test ground, below adapt to be usable in real model with configs etc...
# ========================================================================================


# Conditional VAE needs to encode the input and X and a condition c towards a latent space z -> Q(z|X,condition)
# X is the image matrix of dimensions: (batchsize, x_dim, y_dim, time, channels) = (8,32,32,48,4)
# To add the conditions we need to create a conditional matrix that captures the z location for each slice inside the image matrix
# The steps that we take to achieve this are:
#   1. convert to one_hot vector of shape (1, # of conditions/classes or slices in our case)
#   2. reshape that vector to be able to tile it to the larger image matrix
#   3. broadcast to desired shape of one volume slice of the image matrix
#   4. concatenate over all the batch items to get the full conditional matrix

temp = []
# Iterate over the slice information  
for z_slice_location in self.batch_slice_info:
    
    # For each slice, encode the position along the aorta in a one-hot vector
    one_hot = np.zeros(self.config["spatial_size_z"])
    one_hot[z_slice_location] = 1.

    # Reshape this vector into the shape of 1x1x1x1xnumber_of_categories aka the number of slices
    reshaped = np.reshape(one_hot, (1,1,1,1,self.config["spatial_size_z"]))
    
    # Broadcast this one_hot vector to the total shape of one slice volume
    broadcast = np.broadcast_to(reshaped, 
                                (1,
                                 self.config["spatial_size_x"],
                                 self.config["spatial_size_y"],
                                 self.config["spatial_size_t"],
                                 self.config["spatial_size_z"]))

    # Append this to a temp array, we are doing this for each slice volume in the batch and will concatenate after
    temp.append(broadcast)

# Concatenate the batch into one big matrix that hold the conditional information
condition_matrix = np.concatenate(temp)


# input
test_input = np.ones((8,32,32,48,4))

batch_slice_info = [0,1,2,3,4,5,6,7]
temp = []
for z_slice_location in batch_slice_info:
    one_hot = np.zeros(64)
    one_hot[z_slice_location] = 1.

    reshaped = np.reshape(one_hot, (1,1,1,1,64))
    broadcast = np.broadcast_to(reshaped, (1,32,32,48,64))
    temp.append(broadcast)

# condition
condition_matrix = np.concatenate(temp)

concatenated = tf.concat(axis=0, values=[test_input, condition_matrix])

print(concatenated.shape)
print(concatenated.size)

"""

import numpy as np

bins = np.linspace(0, 64, 5)
print(bins)

x = np.array([2,5,15,36,63])
bins = np.array([0,8,16,24,32,40,48,56,64])

inds = np.digitize(x, bins)-1
print(inds)