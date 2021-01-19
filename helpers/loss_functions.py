import tensorflow as tf
import numpy as np


def l2loss(x,y):
    """
    Computes l2 loss of input x and output y.
    :param x: input
    :param y: output of the network
    :result: L2 Loss
    """

    averaged = tf.reduce_mean((y - x)**2,
                           [1, 2, 3, 4])
    l2_loss = averaged
    return l2_loss


def kl_loss_1d(z_mean,z_stddev):
    """
    Computes the Kullback Leibler divergence for flattened 1D latent space
    """
    latent_loss = tf.reduce_mean(
        tf.square(z_mean) + tf.square(z_stddev)- 2.*tf.log(tf.abs(z_stddev+1e-7)) - 1, [1,2,3,4])


    return 0.5*latent_loss
