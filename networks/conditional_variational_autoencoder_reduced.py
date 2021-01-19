import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from pdb import set_trace as bp

# ========================================================
# Residual block for downsampling
# ========================================================
def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        # Initialization
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)

        # Convolution block 1
        conv1 = tf.layers.conv3d(inputs, filters_in,(3,3,3), (2,2,2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)

        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)

        # Convolution block 2
        conv2 = tf.layers.conv3d(conv1, filters_out, (3, 3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2',reuse=reuse)
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

        # Convolution block 3
        conv3 = tf.layers.conv3d(inputs, filters_out, (3, 3, 3), (2, 2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)

        conv_out = tf.add(conv2,conv3)
    return conv_out


# ========================================================
# Resiudal block for upsampling
# ========================================================
def resblock_up(inputs, filters_in, filters_out, scope_name, reuse, phase_train,
                act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)



        conv1 = tf.layers.conv3d_transpose(inputs, filters_in, (3, 3, 3), (2, 2, 2), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                             scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init,
                                              name='rbu_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)


        conv2 = tf.layers.conv3d_transpose(conv1, filters_out, (3, 3, 3), (1, 1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2', reuse=reuse)
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)


        conv3 = tf.layers.conv3d_transpose(inputs, filters_out, (3, 3, 3), (2, 2, 2), padding='same', kernel_initializer=w_init,
                                          bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init,  name='rbu_bn3',
                                              reuse=reuse)
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)


        conv_out = tf.add(conv2, conv3)
    return conv_out


class ConditionalVariationalAutoencoderReduced():
    def __init__(self,model_name=None, image_size=32):
        self.model_name = model_name
        self.image_size = image_size
        self.z_mean = None
        self.z_stddev = None

    def MLP(self, inputs, n_output, name):
        dense = tf.layers.dense(inputs, n_output*2, name=name+"_dense1")
        dense = tf.nn.leaky_relu(dense, 0.2)
        dense = tf.layers.dense(dense, n_output*4, name=name+"_dense2")
        dense = tf.nn.leaky_relu(dense, 0.2)
        dense = tf.layers.dense(dense, n_output, name=name+"_dense3")
        return dense

    def instance_norm(self, inputs, inputs_latent, name):
        inputs_rank = inputs.shape.ndims
        n_outputs = np.int(inputs.shape[-1])
        n_batch = np.int(inputs.shape[0])
        inputs_latent_flatten = tf.layers.flatten(inputs_latent)
        gamma = self.MLP(inputs_latent_flatten, n_outputs, name+"_gamma")
        beta = self.MLP(inputs_latent_flatten, n_outputs, name+"_beta")
        gamma = tf.reshape(gamma, [n_batch, 1, 1, n_outputs])
        beta = tf.reshape(beta, [n_batch, 1, 1, n_outputs])
        moments_axes = list(range(inputs_rank))
        mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
        outputs = nn.batch_normalization(
            inputs, mean, variance, beta, gamma, 1e-6, name=name)
        return outputs

    def encoder(self, x, reuse=False, is_train=True):
        """
        Encoder network of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """

        image_size = self.image_size
        gf_dim = 12  # Dimension of gen filters in first conv layer. 4 channels + 8 classes

        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):

            # ========================================================
            # Initialization
            # ========================================================
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)


            # =========================================================
            # 1st Conv block - one conv layer, followed by batch normalization and activation
            # =========================================================
            conv1 = tf.layers.conv3d(x, gf_dim, (3, 3, 3), padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     trainable=True,
                                     name="e_conv1",
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                 scale=True, trainable=True,
                                                gamma_initializer=gamma_init,
                                                  training=is_train,
                                                 name='e_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            # =========================================================
            # Res-Blocks (for effective deep architecture)
            # =========================================================
            res1 = resblock_down(conv1, gf_dim, gf_dim, "res1", reuse, is_train)

            res2 = resblock_down(res1, gf_dim, gf_dim * 2, "res2", reuse, is_train)

            res3 = resblock_down(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)

            res4 = resblock_down(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)

            # =========================================================
            # Convolutions to compute latent mu and latent standard deviation
            # =========================================================
            conv_latent = tf.layers.conv3d(res4, gf_dim*32, (1,1,1), padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="latent_mu",
                                     reuse=reuse)
            conv_latent_std = tf.layers.conv3d(res4, gf_dim * 32, (1, 1, 1), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="latent_std",
                                           reuse=reuse)

            # =========================================================
            # TODO: flatten the latent mu and standard deviation?
            # =========================================================



            # =========================================================
            # TODO: Ask Chen what these blocks are doing, they seem
            # to be computing residuals that flow into the loss function?

            # ANSWER: these are not really needed anymore and were more of
            # a test to include residual loss
            # =========================================================
            conv2 = tf.layers.conv3d(conv1, gf_dim, (3,3,3), dilation_rate =2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="stddev_conv2",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

            # Convolution block ====================================
            conv3 = tf.layers.conv3d(conv2, gf_dim*2, (3,3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,name="stddev_conv3",
                                     reuse=reuse)
            conv3 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn3',
                                                  reuse=reuse)
            conv3 = tf.nn.leaky_relu(conv3, 0.2)

            # Convolution block ====================================
            conv4 = tf.layers.conv3d(conv3, gf_dim, (3,3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="stddev_conv4",
                                     reuse=reuse)
            conv4 = tf.layers.batch_normalization(conv3,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn4',
                                                  reuse=reuse)
            conv4 = tf.nn.leaky_relu(conv3, 0.2)

            # Convolution block ====================================
            conv5 = tf.layers.conv3d(conv4, 1, (3,3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="stddev_conv5",
                                     reuse=reuse)


            # ====================================
            # Print shapes at various layers in the encoder network
            # ====================================
            print('===================== ENCODER NETWORK ====================== ')
            print('Shape of input: ' + str(x.shape))
            print('Shape after 1st convolution block: ' + str(conv1.shape))
            print('Shape after 1st res block: ' + str(res1.shape))
            print('Shape after 2nd res block: ' + str(res2.shape))
            print('Shape after 3rd res block: ' + str(res3.shape))
            print('Shape after 4th res block: ' + str(res4.shape))
            print('-------------------------------------------------')
            print('Shape of latent_Mu: ' + str(conv_latent.shape))
            print('Shape of latent_stddev: ' + str(conv_latent_std.shape))
            print('-------------------------------------------------')
            print('=========================================================== ')

        return conv_latent, conv_latent_std, conv5

    def decoder(self, z, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        # Dimension of gen filters in first conv layer. [64]
        gf_dim = 12

        with tf.variable_scope("predictor" , reuse=reuse):

            # ========================================================
            # Initialization
            # ========================================================
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.ones_initializer()

            print(' Input to decoder has the following shape:' + str(z.shape))

            # =========================================================
            # Res-Blocks (for effective deep architecture)
            # =========================================================
            resp1 = resblock_up(z, gf_dim * 32, gf_dim * 16, "decode_resp1", reuse, is_train)

            res0 = resblock_up(resp1, gf_dim * 16, gf_dim * 8, "decode_res0", reuse, is_train)

            res1 = resblock_up(res0, gf_dim * 8, gf_dim * 4, "decode_res1", reuse, is_train)

            res2 = resblock_up(res1, gf_dim * 4, gf_dim * 2, "decode_res2", reuse, is_train)

            # =========================================================
            # 1st convolution block: convolution, followed by batch normalization and activation
            # =========================================================
            conv1 = tf.layers.conv3d(res2, gf_dim, (3, 3, 3), (1, 1, 1),
                                     padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,
                                     name="decode_conv1", reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                  scale=True, trainable=True,
                                                  gamma_initializer=gamma_init,
                                                  training=is_train,
                                                  name='decode_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            # =========================================================
            # 2nd convolution block: convolution
            # =========================================================
            conv2 = tf.layers.conv3d(conv1, 4, (3, 3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="decode_conv2",
                                     reuse=reuse)


            print('===================== DECODER NETWORK ====================== ')
            print('Shape of input: ' + str(z.shape))
            print('Shape after 1st convolution block: ' + str(resp1.shape))
            print('Shape after 1st res block: ' + str(res0.shape))
            print('Shape after 2nd res block: ' + str(res1.shape))
            print('Shape after 3rd res block: ' + str(res2.shape))
            #print('Shape after 4th res block: ' + str(res3.shape))
            #print('Shape after 5th res block: ' + str(res4.shape))
            print('-------------------------------------------------')
            print('Shape after 1st convolution ' + str(conv1.shape))
            print('Shape of output of decoder' + str(conv2.shape))
            print('-------------------------------------------------')
            print('=========================================================== ')

        return conv2
