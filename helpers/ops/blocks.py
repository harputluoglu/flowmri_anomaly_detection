import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn

def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2',reuse=reuse)
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2,conv3)
    return conv_out

def resblock_down_in(inputs, filters_in, filters_out, scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)

        conv1 = tf.contrib.layers.instance_norm(conv1, center=True, scale=True,
                                              trainable=True, scope='rbd_in1',reuse=reuse)

        conv1 = tf.nn.leaky_relu(conv1, 0.2)

        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)

        if act:
            conv2 = tf.contrib.layers.instance_norm(conv2, center=True, scale=True,
                                                    trainable=True,
                                                    scope='rbd_in2', reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)


        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)

        if act:
            conv3 = tf.contrib.layers.instance_norm(conv3, center=True, scale=True,
                                              trainable=True,
                                              scope='rbd_in3', reuse=reuse)
            conv3 = tf.nn.leaky_relu(conv3, 0.2)


        conv_out = tf.add(conv2,conv3)
    return conv_out

def resblock_up(inputs, filters_in, filters_out, scope_name, reuse, phase_train,
                act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                             scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init,
                                              name='rbu_bn1', reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2', reuse=reuse)
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
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


def resblock_up_adabn(inputs, filters_in, filters_out, scope_name, reuse, phase_train,
                      gamma, beta, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)

        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                    trainable=True, training=phase_train,
                                    name='rbu_bn1', reuse=reuse)

        conv1 = gamma*conv1+beta

        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                    trainable=True, training=phase_train,
                                    name='rbu_bn2', reuse=reuse)
        conv2 = gamma * conv2 + beta
        if act:
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                    trainable=True, training=phase_train, name='rbu_bn3',
                                    reuse=reuse)
        conv3 = gamma * conv3 + beta
        if act:
            conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, conv3)
    return conv_out

def resblock_up_in(inputs, filters_in, filters_out, scope_name, reuse, phase_train,
                act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)

        conv1 = tf.contrib.layers.instance_norm(conv1, center=True,scale=True,
                                              trainable=True,scope='rbu_in1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)

        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        if act:
            conv2 = tf.contrib.layers.instance_norm(conv2, center=True, scale=True,
                                              trainable=True,  scope='rbu_in2',reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                          bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        if act:
            conv3 = tf.contrib.layers.instance_norm(conv3, center=True, scale=True,
                                              trainable=True,
                                              scope='rbu_in3',
                                              reuse=reuse)
            conv3 = tf.nn.leaky_relu(conv3, 0.2)

        conv_out = tf.add(conv2, conv3)
    return conv_out


def MLP(inputs, n_output, name):
    dense = tf.layers.dense(inputs, n_output*2, name=name+"_dense1")
    dense = tf.nn.leaky_relu(dense, 0.2)
    dense = tf.layers.dense(dense, n_output*4, name=name+"_dense2")
    dense = tf.nn.leaky_relu(dense, 0.2)
    dense = tf.layers.dense(dense, n_output, name=name+"_dense3")
    return dense

def instance_norm(inputs, inputs_latent, name):
    inputs_rank = inputs.shape.ndims
    n_outputs = np.int(inputs.shape[-1])
    n_batch = np.int(inputs.shape[0])
    inputs_latent_flatten = tf.layers.flatten(inputs_latent)
    gamma = MLP(inputs_latent_flatten, n_outputs, name+"_gamma")
    beta = MLP(inputs_latent_flatten, n_outputs, name+"_beta")
    gamma = tf.reshape(gamma, [n_batch, 1, 1, n_outputs])
    beta = tf.reshape(beta, [n_batch, 1, 1, n_outputs])
    moments_axes = list(range(inputs_rank))
    mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, 1e-6, name=name)
    return outputs


def resblock_up_adain(inputs, s, filters_in, filters_out, scope_name, reuse, phase_train,
                act=True, insnorm=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')

        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        if insnorm:
            conv1 = instance_norm(conv1, s, "ib_conv1")
        # conv1 = tf.contrib.layers.instance_norm(conv1, center=True,scale=True,
        #                                       trainable=True,scope='rbu_in1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)

        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        if act:
            if insnorm:
                conv2 = instance_norm(conv2, s, "ib_conv2")
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                          bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        if act:
            if insnorm:
                conv3 = instance_norm(conv3, s, "ib_conv3")
            conv3 = tf.nn.leaky_relu(conv3, 0.2)

        conv_out = tf.add(conv2, conv3)
    return conv_out


def resblock_valid_enc(inputs, filters_in, filters_out,
                       scope_name, reuse, phase_train, act=True):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), padding = 'valid', kernel_initializer = w_init,
                                 bias_initializer=b_init, name="rb_conv1")
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn1')
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, name="rb_conv2")
        conv2 = tf.layers.batch_normalization(conv2,center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,name='rb_bn2')
        #conv2 = tf.nn.leaky_relu(conv2, 0.2)

        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (1, 1), padding='valid', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)

        h = tf.shape(conv2)[1]
        w = tf.shape(conv2)[2]
        # inputs = tf.image.resize_images(inputs, tf.cast([h,w], tf.int32),
        #                                method=tf.image.ResizeMethod.BILINEAR)
        conv_out = tf.add(conv2, conv3)
        if act:
            conv_out = lrelu(conv_out, 0.2)
    return conv_out

def resblock_up_bilinear(inputs, filters_in, filters_out,
                         scope_name, reuse, phase_train, act=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.image.resize_images(conv1, tf.cast([h*2, w*2], tf.int32), method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                             scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2',reuse=reuse)
        conv2_leaky = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                            reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn3',
                                              reuse=reuse)
        input_identity = tf.image.resize_images(conv3, tf.cast([h * 2, w * 2], tf.int32),
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners =True)
        #conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, input_identity)
        if act:
            conv_out = tf.nn.leaky_relu(conv_out, 0.2)
    return conv_out
