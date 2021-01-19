import os
import tensorflow as tf
import numpy as np

import helpers.loss_functions as losses
from helpers.batches import plot_batch_3d

from pdb import set_trace as bp


class ConditionalVAEReducedModel():
    def __init__(self, model, config, model_name, log_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model = model()
        self.model.__init__(model_name, image_size = config["spatial_size_x"])
        self.batch_slice_info = tf.placeholder(tf.int64, [config["batch_size"]])
        # self.batch_slice_info = np.zeros(config["batch_size"])
        self.weight = tf.placeholder("float32", name='kl_weight')
        self.model_name = model_name
        self.config = config
        self.loss = None
        self.train_op = None
        self.summary_op = None
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        self.build_network()
        self.saver = tf.train.Saver()


    def one_hotify(self, z_slice_location):
        # For each slice, encode the position along the aorta in a one-hot vector
        z_slice_location = int(z_slice_location)

        one_hot = np.zeros(self.config["number_of_classes"])
        one_hot[z_slice_location] = 1.

        # Reshape this vector into the shape of 1x1x1x1xnumber_of_categories aka the number of slices
        reshaped = np.reshape(one_hot, (1,1,1,1,self.config["number_of_classes"]))
        
        # Broadcast this one_hot vector to the total shape of one slice volume
        broadcast = np.broadcast_to(reshaped, 
                                    (1,
                                        x_dim,
                                        y_dim,
                                        t_dim,
                                    self.config["number_of_classes"]))
        
        return broadcast


    def tile_condition_matrix(self, input_or_latent):
        """
            :param input_or_latent: define whether to create the condition matrix for the input or latent space dimensions - either 'input' or'latent'
        """

        # Conditional VAE needs to encode the input and X and a condition c towards a latent space z -> Q(z|X,condition)
        # X is the image matrix of dimensions: (batchsize, x_dim, y_dim, time, channels) = (8,32,32,48,4)
        # To add the conditions we need to create a conditional matrix that captures the z location for each slice inside the image matrix
        # The steps that we take to achieve this are:
        #   1. convert to one_hot vector of shape (1, # of conditions/classes or slices in our case)
        #   2. reshape that vector to be able to tile it to the larger image matrix
        #   3. broadcast to desired shape of one volume slice of the image matrix
        #   4. concatenate over all the batch items to get the full conditional matrix

        # get the dimensions needed depending on whether we are tiling for the input or latent space
        if input_or_latent == 'input':
            x_dim = self.config["spatial_size_x"]
            y_dim = self.config["spatial_size_y"]
            t_dim = self.config["spatial_size_t"]
        
        elif input_or_latent == 'latent':
            x_dim = self.config["latent_x"]
            y_dim = self.config["latent_y"]
            t_dim = self.config["latent_t"]


        # This will give us a one_hot representation of (batch size, number of labels)
        one_hot = tf.one_hot(self.batch_slice_info, self.config["number_of_classes"])
        
        # Reshape this vector into the shape of 1x1x1x1xnumber_of_categories aka the number of slices
        reshaped = tf.reshape(one_hot, (self.config["batch_size"],1,1,1,self.config["number_of_classes"]) )

        # Broadcast this one_hot vector to the total shape of one slice volume
        broadcast = tf.broadcast_to(reshaped, 
                                        (self.config["batch_size"],
                                         x_dim,
                                         y_dim,
                                         t_dim,
                                        self.config["number_of_classes"]))
                                        
        return broadcast

    def build_network(self):
        """
        Method to take the individual parts of the network and put them together with the loss function
        into a functional model
        """

        # ==================================================================================================================================================================
        # ============== MODEL STRUCTURE TRAIN =============================================================================================================================
        # ==================================================================================================================================================================

        # Input image - Dimensions: x, y, t, 4 channels
        self.image_matrix = tf.placeholder('float32',
                                           [self.config["batch_size"], self.config["spatial_size_x"],
                                            self.config["spatial_size_y"], self.config["spatial_size_t"], 4],
                                           name='input')
        
        # Create the condition matrix for the conditional VAE      
        condition_matrix_input = self.tile_condition_matrix('input')

        # Concatenate input image with the conditional matrix to create input for encoder
        encoder_input = tf.concat(axis=4, values=[self.image_matrix, condition_matrix_input])
        
        # Run encoder network and get the latent space distribution
        self.z_mean, self.z_std, self.res = self.model.encoder(encoder_input,
                                                               is_train=True, reuse=False)

        # =======================================================================================================

        # Sample the latent space using a normal distribution (samples)
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std*samples

        # Create the latent condition matrix for the conditional VAE  
        condition_matrix_latent = self.tile_condition_matrix('latent')

        # Concatenate input image with the conditional matrix to create input for encoder
        decoder_input = tf.concat(axis=4, values=[self.guessed_z, condition_matrix_latent])

        # Pass the sampled concatenated matrix as intut into the decoder network
        self.decoder_output = self.model.decoder(decoder_input,
                                                 is_train=True, reuse=False)


        # ==================================================================================================================================================================
        # ============== MODEL STRUCTURE TEST ==============================================================================================================================
        # ==================================================================================================================================================================

        z_mean_valid, z_std_valid, self.res_test = self.model.encoder(encoder_input,
                                                                      is_train=False, reuse=True)

        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid*samples_valid

        # Concatenate input image with the conditional matrix to create input for encoder
        decoder_input_valid = tf.concat(axis=4, values=[guessed_z_valid, condition_matrix_latent])

        self.decoder_output_test = self.model.decoder(decoder_input_valid, is_train=False, reuse=True)


        # =================================================================================
        # ============== LOSS SETUP TRAIN =================================================
        # =================================================================================

        # Compute the reconstruction loss of the autoencoder
        self.autoencoder_loss = losses.l2loss(self.decoder_output, self.image_matrix)

        # Compute the residual loss of the autoencoder
        self.true_residuals = tf.abs(self.image_matrix-self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)

        # Compute the 1d Kullback Leibler Divergence of the latent space
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        # Assemple the different loss terms into our final loss objective function
        self.loss = tf.reduce_mean(100.*self.autoencoder_loss + self.weight*self.latent_loss)


        # =================================================================================
        # ============== LOSS SETUP TEST =================================================
        # =================================================================================
        self.autoencoder_loss_test = losses.l2loss(self.decoder_output_test, self.image_matrix)
        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)

        self.latent_loss_test = losses.kl_loss_1d(z_mean_valid, z_std_valid)

        self.loss_test = tf.reduce_mean(100.*self.autoencoder_loss_test + self.weight*self.latent_loss_test)

    def initialize(self):
        """
        This method initializes the tensorflow graph and model.
        """

        with tf.device("/gpu:0"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.loss)
                self.train_op = tf.group([self.train_op, update_ops])
        self.sess.run(tf.initializers.global_variables())

    def summarize(self):
        """
        Method builds scalar summaries for loss values that we want to track in tensorboard.
        """

        tf.summary.scalar("train_lat_loss", tf.reduce_mean(self.latent_loss))
        tf.summary.scalar("train_gen_loss", tf.reduce_mean(self.autoencoder_loss))
        tf.summary.scalar("train_total_loss", tf.reduce_mean(self.loss))

        tf.summary.scalar("test_gen_loss", tf.reduce_mean(self.autoencoder_loss_test))
        tf.summary.scalar("test_lat_loss", tf.reduce_mean(self.latent_loss_test))
        tf.summary.scalar("test_total_loss", tf.reduce_mean(self.loss_test))
        self.summary_op = tf.summary.merge_all()

    def train(self, input_images, batch_slice_info,  weight):
        self.input_images = input_images.astype("float32")

        feed_dict = {self.image_matrix: input_images,
                     self.batch_slice_info: batch_slice_info,
                     self.weight: weight}

        self.sess.run(self.train_op, feed_dict)

    def validate(self, input_images, batch_slice_info, weight):
        self.input_images_test = input_images.astype("float32")
        feed_dict = {self.image_matrix: self.input_images_test,
                     self.batch_slice_info: batch_slice_info,
                     self.weight: weight}

        self.out_mu_test = self.sess.run(self.decoder_output_test, feed_dict)
        self.residual_output_test = self.sess.run(self.res_test, feed_dict)

    def visualize(self, model_name, ep, project_code_root):

        # =================================================================================
        # ============== SETUP ============================================================
        # =================================================================================

        # Sample the current latent space for a z and generate an decoder output
        # Take a random class 
        cvae_class = np.random.randint(0,8)
        batch_slice_info = [cvae_class for x in range(8)]
        batch_slice_info = np.array(batch_slice_info)

        # create the samples based on that class
        samples = self.sample(batch_slice_info)

        # Run a decoder output from the current input images
        feed_dict = {self.image_matrix: self.input_images}
        self.out_mu = self.sess.run(self.decoder_output, feed_dict)

        # Create folders if they do not exist yet
        if not os.path.exists(os.path.join(project_code_root, 'Results/'+ model_name + '_samples/')):
            os.makedirs(os.path.join(project_code_root, 'Results/'+ model_name + '_samples/'))

        model_name = os.path.join(project_code_root, 'Results/'+ model_name + '_samples/')

        # =================================================================================
        # ============== VISUALIZE TRAINING ===============================================
        # =================================================================================

        #for now let's plot intensity
        every_x_time_step = 3
        channel_map = ["intensity","velocity_x","velocity_y","velocity_z"]

        for channel in range(4):
            path= model_name + 'input_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=self.input_images,channel=channel, every_x_time_step=every_x_time_step, out_path=path)

            path = model_name + 'out_mu_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=self.out_mu,channel=channel, every_x_time_step=every_x_time_step, out_path=path)


            path= model_name + 'difference_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=np.abs(self.input_images - self.out_mu),channel=channel, every_x_time_step=every_x_time_step, out_path=path)

            path= model_name + 'generated_' + str(ep) + '_' + str(channel_map[channel]) + '_Class'+ str(cvae_class) + '.png'
            plot_batch_3d(samples,channel=channel, every_x_time_step =every_x_time_step, out_path=path)

            # =================================================================================
            # ============== VISUALIZE TEST ===================================================
            # =================================================================================

            path = model_name + 'test_input_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=self.input_images_test,channel=channel, every_x_time_step=every_x_time_step, out_path=path)

            path = model_name + 'test_out_mu_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=self.out_mu_test,channel=channel, every_x_time_step=every_x_time_step, out_path=path)

            path = model_name + 'test_difference_' + str(ep) + '_' + str(channel_map[channel]) + '.png'
            plot_batch_3d(X=np.abs(self.input_images_test - self.out_mu_test),channel= channel, every_x_time_step = every_x_time_step, out_path=path)

    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))

    def load_from_path(self, path, model_name, step):
        self.saver.restore(self.sess, path + '/' + model_name + ".ckpt-" + str(step))

    def sample(self, batch_slice_info):
        """
        Generate a randomized sample from the model.
        :parameter batch_slice_info: array of length of the batch_size that gives the class to the network
        """
        # z = np.random.normal(0,1,(4,9,7,3,256))
        
        assert len(batch_slice_info) == self.config["batch_size"], "Length of desired condition does not batch size"
        
        z = np.random.normal(0,1,(self.config["batch_size"],
                                  self.config["latent_x"],
                                  self.config["latent_y"],
                                  self.config["latent_t"],384)) # Input is 4 channels + 8 classes = 12 --> 12 x 32 = 384

        feed_dict = {self.guessed_z: z, self.batch_slice_info: batch_slice_info}
        self.samples = self.sess.run(self.decoder_output, feed_dict)
        
        return self.samples
