
# ========================================================================================
# Imports of needed libraries
# ========================================================================================
import numpy as np
import math

from matplotlib import pyplot as plt
import imageio
import os
import glob
import sys
import shutil


# ========================================================================================
# Create an animated GIF from the velocities of a subject or output
# ========================================================================================
def create_gif_of_velocities(input_data, z_slice, output_folder, output_file_name):
    """
    input_data:        an input from the Freiburg Dataset - (x,y,z,t,channels)
                       Could either be an already preprocessed view where z goes along centerline of the Aorta
                       or an unpreprocessed input image where z is the depth sice into the heart

    z_slice:           the slice that we want to display in z direction
    output_folder:     the folder where the output gif will be saved to
    output_file_name:  the name of the output file, should end with '.gif' - example 'Subject3.gif'
    """

    anim_file = output_folder + '/' + output_file_name
    temp_folder = 'Temp'

    try:
        os.makedir(output_folder + '/' + temp_folder)
    except:
        print("creating temp output failed")

    for timestep in range(48):

        plt.figure(figsize=[25,5])
        plot1 = plt.subplot(1,4,1)
        plt.imshow(input_data[:,:,z_slice,timestep,1], cmap='gray')
        plt.colorbar()
        plot1.set_title("VX"+str(timestep))

        plot2 = plt.subplot(1,4,2)
        plt.imshow(input_data[:,:,z_slice,timestep,2], cmap='gray')
        plt.colorbar()
        plot2.set_title("VY"+str(timestep))

        plot3 = plt.subplot(1,4,3)
        plt.imshow(input_data[:,:,z_slice,timestep,3], cmap='gray')
        plt.colorbar()
        plot3.set_title("VZ"+str(timestep))

        plot3 = plt.subplot(1,4,4)
        plt.imshow(np.linalg.norm(input_data[:,:,z_slice,timestep,:], axis=-1), cmap='gray')
        plt.colorbar()
        plot3.set_title("Norm of Velocities"+str(timestep))

        plt.savefig(output_folder + '/' + temp_folder + '/' + 'Image_{:02d}'.format(timestep))
        plt.close()

    with imageio.get_writer(anim_file, mode='I', duration=0.15) as writer:
      filenames = glob.glob(output_folder + '/' + temp_folder + '/' + 'Image_*.png')
      filenames = sorted(filenames)
      for i,filename in enumerate(filenames):
        print(filename)
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

    print("Creating gif done, now deleting temp folder")

    try:
        shutil.rmtree(output_folder + '/' + temp_folder)
        return 0
    except:
        print("Could not delete temp folder")
