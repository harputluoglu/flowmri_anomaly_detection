
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

def crop_aorta(image_input_folder,
               output_file,
               size,
               target_resolution,
               split_path,
               nr_augmented,
               noise_mean_dist,
               clip_range):

    """
    This function crops out the Aorta using a geometric transform and stacks it together
    to form a 'straightened' version of it as input to the models.

    :param input_folder: Folder where the raw data (numpy arrays) is located
    :param output_file: Folder where the proprocessed data should be written to
    :param size: Size of the output volumes in voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param noise_mean_dist: mean distance of noisy point to ground truth, mm
    :param clip_range: The intensity range to clip the image at
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    """





    sitk_image = sitk.ReadImage(file)




    return 0

def prepare_data(image_input_folder,
                 table_input_folder,
                 output_file,
                 size,
                 target_resolution,
                 split_path,
                 nr_augmented,
                 noise_mean_dist,
                 clip_range):

    '''
    Main function that prepares a dataset from the raw Nifti data to an hdf5 dataset
    '''

    assert len(size) == 3, 'Inadequate number of size parameters'
    assert len(target_resolution) == 3, 'Inadequate number of target resolution parameters'

    hdf5_file = h5py.File(output_file, "w")

    file_list = []
    coordinates_list = []
    patient_id_list = []

    logging.info('Counting files and parsing meta data...')

    # complete list of patients considered in this study
    complete_pat_ids = [line.split('\n')[0] for line in open(os.path.join(split_path, 'all_pat_ids.txt'))]

    for patient_id in os.listdir(image_input_folder):

        if patient_id not in complete_pat_ids:
            continue

        folder_path = os.path.join(image_input_folder, patient_id)

        txt_filenames = glob.glob(os.path.join(folder_path, 'img_?_gt_aorta_points.txt'))
        if len(txt_filenames) > 1:
            logging.info('Patient {} has several txt files, I choose the first.'.format(patient_id))

        lm_coord = []
        for line in open(txt_filenames[0]):
            lm_coord.append(np.array(line.split(' ')).astype(float))

        assert len(lm_coord) == 5, 'Patient {} does not have 5 landmarks in txt file.'.format(patient_id)

        img_nr = txt_filenames[0].split('/')[-1][4]
        file = os.path.join(folder_path, 'img_{}.nii.gz'.format(img_nr))

        patient_id_list.append(int(patient_id))
        coordinates_list.append(np.asarray(lm_coord))
        file_list.append(file)

    assert len(patient_id_list) == len(complete_pat_ids), 'Some patients could not be found in the Nifti root directory.'

    # TODO: enable reading of password protected table
    table_filenames = glob.glob(os.path.join(table_input_folder, 'ZurichTAVIRegistry_*_final.xlsx'))
    assert len(table_filenames) == 1, 'There is not one unique excel file in the table directory.'

    registry_table = pd.read_excel(table_filenames[0])
    # rename some columns
    map_dict = {'Body mass index (BMI)': 'BMI',
                'Body surface area (BSA, KOF)': 'BSA',
                'LFEF, %': 'LVEF',
                'Mean transaortic pressure gradient, mm Hg': 'Mean transaortic pressure gradient',
                'Clinically relevant coronary artery disease ': 'Clinically relevant coronary artery disease',
                'AVA, mm2': 'AVA',
                'AVAi, mm2/m2': 'AVAi',
                'Sphericity of LVOT, %': 'Sphericity of LVOT',
                'Size of the prostesis': 'Size of the prosthesis',
                'Survival during follow up, days': 'Survival during follow up'}
    registry_table = registry_table.rename(columns=map_dict)
    registry_table.set_index('Patient Number', inplace=True)
    registry_table = registry_table.reindex(patient_id_list)
    registry_table = registry_table.replace(['na', 'an'], np.nan)

    features_registry = registry_table[TABLE_FEATURES]
    features_categorical = convert_to_categorical(features_registry)
    features_numeric = convert_to_numerical(features_categorical, features_registry)

    labels_registry = registry_table[TABLE_LABELS]
    labels_categorical = convert_to_categorical(labels_registry)
    labels_numeric = convert_to_numerical(labels_categorical, labels_registry)
    labels_numeric['Device success'] = 1.0 - labels_numeric['Device success']

    # Write the small datasets
    hdf5_file.create_dataset('patient_id', data=np.asarray(patient_id_list), dtype=np.uint16)
    labels_grp = hdf5_file.create_group('labels')
    features_grp = hdf5_file.create_group('features')
    for col in features_numeric.columns:
        features_grp.create_dataset(col.replace(' ', '_').replace('-', '_'), data=features_numeric[col], dtype=np.float32)
    # note: dtype is np.float32 here, because np.uint8 saves nans as 0 (known bug)
    for col in labels_numeric.columns:
        labels_grp.create_dataset(col.replace(' ', '_').replace('-', '_'), data=labels_numeric[col], dtype=np.float32)

    # Create dataset for images
    # images_grp = hdf5_file.create_group('images')
    # for i in range(nr_augmented):
    #     images_grp.create_dataset('aug_v{}'.format(i), [len(file_list)] + list(size), dtype=np.float32)
    images_data = hdf5_file.create_dataset('images', [nr_augmented * len(file_list)] + list(size), dtype=np.float32)

    img_list = []

    logging.info('Parsing image files')

    write_buffer = 0
    counter_from = 0

    # Spline parametrization
    params = [i / (size[2] - 1) for i in range(size[2])]

    # augment images by perturbing labels with noise
    noise_std = noise_mean_dist * math.sqrt(math.pi / 2)  # because we have a folded gaussian

    for coord, file in zip(coordinates_list, file_list):

        logging.info('-----------------------------------------------------------')
        logging.info('Doing: %s' % file)

        sitk_image = sitk.ReadImage(file)
        sitk_image = utils.clip_and_rescale_sitk_image(sitk_image,
                                                       clip_range)
        sitk_image, size_factor = utils.resample_sitk_image(sitk_image,
                                                            target_resolution,
                                                            fill_value=0,
                                                            interpolator='linear',
                                                            return_factor=True)
        scaled_coord = np.round(size_factor * coord)

        for ind in range(nr_augmented):
            if ind == 0:  # first image is ground truth
                noisy_coord = scaled_coord
            else:  # introduce noise
                noise_abs = noise_std * np.random.randn(*scaled_coord.shape) / np.array(target_resolution)[np.newaxis, :]
                random_vecs = random_three_vector(nr=scaled_coord.shape[0])
                noisy_coord = scaled_coord + noise_abs * random_vecs

            tck, _ = interpolate.splprep(np.swapaxes(noisy_coord, 0, 1), k=3, s=100)

            # derivative is tangent to the curve
            points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
            Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
            direc = np.array(sitk_image.GetDirection()[3:6])

            slices = []
            for i in range(len(Zs)):
                # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
                xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
                sitk_slice = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
                np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
                slices.append(np_image)
            # stick slices together
            img_list.append(np.concatenate(slices, axis=2))

        write_buffer += 1

        if write_buffer >= MAX_WRITE_BUFFER:

            counter_to = counter_from + write_buffer
            _write_range_to_hdf5(images_data, img_list, counter_from, counter_to, nr_augmented)
            _release_tmp_memory(img_list)

            # reset stuff for next iteration
            counter_from = counter_to
            write_buffer = 0

    # after file loop: Write the remaining data

    logging.info('Writing remaining data')
    counter_to = counter_from + write_buffer

    _write_range_to_hdf5(images_data, img_list, counter_from, counter_to, nr_augmented)
    _release_tmp_memory(img_list)

    hdf5_file.close()

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



# ========= Import patient 14 data =========

# Load numpy image: [x,y,z,t, channel (intensity, vx, vy,vz) ]
im = np.load('../../experiments/sample_data/image.npy')

# plt.imshow(im[:,:,15,7,3], cmap='gray')
normal = normalize_image(im)
plt.imshow(normal[:,:,15,7,3], cmap='gray')
plt.colorbar()

for timestep in range(25):

    plt.figure(figsize=[25,5])
    plot1 = plt.subplot(1,4,1)
    plt.imshow(normal[:,:,15,timestep,1], cmap='gray')
    plt.colorbar()
    plot1.set_title("VX"+str(timestep))

    plot2 = plt.subplot(1,4,2)
    plt.imshow(normal[:,:,15,timestep,2], cmap='gray')
    plt.colorbar()
    plot2.set_title("VY"+str(timestep))

    plot3 = plt.subplot(1,4,3)
    plt.imshow(normal[:,:,15,timestep,3], cmap='gray')
    plt.colorbar()
    plot3.set_title("VZ"+str(timestep))

    plot3 = plt.subplot(1,4,4)
    plt.imshow(np.linalg.norm(normal[:,:,15,timestep,:], axis=-1), cmap='gray')
    plt.colorbar()
    plot3.set_title("Norm of Velocities"+str(timestep))

    plt.savefig('Test/Image_{:02d}'.format(timestep))
    plt.close()


import imageio

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I', duration=0.1) as writer:
  filenames = glob.glob('Test/Image_*.png')
  filenames = sorted(filenames)
  for i,filename in enumerate(filenames):
    print(filename)
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


velocity_arrays = np.array(im[...,1:4])
im = np.linalg.norm(velocity_arrays, axis=-1)

plt.imshow(im[:,:,15,15], cmap='Purples')

sitk_image = sitk.GetImageFromArray(im[:,:,:,20])

sitk_image.GetSize()
sitk_image.GetSpacing()
sitk_image.GetDirection()
sitk_image.GetDimension()



# ========= Import patient 14 segmentation ========
def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


# Load an example segemtation
segmented = np.load('../../experiments/segmented_data/14/10002CA3/10002CA4/random_walker_prediction.npy')


masked_intensity = segmented * normal[:,:,:,:,0]
masked_vx = segmented * normal[:,:,:,:,1]
masked_vy = segmented * normal[:,:,:,:,2]
masked_vz = segmented * normal[:,:,:,:,3]

masked = np.stack([masked_intensity,masked_vx,masked_vy,masked_vz],axis=4)

plt.imshow(segmented[:,:,15,12], cmap='gray')

enlarged_mask = binary_dilation(segmented)

masked_intensity = enlarged_mask * normal[:,:,:,:,0]
masked_vx = enlarged_mask * normal[:,:,:,:,1]
masked_vy = enlarged_mask * normal[:,:,:,:,2]
masked_vz = enlarged_mask * normal[:,:,:,:,3]

bigger_masked = np.stack([masked_intensity,masked_vx,masked_vy,masked_vz],axis=4)
(x,y,z) = np.where(masked[:120,:70,:,7,1]<0)


plt.imshow(masked[:,:,15,7,1], cmap='gray')
plt.scatter(y,x)

shape = np.zeros(bigger_masked.shape[0:3])

shape[x,y,z] = 1.

# okay, not sure what I am doing....

plt.imshow(shape[:,:])

skeleton = skeletonize(shape[:,:])

plt.imshow(skeleton[:,:,12])

points = np.array(np.where(skeleton != 0)).transpose([1,0])

plt.imshow(masked[:,:,15,7,1], cmap='gray')
plt.scatter(points[:,1], points[:,0], s=1)





for i in range(len(points)):
    print("Index {}:".format(str(i)) + str(points[i]))

# testing another approach here...

for i in range(segmented.shape[3]):
    skeleton = skeletonize(segmented[:,:,:,i])
    plt.imshow(skeleton[:,:,15], cmap='gray')
    test.append(skeleton)

test2 = np.average(test,axis=1)
plt.imshow(test2[:,:,15], cmap='gray')


# ==============================================================================
# ==============================================================================
skeleton = skeletonize(segmented[:,:,:,15])
summed = np.sum(skeleton,axis=2)

points = np.array(np.where(skeleton != 0)).transpose([1,0])

n= 100
points = points[np.random.choice(points.shape[0], n, replace=False)]

plt.imshow(summed[:,:])

samples = []

for index, point in enumerate(points[0:]):
    if index == 0:
        continue
    if index % 10 == 0:
        samples.append(point)

samples = np.array(samples)
min = samples[np.argmin(samples[:,0])]

samples = samples.tolist()
min = min.tolist()

def distance(point1, point2):
    return math.sqrt(sum((a-b)**2 for a,b in zip(point1,point2)))


collected = [min]
remaining = samples
remaining.remove(min)

while len(remaining) != 0:
    min = distance(collected[-1], samples[0])
    min_index = 0
    for i in range(len(remaining)):
        dist = distance(collected[-1], remaining[i])
        print(str(dist))

        if dist < min:
            min = dist
            min_index = i

    print("append and remove")
    collected.append(samples[min_index])
    remaining.remove(samples[min_index])

remaining
collected

collected = np.array(collected)


plt.scatter(collected[:,0],collected[:,1], s=2, c='red', marker='o')
collected
"""
removed = []
for index,point in enumerate(samples):
    if (index != len(samples)-1):
        min = distance(point, samples[index+1])
        min_index = 0
        for idx2, point2 in enumerate(samples[index+1:]):
            dist = distance(point, point2)

            print("POINT: " + str(point2))
            print(str(dist))
            if(dist< min):
                print("New Min")
                min_index = idx2

        removed.append(min_index)






def f(b,l):l.sort(key=lambda p: math.sqrt(sum((a-b)**2 for a,b in zip(b,p))))

samples = samples.tolist()
min = min.tolist()
f(min, samples)

"""
scaled_coord = collected

scaled_coord

size = (30,30,256)
# spline parametrization
params = [i / (size[2] - 1) for i in range(size[2])]

tck, _ = interpolate.splprep(np.swapaxes(scaled_coord, 0, 1), k=3, s=200)

# derivative is tangent to the curve
points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
direc = np.array(sitk_image.GetDirection()[3:6])

points

plt.figure('Centerline')
plt.imshow(im[:,:,15,20], cmap='gray')
plt.scatter(points[:,0],points[:,1], s=2, c='red', marker='o')
plt.savefig('image_with_centerline_velocity.png')
plt.close()


# ==============================================================================
# ===========================================================================================
# x, y, z needs to become z, y, x for SITK
x = points[:,0]
y = points[:,1]
z = points[:,2]


coords = np.array([z,y,x]).transpose([1,0])
coords

sampled_coords = []
for idx, element in enumerate(coords):

    if idx % 15 == 0:
        sampled_coords.append(element)

sitk_image = utils.clip_and_rescale_sitk_image(sitk_image,
                                               (-8000,8000))
sampled_coords

test_coords = [[13,43,100],
               [14,35,80],
               [18,47,52],
               [14,75,57],
               [15,85,95],
               [14,77,120],
               [14,60,150]]

nda = sitk.GetArrayFromImage(sitk_image)
# plt.imshow(nda[:,:,15], cmap='gray')


original_spacing = sitk_image.GetSpacing()
sitk_image, size_factor = utils.resample_sitk_image(sitk_image,
                                                    original_spacing,
                                                    fill_value=0,
                                                    interpolator='linear',
                                                    return_factor=True)

scaled_coord = np.round(size_factor * test_coords)

size = (30,30,256)
# spline parametrization
params = [i / (size[2] - 1) for i in range(size[2])]
"""
# ======== Move to batches to do this since otherwise we cannot fit a proper curve =====
# TODO: Create a batch wise evaluation of this
# ======================================================================================
total_number_of_slices = size[2]
slices= []
batchsize = 8
start = 0
while (start + batchsize <= total_number_of_slices-1):
    params = [i / (batchsize - 1) for i in range(batchsize)]

    tck, _ = interpolate.splprep(np.swapaxes(scaled_coord[start:start+batchsize], 0, 1), k=3, s=1000)

    # derivative is tangent to the curve
    points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)

    plt.figure('SuperFigure')
    plt.imshow(im[:,:,15,20,0], cmap='gray')
    plt.scatter(scaled_coord[start:start+batchsize,1],scaled_coord[start:start+batchsize,2], s=2, c='blue', marker='o')
    plt.scatter(points[:,1],points[:,2], s=2, c='red', marker='o')
    plt.show()
    plt.savefig('image_with_centerline.png')
    plt.close()

    Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
    direc = np.array(sitk_image.GetDirection()[3:6])


    for i in range(len(Zs)):
        # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
        xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
        sitk_slice = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
        np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)

        slices.append(np_image)

    start += batchsize


# stick slices together
img_list = []

img_list.append(np.concatenate(slices, axis=2))

image= img_list[0]

num_slices = image.shape[2]
num_slices
"""



tck, _ = interpolate.splprep(np.swapaxes(scaled_coord, 0, 1), k=3, s=100)

# derivative is tangent to the curve
points = np.swapaxes(interpolate.splev(params, tck, der=0), 0, 1)
Zs = np.swapaxes(interpolate.splev(params, tck, der=1), 0, 1)
direc = np.array(sitk_image.GetDirection()[3:6])

slices = []
for i in range(len(Zs)):
    # I define the x'-vector as the projection of the y-vector onto the plane perpendicular to the spline
    xs = (direc - np.dot(direc, Zs[i]) / (np.power(np.linalg.norm(Zs[i]), 2)) * Zs[i])
    sitk_slice = extract_slice_from_sitk_image(sitk_image, points[i], Zs[i], xs, list(size[:2]) + [1], fill_value=0)
    np_image = sitk.GetArrayFromImage(sitk_slice).transpose(2, 1, 0)
    slices.append(np_image)
# stick slices together
img_list = []
img_list.append(np.concatenate(slices, axis=2))


plt.figure('Centerline')
plt.imshow(im[:,:,15,20], cmap='gray')
plt.scatter(scaled_coord[:,1],scaled_coord[:,2], s=2, c='blue', marker='o')
plt.scatter(points[:,1],points[:,2], s=2, c='red', marker='o')
plt.savefig('image_with_centerline_velocity.png')
plt.close()

num_slices = size[2]
image= img_list[0]
# ================================
# all slices of each time-index as png
# ================================
# %config InlineBackend.figure_format = "retina"
plt.figure(figsize=[120,120])
for j in range(num_slices-1):
    plt.subplot(16, 16 , j+1)
    plt.imshow(image[:,:,j], cmap='gray')
    # plt.xticks([], []); plt.yticks([], [])
    # plt.title('M, s: ' + str(j))

#plt.figure()
#plt.imshow(image[20,:,:], cmap='gray')

# plt.show()
plt.savefig('Velocity.png')
plt.close()





# =======================================================================================================================

def multi_slice_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[3] // 2
    ax.imshow(ax.volume[:,:,ax.index,7,0])
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[3]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[3]
    ax.images[0].set_array(volume[:,:,ax.index,7,0])

multi_slice_viewer(im)
