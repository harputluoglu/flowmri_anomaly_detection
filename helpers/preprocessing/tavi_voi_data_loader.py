# Authors:


import os
import glob
import numpy as np
import logging
import gc
import h5py
import SimpleITK as sitk
import pandas as pd
import math
from scipy import interpolate

import sys
sys.path.append('/home/brdavid/Documents/TAVI_Project/code/discriminative_learning_toolbox-master')
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# Table info
TABLE_FEATURES = ['Age',
                  'Gender',
                  'Weight',
                  'Height',
                  'BMI',
                  'BSA',
                  'EuroSCORE II',
                  'STS Score',
                  'Family history of any cardiovascular disease',
                  'Smoking status',
                  'Dyslipidaemia',
                  'Clinically relevant coronary artery disease',
                  'Coronary artery bypass grafting',
                  'Coronary calcium',
                  'Cerebrovascular disease',
                  'Peripheral artery disease',
                  'Any cardiovascular disease',
                  'proBNP',
                  'LVEF',
                  'Mean transaortic pressure gradient',
                  'AVA',
                  'AVAi',
                  'Aortic regurgitation',
                  'Generation of TAVI device',
                  'Type of TAVI device',
                  'Expanding type',
                  'Size of the prosthesis']
TABLE_LABELS = ['Death during follow up',
                'Type of death',
                'Device success',
                'Early safety endpoint',
                'Agatston Score Aortic Valve',
                'Diameter of ascending aorta',
                'LVOT Area']


def convert_to_categorical(df):
    df_categorical = df.copy()
    for c in df_categorical.columns:
        if df[c].dtypes == 'O':
            df_categorical[c] = df[c].astype('category')
    return df_categorical


def convert_to_numerical(df_categorical, df):
    # changing the columns into numerical.
    # binary categorical variables will be directly mapped to 0 and 1.
    # categorical variables with more categories will be mapped to different binary variables.
    # N categories = N-1 bin vars, as N-1 implies the Nth
    df_numeric = df_categorical.copy()
    for c in df_categorical.columns:
        if df[c].dtypes == 'O' and df_categorical[c].cat.categories.size <= 2:
            # binary categories - life is easy
            df_numeric[c] = df_categorical[c].cat.codes
            df_numeric[c].replace(-1, np.nan, inplace=True)  # cat code of missing value is -1
        elif df[c].dtypes == 'O':
            # we need to create new columns
            ncat = df_categorical[c].cat.categories.size
            cat_data = np.zeros([df_categorical.shape[0], ncat])
            col_names = []
            for cats, n in zip(df_categorical[c].cat.categories, range(ncat)):
                cat_data[df_categorical[c] == cats, n] = 1.0
                col_names = col_names + [c + ' ' + cats]
            df_cat = pd.DataFrame(cat_data, columns=col_names, index=df_categorical.index)
            df_numeric = pd.concat([df_numeric, df_cat], axis=1).drop(c,axis=1)
    return df_numeric


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


def random_three_vector(nr):
    """
    Generate 'nr' random 3D unit vectors (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    vec = np.zeros((nr, 3))
    for l in range(nr):
        phi = np.random.uniform(0, np.pi * 2)
        costheta = np.random.uniform(-1, 1)

        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vec[l] = (x, y, z)
    return vec


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

    # spline parametrization
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


def _write_range_to_hdf5(hdf5_data, img_list, counter_from, counter_to, nr_augmented):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list, dtype=np.float32)

    hdf5_data[nr_augmented * counter_from:nr_augmented * counter_to, ...] = img_arr


def _release_tmp_memory(img_list):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list.clear()
    gc.collect()


def load_and_maybe_process_data(image_input_folder,
                                table_input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                split_path,
                                nr_augmented,
                                noise_mean_dist=5,
                                clip_range=(-200, 800),
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the CT chest landmark data

    :param input_folder: Folder where the raw Nifti images are located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param size: Size of the output volumes in voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param noise_mean_dist: mean distance of noisy point to ground truth, mm
    :param clip_range: The intensity range to clip the image at
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    nraug_str = str(nr_augmented)

    data_file_name = 'voiclass_size_%s_res_%s_nraug_%s.hdf5' % (size_str, res_str, nraug_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(image_input_folder,
                     table_input_folder,
                     data_file_path,
                     size,
                     target_resolution,
                     split_path,
                     nr_augmented,
                     noise_mean_dist,
                     clip_range)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    input_folder = '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/USZ/AorticStenosisProject/Nifti'
    table_root = '/home/brdavid/Documents/TAVI_Project/table'
    preprocessing_folder = '/scratch_net/hopsing/brdavid/discriminative/data'
    split_path = '/home/brdavid/Documents/TAVI_Project/train_test_split'

    d = load_and_maybe_process_data(input_folder, table_root, preprocessing_folder, (64, 64, 64), (0.8, 0.8, 0.8), split_path, 50)

