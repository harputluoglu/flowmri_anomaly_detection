from dataset_processing import data_freiburg_numpy_to_preprocessed_hdf5, data_freiburg_numpy_to_hdf5



# ===============================================================
# Main function that runs if this file is run directly
# ===============================================================
if __name__ == '__main__':

    # ==========================================
    # The original dicom images have been saved as numpy arrays at this basepath
    # ==========================================
    project_code_root = '/scratch_net/biwidl210/peifferp/thesis/master-thesis'
    project_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/processed_data'


    # data_train = data_freiburg_numpy_to_hdf5.load_data(basepath = project_data_root, idx_start = 0, idx_end = 19, train_test='train')
    # data_val = data_freiburg_numpy_to_hdf5.load_data(basepath = project_data_root, idx_start = 20, idx_end = 24, train_test='validation')
    # data_test = data_freiburg_numpy_to_hdf5.load_data(basepath = project_data_root, idx_start = 25, idx_end = 28, train_test='test')

    test = data_freiburg_numpy_to_preprocessed_hdf5.load_masked_data_sliced(project_data_root, 0, 19, 'train')
