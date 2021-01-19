import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY
# ==================================================================

# ==================================================================
# name of the host - used to check if running on cluster or not
# ==================================================================
local_hostnames = ['bmicdl05']

# ==================================================================
# project dirs
# ==================================================================
project_code_root = '/scratch_net/biwidl210/peifferp/thesis/master-thesis/dataset_processing'
project_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/processed_data'
# Note that this is the base direectory where the freiburg images have been saved a numpy arrays.
# The original dicom files are not here and they are not required for any further processing.
# The base path for the original dicom files of the freiburg dataset are here:
orig_data_root = '/scratch_net/biwidl210/peifferp/thesis/freiburg_data/source_data'

# ==================================================================
# log root
# ==================================================================
log_root = os.path.join(project_code_root, 'logdir/v0.1/')
