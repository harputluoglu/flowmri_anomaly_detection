# import module and set paths
# ============================   
import os, sys
import numpy as np
import utils
import imageio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# ==================================================================
# import general modules written by cscs for the hpc-predict project
# ==================================================================
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-39] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import FlowMRI, SegmentedFlowMRI

def norm(x,y,z):
    normed_array = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return normed_array 

# Define basepath and basepath_segmenter according to your local paths
basepath = "/scratch/hharputlu/pollux/"
basepath_segmenter ="/scratch/hharputlu/hpc-predict/data/v1/decrypt/segmenter/cnn_segmenter/hpc_predict/v1/inference/2021-03-30_12-53-16_daint102_volN"

# plot
R_values = [1, 8, 10, 12, 14, 16, 18, 20, 22] # undersampling ratio
for subnum in [4, 5, 6, 7]: # subject number (change this for plotting patient results)
    
    for r in range(len(R_values)):
        
        R = R_values[r]
        print(R)  
            
        imagepath = basepath + 'v' + str(subnum) + '_R' + str(R) + '.h5'
        predlabelpath = basepath_segmenter + str(subnum) + '_R' + str(R) +"/output/seg_cnn.h5"
        truelabelpath = basepath + 'v' + str(subnum) + '_seg_rw.h5'

        flow_mri = FlowMRI.read_hdf5(imagepath)
        print("Read the file from {}:", imagepath)
        image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)
        
        segmented_flow_mri = SegmentedFlowMRI.read_hdf5(predlabelpath)
        pred = segmented_flow_mri.segmentation_prob
        
        segmented_flow_mri = SegmentedFlowMRI.read_hdf5(truelabelpath)
        true = segmented_flow_mri.segmentation_prob
        
        nr = 2
        nc = 3
        
        for idx in range(19):
            print("idx: {}", idx)
            plt.figure(figsize=[4*nc, 4*nr])
            
            zidx = idx
            # across z axis
            plt.subplot(nr, nc, 1);  plt.imshow(norm(image[:,:,zidx,3,1], image[:,:,zidx,3,2], image[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('image_z'+str(zidx)+'_t3');
            plt.subplot(nr, nc, 2);  plt.imshow(pred[:,:,zidx,3], cmap='gray'); plt.axis('off'); plt.title('pred (cnn) z'+str(zidx)+'_t3');
            plt.subplot(nr, nc, 3);  plt.imshow(true[:,:,zidx,3], cmap='gray'); plt.axis('off'); plt.title('true (rw) z'+str(zidx)+'_t3');

            # across t axis
            plt.subplot(nr, nc, 4);  plt.imshow(norm(image[:,:,8,idx,1], image[:,:,8,idx,2], image[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('image_z8_t'+str(idx));
            plt.subplot(nr, nc, 5);  plt.imshow(pred[:,:,8,idx], cmap='gray'); plt.axis('off'); plt.title('pred (cnn) z8_t'+str(idx));
            plt.subplot(nr, nc, 6);  plt.imshow(true[:,:,8,idx], cmap='gray'); plt.axis('off'); plt.title('true (rw) z8_t'+str(idx));
            
            if not os.path.exists(basepath + 'tmp/pngs/'):
                os.makedirs(basepath + 'tmp/pngs/')
                print("Path for image saving: " + basepath + "tmp/pngs/")
            plt.savefig(basepath + 'tmp/pngs/' + str(idx) + '.png')
            plt.close()
            
            
        _gif = []
        for idx in range(19):
            print("creating gif")
            _gif.append(imageio.imread(basepath + 'tmp/pngs/' + str(idx) + '.png'))
        imageio.mimsave(basepath_segmenter + str(subnum) + '_R' + str(R) + "/output/" + 'v' + str(subnum) + '_R' + str(R) +'.gif', _gif, format='GIF', duration=0.25)
