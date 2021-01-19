import os
import h5py
import numpy as np
from scipy.misc import imresize
from pdb import set_trace as bp

def resize(img, shape, mode):
    i_ = imresize(img, shape, interp=mode)
    i_re = (i_/255.0)*(img.max()-img.min())+img.min()
    return np.asarray(i_re)


class IndexFlowCamCAN(object):
    """Batches from index file."""
    def __init__(
            self,
            shape,
            index_path,
            train,
            box_factor,
            fill_batches = True,
            shuffle = True,
            return_keys=["imgs", "mask", "norm_imgs", "norm_mask"]):
        self.shape = shape
        self.batch_size = self.shape[0]
        self.img_shape = self.shape[1:]
        self.box_factor = box_factor
        self.nimg_shape = self.img_shape[0]//(2**self.box_factor)
        # with open(index_path, "rb") as f:
        #     self.index = pickle.load(f)
        self.basepath = os.path.dirname(index_path)
        self.train = train
        self.fill_batches = fill_batches
        self.shuffle_ = shuffle
        # self.return_keys = return_keys
        #data_file = h5py(self.basepath)
        #bp()
        self.data_file = h5py.File(index_path, 'r')
        self.return_keys = return_keys
        #self.return_keys = [key for key in self.data_file.keys()]

        # rescale joint coordinates to image shape
        # h,w = self.img_shape[:2]
        # wh = np.array([[[w,h]]])
        # # self.index["joints"] = self.index["joints"] * wh
        #bp()
        self.indices = np.array(
                [i for i in range(self.data_file['Scan'].shape[0])])

        self.n = self.indices.shape[0]
        self.shuffle()
    # def _filter(self, i):
    #     good = True
    #     good = good and (self.index["train"][i] == self.train)
    #     joints = self.index["joints"][i]
    #     required_joints = ["lshoulder","rshoulder","lhip","rhip"]
    #     joint_indices = [self.jo.index(b) for b in required_joints]
    #     joints = np.float32(joints[joint_indices])
    #     good = good and valid_joints(joints)
    #     return good


    def __next__(self):
        batch = dict()

        # get indices for batch
        batch_start, batch_end = self.batch_start, self.batch_start + self.batch_size
        batch_indices = self.indices[batch_start:batch_end]
        if self.fill_batches and batch_indices.shape[0] != self.batch_size:
            n_missing = self.batch_size - batch_indices.shape[0]
            batch_indices = np.concatenate([batch_indices, self.indices[:n_missing]], axis = 0)
            assert batch_indices.shape[0] == self.batch_size
        batch_indices = np.array(batch_indices)
        batch["indices"] = batch_indices

        # prepare next batch
        if batch_end >= self.n:
            self.shuffle()
        else:
            self.batch_start = batch_end

        # prepare batch data
        # load images
        batch["imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan'][i].reshape(200,200)

            #scan = imresize(scan, self.img_shape, "bilinear")
            scan = resize(scan, self.img_shape, "bilinear")
            scan = scan[:, :, np.newaxis]
            # bp()
            #relpath = self.index["imgs"][i]
            #path = os.path.join(self.basepath, relpath)
            #batch["imgs"].append(load_img(path, target_size = self.img_shape))
            batch["imgs"].append(scan)
        batch["imgs"] = np.stack(batch["imgs"])
        #batch["imgs"] = preprocess(batch["imgs"])

        # load masks
        batch["mask"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask'][i].reshape(200,200)
            #mask = imresize(mask, self.img_shape, 'nearest')
            mask = resize(mask, self.img_shape, 'nearest')
            mask = mask[:, :, np.newaxis]
            batch["mask"].append(mask)
        batch["mask"] = np.stack(batch["mask"])
        #batch["imgs"] = preprocess(batch["imgs"])

        # load joint coordinates
        #batch["joints_coordinates"] = [self.index["joints"][i] for i in batch_indices]

        # generate stickmen images from coordinates
        #batch["joints"] = list()
        # for joints in batch["joints_coordinates"]:
        #     img = make_joint_img(self.img_shape, self.jo, joints)
        #     batch["joints"].append(img)
        # batch["joints"] = np.stack(batch["joints"])
        # batch["joints"] = preprocess(batch["joints"])

        # imgs, joints = normalize(batch["imgs"], batch["joints_coordinates"], batch["joints"], self.jo, self.box_factor)
        batch["norm_imgs"] = list()
        for i in batch_indices:
            scan = self.data_file['Scan'][i].reshape(200,200)
            #scan = imresize(scan+3.5, 2*[self.nimg_shape]+[1], 'bilinear')
            scan = resize(scan, 2*[self.nimg_shape]+[1],'bilinear')
            scan = scan[:, :, np.newaxis]
            batch["norm_imgs"].append(scan)
        batch["norm_imgs"] = np.stack(batch["norm_imgs"])

        batch["norm_mask"] = list()
        for i in batch_indices:
            mask = self.data_file['Mask'][i].reshape(200, 200)
            #mask = imresize(mask+3.5, 2*[self.nimg_shape]+[1], 'nearest')
            mask = resize(mask, 2*[self.nimg_shape]+[1], 'nearest')
            mask = mask[:,:,np.newaxis]
            batch["norm_mask"].append(mask)
        batch["norm_mask"] = np.stack(batch["norm_mask"])
        #
        # batch["norm_imgs"] = batch['imgs']
        # batch["norm_mask"] = batch['mask']
        batch_list = [batch[k] for k in self.return_keys]
        return batch_list
        self.data_file.close()


    def shuffle(self):
        self.batch_start = 0
        if self.shuffle_:
            np.random.shuffle(self.indices)
