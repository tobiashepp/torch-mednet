import logging
import time
import nibabel
import h5py
import zarr
import nibabel as nib
import tracemalloc
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from collections import deque
import torch
from torch.utils.data import Dataset, DataLoader


def get_labeled_position(label, class_value, label_any=None):
    """Sample valid idx position inside the specified class.
    
    Sample a position inside the specified class.
    Using pre-computed np.any(label == class_value, axis=2)  
    along third axis makes sampling more efficient. If there
    is no valid position, None is returned.

    Args:
        label (np.array): array with label information H,W,D 
        class_value (int): value of specified class 
        label_any (list): pre-computed np.any(label == class_value, axis=2)  
    
    Returns:
        list: indices of a random valid position inside the given label
    """
    if label_any is None:
        label_any = np.any(label == class_value, axis=2)   
   
    # Are there any positions with label == class_value?
    valid_idx = np.argwhere(label_any==True)
    if valid_idx.size:
        # choose random valid position (2d)
        rnd = np.random.randint(0, valid_idx.shape[0])
        idx = valid_idx[rnd]
        # Sample additional index along the third axis(=2).
        # Voxel value should be equal to the class value.
        valid_idx = label[idx[0], idx[1], :]
        valid_idx = np.argwhere(valid_idx == class_value)[0]
        rnd = np.random.choice(valid_idx)
        idx = [idx[0], idx[1], rnd]
    else:
        idx = None

    return idx


def get_random_patch_indices(patch_size, img_shape, pos=None):
    """ Create random patch indices.
    
    Creates (valid) max./min. corner indices of a patch.
    If a specific position is given, the patch must contain
    this index position. If position is None, a random
    patch will be produced.
    
    Args:
        patch_size (np.array): patch dimensions (H,W,D)
        img_shape  (np.array): shape of the image (H,W,D)
        pos (np.array, optional): specify position (H,W,D), wich should be 
                    included in the sampled patch. Defaults to None.
    
    Returns:
        (np.array, np.array): patch corner indices (e.g. first axis 
                              index_ini[0]:index_fin[0])
    """
        
    # 3d - image array should have shape H,W,D
    # if idx is given, the patch has to surround this position
    if pos:
        pos = np.array(pos, dtype=np.int)
        min_index = np.maximum(pos-patch_size+1, 0)
        max_index = np.minimum(img_shape-patch_size+1, pos+1)
    else:
        min_index = np.array([0, 0, 0])
        max_index = img_shape-patch_size+1

    # create valid patch boundaries
    index_ini = np.random.randint(low=min_index, high=max_index)
    index_fin = index_ini + patch_size

    return index_ini, index_fin


def one_hot_to_label(data,
                     add_background=True):
    """Convert one hot encoded array to 1d-class values array.
    
    Args:
        data (np.array): One-hot encoded array C,H,W,D.
        add_background (bool, optional): Add additional background channel (0). Defaults to True.
    
    Returns:
        np.array: C[0],H,W,D (1-dim) class value array
    """
    if add_background:
        background = np.invert(np.any(data, axis=0, keepdims=True))
        data = np.concatenate([background, data], axis=0)
    data = np.argmax(data, axis=0)
    data = np.expand_dims(data, axis=0)
    return data


def read_nifti(path_data_dir, group_subdir, subj_keys, dtype):
    logger = logging.getLogger(__name__)
    data_dir = Path(path_data_dir)
    data_dir = data_dir.joinpath(group_subdir)
    for k in subj_keys:
        logger.debug(f'loading {group_subdir}/{k}')
        files = list(data_dir.glob(f'**/*{k}*nii*'))
        files.sort()
        yield np.concatenate([np.expand_dims(nib.load(f).get_fdata(), axis=0).astype(dtype) 
                              for f in files], axis=0)


def read_h5(path_h5data, group_key, subj_keys, dtype):
    logger = logging.getLogger(__name__)
    with h5py.File(str(path_h5data), 'r') as hf:
        for k in subj_keys:
            logger.debug(f'loading {group_key}/{k}')
            yield hf[f'{group_key}/{k}'][:].astype(dtype)


def read_zarr(path_zarr, group_key, subj_keys, dtype):
    logger = logging.getLogger(__name__)
    with zarr.open(str(path_zarr), 'r') as zf:
        for k in subj_keys:
            logger.debug(f'loading {group_key}/{k}')
            yield zf[f'{group_key}/{k}'][:].astype(dtype)


def read_data_to_memory(data_path, subject_keys, group, data_generator=read_zarr, dtype=np.float16):    
    logger = logging.getLogger(__name__)
    logger.info('loading data ...')
    # check timing and memory allocation
    t = time.perf_counter()
    tracemalloc.start()
    data = deque(data_generator(data_path, group, subject_keys, dtype))
    current, peak = tracemalloc.get_traced_memory()
    logger.debug(f'finished: {time.perf_counter() - t :.3f} s, current memory usage {current / 10**9: .2f}GB, peak memory usage {peak / 10**9:.2f}GB')
    return data


class MedDataset(Dataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 samples_per_subject,
                 patch_size,
                 class_probabilities=None,
                 transform=None):
     
        # set member variables
        self.data_path = data_path
        self.transform = transform
        self.subject_keys = subject_keys
        self.samples_per_subject = samples_per_subject
        self.patch_size = np.array(patch_size, dtype=np.int)
        self.class_probabilities = class_probabilities

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # normalize class probabilities
        if class_probabilities:
            self.class_probabilities = class_probabilities / \
                np.sum(class_probabilities)

        # data container
        self.images = []
        self.labels = []
        self._label_ax2_any = []

    def __len__(self):
        return len(self.images)*self.samples_per_subject
 
    def __getitem__(self, idx):
        # create multiple patches for each subject
        idx = idx % len(self.images)

        # load data from container
        imgs = self.images[idx]
        lbls = self.labels[idx]

        # if a class probabilty list is defined, choose a random
        # class value and sample a position inside this label class
        pos = None
        selected_class = None
        if self.class_probabilities is not None:
            # select a class to compute a patch position for
            selected_class = np.random.choice(range(len(self.class_probabilities)),
                                            p=self.class_probabilities)

            # get a random point inside the given label
            # if selected_class == 0, use a random position
            if selected_class > 0:
                pos = get_labeled_position(lbls[-1], selected_class, 
                                           label_any=self._label_ax2_any[idx][selected_class])

        # get valid indices of a random patch containing the specified position
        index_ini, index_fin = get_random_patch_indices(self.patch_size,
                                                        imgs.shape[1:],
                                                        pos=pos)

        # crop labels and images
        cropped_imgs = imgs[np.newaxis, :,
                            index_ini[0]:index_fin[0],
                            index_ini[1]:index_fin[1],
                            index_ini[2]:index_fin[2]].astype(np.float32)

        cropped_lbls = lbls[np.newaxis, :,
                             index_ini[0]:index_fin[0],
                             index_ini[1]:index_fin[1],
                             index_ini[2]:index_fin[2]].astype(np.uint8)


        patch = {'subject_key': self.subject_keys[idx],
                 'patch_position': index_ini,
                 'selected_class': selected_class,
                 'data': cropped_imgs,
                 'label': cropped_lbls}

        # additional batch dimension for tranform functions
        # format: B,C,H,W,D
        if self.transform:
            patch = self.transform(patch)

        # remove batch dimension
        patch['data'] = np.squeeze(patch['data'], axis=0) 
        patch['label'] = np.squeeze(patch['label'], axis=0)
        return patch


class SegmentationDataset(MedDataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 samples_per_subject,
                 patch_size,
                 class_probabilities=None,
                 transform=None,
                 data_reader=read_zarr,
                 image_group='images',
                 label_group='labels'):
     
        super().__init__(data_path,
                         subject_keys,
                         samples_per_subject,
                         patch_size,
                         class_probabilities=class_probabilities,
                         transform=transform)

        # load images and labels from data file
        self.images = read_data_to_memory(data_path, subject_keys, image_group, data_reader, dtype=np.float16)
        self.labels = read_data_to_memory(data_path, subject_keys, label_group, data_reader, dtype=np.uint8)

        # check if the number of labels and images are the same
        assert(len(self.images)==len(self.labels))

        # for each label and class value (nested list [idx][class_value]) 
        # compute if any voxel value along axis=2 is equal to the class value
        # this improves the computational effiency of the label sampling 
        # signficantly
        self.logger.info('pre-computing sampling maps ...')
        t = time.perf_counter()
        self._label_ax2_any = []
        if class_probabilities:
            max_class_value =  len(class_probabilities) 
            for idx in range(len(self.labels)):
                self._label_ax2_any.append([np.any(self.labels[idx][-1, ...] == c, axis=2)
                                            for c in range(max_class_value)])
        self.logger.debug(f'finished {time.perf_counter() - t : .3f} s')


class LandmarkDataset(MedDataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 samples_per_subject,
                 patch_size,
                 class_probabilities=None,
                 transform=None,
                 data_reader=read_zarr,
                 verbose=True,
                 heatmap_treshold=30,
                 heatmap_num_workers=4,
                 image_group='images',
                 heatmap_group='heatmaps'):

        
        super().__init__(data_path,
                         subject_keys,
                         samples_per_subject,
                         patch_size,
                         class_probabilities=class_probabilities,
                         transform=transform)
        
        # set parameters
        self.heatmap_treshold = heatmap_treshold
        
        # load images and heatmaps from data file
        self.images = read_data_to_memory(data_path, subject_keys, image_group, data_reader, dtype=np.float16)
        self.labels = read_data_to_memory(data_path, subject_keys, heatmap_group, data_reader, dtype=np.uint8)  

        # threshold heatmap to generate class labels
        self.logger.debug('generating class labels from heatmaps ...')
        
        # combining heatmap and labels to joint array (uint8)
        # class labels will be stored as last (!) channel
        # multithreading version (!)
        def process_heatmap(idx, heatmap):
            class_label = one_hot_to_label(
                 heatmap > self.heatmap_treshold, add_background=True)
            self.labels[idx] = np.concatenate([heatmap, class_label], axis=0)

        t = time.perf_counter()
        with parallel_backend('threading', n_jobs=heatmap_num_workers):
            Parallel()(delayed(process_heatmap)(idx, heatmap) 
                                    for idx, heatmap in enumerate(self.labels))
        self.logger.debug(f'finished {time.perf_counter() - t : .3f} s')

        # for each label and class value (nested list [idx][class_value]) 
        # compute if any voxel value along axis=2 is equal to the class value
        # this improves the computational effiency of the label sampling 
        # signficantly
        self.logger.debug('pre-computing sampling maps ...')
        t = time.perf_counter()
        self._label_ax2_any = []
        if class_probabilities:
            max_class_value =  len(class_probabilities) 
            for idx in range(len(self.labels)):
                self._label_ax2_any.append([np.any(self.labels[idx][-1, ...] == c, axis=2)
                                            for c in range(max_class_value)])
        self.logger.debug(f'finished {time.perf_counter() - t : .3f} s')



