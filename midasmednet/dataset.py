import logging
import time
import nibabel
import h5py
import zarr
import nibabel as nib
import tracemalloc
from tqdm import tqdm
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from collections import deque, defaultdict
import torch
import zarr
from torch.utils.data import Dataset, DataLoader, IterableDataset


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

class DataReader:

    def read(self, group_key, subj_keys, dtype=True, preload=True):
        pass
    
    def read_data_to_memory(self, subject_keys, group, dtype=np.float16, preload=True):    
        """Reads data from source to memory.
        
        The dataset should be stored using the following structure:
        <data_path>/<group>/<key>... 
        A generator function (data_generator) can be defined to read data respecting this
        structure (implementations for hdf5/zarr/nifti directory are available).

        Args:
            subject_keys (list): identifying keys
            group (str): data group name
            dtype (type, optional): store dtype (default np.float16/np.uint8). Defaults to np.float16.
            preload (bool, optional): if False, data will be loaded on the fly. Defaults to True.
        
        Returns
            object: collections.deque list containing the dataset
        """
        logger = logging.getLogger(__name__)
        logger.info(f'loading group [{group}]...')
        # check timing and memory allocation
        t = time.perf_counter()
        tracemalloc.start()
        data = deque(self.read(subject_keys, group, dtype, preload))
        current, peak = tracemalloc.get_traced_memory()
        logger.debug(f'finished: {time.perf_counter() - t :.3f} s, current memory usage {current / 10**9: .2f}GB, peak memory usage {peak / 10**9:.2f}GB')
        return data
    
    def get_data_shape(self, subject_keys, group):
        pass

    def get_data_attribute(self, subject_keys, group, attribute):
        pass

    def close(self):
        pass

class DataReaderHDF5(DataReader):

    def __init__(self, path_data):
        self.path_data = path_data
        self.hf = h5py.File(str(path_data), 'r')
        self.logger = logging.getLogger(__name__)

    def read(self, subject_keys, group, dtype=np.float16, preload=True):
        for k in tqdm(subject_keys):
            data = self.hf[f'{group}/{k}']
            if preload:
                data = data[:].astype(dtype)
            yield data

    def get_data_shape(self, subject_keys, group):
        shapes = {}
        for k in subject_keys:
            shapes[k] = self.hf[f'{group}/{k}'].shape
        return shapes

    def get_data_attribute(self, subject_keys, group, attribute):
        attr = {}
        for k in subject_keys:
            attr[k] = self.hf[f'{group}/{k}'].attrs[attribute]
        return attr

    def close(self):
        self.hf.close()

class DataReaderZarr(DataReader):
    
    def __init__(self, path_data):
        self.path_data = path_data
        self.zf = zarr.open(str(path_data), 'r')
        self.logger = logging.getLogger(__name__)

    def read(self, subject_keys, group, dtype=np.float16, preload=True):
        self.logger.debug(f'preloading to memory: {preload} ...')
        for k in tqdm(subj_keys):
            data = self.zf[f'{group_key}/{k}']
            if preload:
                data = data[:].astype(dtype)
            yield data

    def get_data_shape(self, subject_keys, group):
        shapes = {}
        for k in subject_keys:
            shapes[k] = self.zf[f'{group}/{k}'].shape
        return shapes

    def get_data_attribute(self, subject_keys, group, attribute):
        attr = {}
        for k in subject_keys:
            attr[k] = self.hf[f'{group}/{k}'].attrs[attribute]
        return attr

    def close(self):
        self.zf.close()


class MedDataset(Dataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 samples_per_subject,
                 patch_size,
                 image_group ='images',
                 label_group ='labels',
                 heatmap_group=None,
                 ReaderClass=DataReaderHDF5,
                 class_probabilities=None,
                 preload=True,
                 transform=None):
        """Creates patched Dataset object.

        Args:
            data_path (str): Path to hdf5/zarr data file.
            subject_keys (list): List of selected subjects keys.
            samples_per_subject (int): Number of patches per subject.
            patch_size (int, int, int): Shape of the patches.
            image_group (str, optional): Group name, image data. Defaults to 'images'.
            label_group (str, optional): Group name, label data. Defaults to 'labels'.
            heatmap_group (str, optional): Group name, heatmap data. Default to None.
            ReaderClass (DataReader, optional): Data loading class. Defaults to DataReaderHDF5.
            class_probabilities (list, optional): Probality per class, that a patch contains at least
                                                  one voxel of this class. Defaults to None.
            preload (bool, optional): Preload data to memory. Defaults to True.
            transform (object, optional): (Patchwise) transformation. Defaults to None.
        """
     
        # set member variables
        self.data_path = data_path
        self.transform = transform
        self.subject_keys = subject_keys
        self.samples_per_subject = samples_per_subject
        self.patch_size = np.array(patch_size, dtype=np.int)
        self.class_probabilities = class_probabilities
        self.heatmap_group = heatmap_group
        self.logger = logging.getLogger(__name__)

        # normalize class probabilities
        if class_probabilities:
            self.class_probabilities = class_probabilities / \
                np.sum(class_probabilities)

        # TODO channel selection
        # load images and labels from data file
        reader = ReaderClass(data_path)
        self.images = reader.read_data_to_memory(subject_keys, image_group, dtype=np.float16, preload=preload)
        self.labels = reader.read_data_to_memory(subject_keys, label_group, dtype=np.uint8, preload=preload)
        if self.heatmap_group:
            self.heatmaps = reader.read_data_to_memory(subject_keys, heatmap_group, dtype=np.uint8, preload=preload)
        reader.close()

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

    def __len__(self):
        return len(self.images)*self.samples_per_subject
 
    def __getitem__(self, idx):
        # create multiple patches for each subject
        idx = idx % len(self.images)

        # load data from container
        imgs = self.images[idx]
        lbls = self.labels[idx]
        if self.heatmap_group:
            hmaps = self.heatmaps[idx]

        # if a class probabilty list is defined, choose a random
        # class value and sample a position inside this label class
        pos = None
        selected_class = 0
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
        index_ini, index_fin = get_random_patch_indices(self.patch_size, imgs.shape[1:],pos=pos)

        # crop labels and images
        cropped_imgs = imgs[:, index_ini[0]:index_fin[0],
                               index_ini[1]:index_fin[1],
                               index_ini[2]:index_fin[2]].astype(np.float32)

        cropped_lbls = lbls[:, index_ini[0]:index_fin[0],
                               index_ini[1]:index_fin[1],
                               index_ini[2]:index_fin[2]].astype(np.uint8)

        # if heatmap_group is specified, crop and append heatmaps
        # the class value encoded label map stays always the last channel
        if self.heatmap_group:
            cropped_hmaps = hmaps[:, index_ini[0]:index_fin[0],
                                     index_ini[1]:index_fin[1],
                                     index_ini[2]:index_fin[2]].astype(np.uint8)

            cropped_lbls = np.concatenate([cropped_hmaps,
                                        cropped_lbls], axis=0)

        patch = {'subject_key': self.subject_keys[idx],
                 'patch_position': index_ini,
                 'selected_class': selected_class,
                 'data': cropped_imgs[np.newaxis, ...],
                 'label': cropped_lbls[np.newaxis, ...]}

        # additional batch dimension for tranform functions
        # format: B,C,H,W,D
        if self.transform:
            patch = self.transform(**patch)

        # remove batch dimension
        patch['data'] = np.squeeze(patch['data'], axis=0) 
        patch['label'] = np.squeeze(patch['label'], axis=0)
        return patch

    
def grid_patch_generator(img, patch_size, patch_overlap, **kwargs):
    """Generates grid of overlapping patches.

    All patches are overlapping (2*patch_overlap per axis).
    Cropping the original image by patch_overlap.
    The resulting patches can be re-assembled to the 
    original image shape.
    
    Additional np.pad argument can be passed via **kwargs.

    Args:
        img (np.array): CxHxWxD 
        patch_size (list/np.array): patch shape [H,W,D]
        patch_overlap (list/np.array): overlap (per axis) [H,W,D]
    
    Yields:
        np.array, np.array, int: patch data CxHxWxD, 
                                 patch position [H,W,D], 
                                 patch number
    """
    dim = 3
    patch_size = np.array(patch_size)
    img_size = np.array(img.shape[1:])
    patch_overlap = np.array(patch_overlap)
    cropped_patch_size = patch_size - 2*patch_overlap
    n_patches = np.ceil(img_size/cropped_patch_size).astype(int)
    overhead = cropped_patch_size - img_size % cropped_patch_size
    padded_img = np.pad(img, [[0,0],
                              [patch_overlap[0], patch_overlap[0] + overhead[0]],
                              [patch_overlap[1], patch_overlap[1] + overhead[1]],
                              [patch_overlap[2], patch_overlap[2] + overhead[2]]], **kwargs)
    pos = [np.arange(0, n_patches[k])*cropped_patch_size[k] for k in range(dim)]
    count = -1
    for p0 in pos[0]:
        for p1 in pos[1]:
            for p2 in pos[2]:
                idx = np.array([p0, p1, p2])
                idx_end = idx + patch_size
                count += 1
                patch = padded_img[:, idx[0]:idx_end[0], idx[1]:idx_end[1], idx[2]:idx_end[2]]
                yield patch, idx, count

class GridPatchSampler(IterableDataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 patch_size, patch_overlap,
                 out_channels=1,
                 out_dtype=np.uint8,
                 channel_selection=None,
                 image_group='images',
                 ReaderClass=DataReaderHDF5,
                 pad_args={'mode': 'symmetric'}):
        """GridPatchSampler for patch based inference.
        
        Creates IterableDataset of overlapping patches (overlap between neighboring
        patches: 2*patch_overlapping). 
        To assemble the original image shape use add_processed_batch(). The 
        number of channels for the assembled images (corresponding to the 
        channels of the processed patches) has to be defined by num_channels: 
        <num_channels>xHxWxD.

        Args:
            data_path (Path/str): data path (e.g. zarr/hdf5 file)
            subject_keys (list): subject keys
            patch_size (list/np.array): [H,W,D] patch shape
            patch_overlap (list/np.array): [H,W,D] patch boundary
            out_channels (int, optional): number of channels for the processed patches. Defaults to 1.
            out_dtype (dtype, optional): data type of processed patches. Defaults to np.uint8.  
            channel_selection (dtype, optional): use only specified channels. Defaults to None.
            image_group (str, optional): image group tag . Defaults to 'images'.
            ReaderClass (function, optional): data reader class. Defaults to DataReaderHDF5.
            pad_args (dict, optional): additional np.pad parameters. Defaults to {'mode': 'symmetric'}.
        """
        self.data_path = str(data_path)
        self.subject_keys = subject_keys
        self.patch_size = np.array(patch_size)
        self.patch_overlap = patch_overlap
        self.image_group = image_group
        self.ReaderClass = ReaderClass
        self.out_channels = out_channels
        self.channel_selection = channel_selection
        self.out_dtype = out_dtype
        self.results = zarr.group()
        self.originals = {}
        self.pad_args = pad_args

        # read image data for each subject in subject_keys
        reader = self.ReaderClass(self.data_path)
        self.data_shape = reader.get_data_shape(self.subject_keys, self.image_group)
        self.data_affine = reader.get_data_attribute(self.subject_keys, self.image_group, "affine")
        self.data_generator = reader.read_data_to_memory(self.subject_keys, self.image_group, dtype=np.float16)
        reader.close()
    
    def add_processed_batch(self, sample):
        """Assembles the processed patches to the original array shape.
        
        Args:
            sample (dict): 'subject_key', 'pos', 'data' (C,H,W,D) for each patch  
        """
        for i, key in enumerate(sample['subject_key']):
            # crop patch overlap
            cropped_patch = np.array(sample['data'][i, :,
                                                    self.patch_overlap[0]:-self.patch_overlap[1],
                                                    self.patch_overlap[1]:-self.patch_overlap[1],
                                                    self.patch_overlap[2]:-self.patch_overlap[2]])
            # start and end position
            pos = np.array(sample['pos'][i])
            pos_end = np.array(pos + np.array(cropped_patch.shape[1:]))
            # check if end position is outside the original array (due to padding)
            # -> crop again (overhead)
            img_size = np.array(self.data_shape[key][1:])
            crop_pos_end = np.minimum(pos_end, img_size)
            overhead = np.maximum(pos_end - crop_pos_end, [0, 0, 0])
            new_patch_size = np.array(cropped_patch.shape[1:]) - overhead
            # add the patch to the corresponing entry in the result container
            ds_shape = np.array(self.data_shape[key])
            ds_shape[0] = self.out_channels
            ds = self.results.require_dataset(key, shape=ds_shape, dtype=self.out_dtype, chunks=False)
            ds.attrs["affine"] = np.array(self.data_affine[key]).tolist()
            ds[:, pos[0]:pos_end[0],
                  pos[1]:pos_end[1],
                  pos[2]:pos_end[2]] = cropped_patch[:, :new_patch_size[0],
                                                        :new_patch_size[1],
                                                        :new_patch_size[2]].astype(self.out_dtype)

    def get_assembled_data(self):
        """Gets the dictionary with assembled/processed images.
        
        Returns:
            dict: Dictionary containing the processed and assembled images (key=subject_key)
        """
        return self.results

    def grid_patch_sampler(self):
        """Data reading and patch generation.
        
        Yields:
            dict: patch dictionary (subject_key, position, count and data)
        """
        
        # create a patch iterator 
        for subj_idx, sample in enumerate(tqdm(self.data_generator)):
            subject_key = self.subject_keys[subj_idx]
            # create patches
            result_shape = np.array(sample.shape)
            result_shape[0] = self.out_channels
            patch_generator = grid_patch_generator(
                sample, self.patch_size, self.patch_overlap, **self.pad_args)
            for patch, idx, count in patch_generator:
                patch_dict = {'data': patch[self.channel_selection, :, :, :],
                              'subject_key': subject_key,
                              'pos': idx,
                              'count': count}
                yield patch_dict

    def __iter__(self):
        return iter(self.grid_patch_sampler())

    def __len__(self):
        return 1


def test_MedDataset():
    train_ds = MedDataset('/mnt/qdata/raheppt1/data/vessel/interim/mra_train.h5',
                    200*['100000'],
                    1,
                    [96, 96, 96],
                    image_group ='images',
                    label_group ='labels',
                    #heatmap_group='heatmaps',
                    ReaderClass=DataReaderHDF5,
                    preload=True)
    sample = train_ds[0]["data"]

# TODO pytest
# TODO test zarr
import time
def test_GridPatchSampler():
    data_path = '/mnt/qdata/raheppt1/data/vessel/interim/mra_train.h5'
    subject_keys = 10*['100000']
    patch_size = [96, 96, 96]
    patch_overlap = [32, 32, 32]
    batch_size = 32
    num_workers = 0

    patch_dataset = GridPatchSampler(
                    data_path,
                    subject_keys,
                    patch_size,
                    patch_overlap,
                    out_channels=1,
                    out_dtype=np.uint8,
                    image_group='images',
                    ReaderClass=DataReaderHDF5,
                    pad_args={'mode': 'symmetric'})


    patch_loader = DataLoader(patch_dataset, batch_size=batch_size, num_workers=num_workers)
    
    t0 = time.perf_counter()
    timing = []
    for sample in patch_loader:
        patch_dataset.add_processed_batch(sample)
        t = time.perf_counter()
        timing.append(t-t0)
        t0 = t
    print(f'{np.mean(timing):.5f} s')
    print(patch_dataset.data_shape['100000'])
    print(patch_dataset.get_assembled_data()['100000'])
    #TODO plot it

def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    test_GridPatchSampler()

if __name__ == "__main__":
    main()