"""
ImageSampler to sample patches which contain foreground voxels.


"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchio.sampler import ImageSampler
from torchio import Image, ImagesDataset, LABEL, INTENSITY


class LabelSampler(ImageSampler):
    """
    This iterable dataset yields patches that contain at least one voxel
    without background. See main() for an example.
    For now, this implementation is not efficient because it uses brute force
    to look for foreground voxels.
    """

    def get_label(self, sample):
        """
        Return random patch class.
        :return:
        """
        return ''

    def extract_patch_generator(self, sample, patch_size):
        while True:
            yield self.extract_patch(sample, patch_size)

    def get_random_indices(self, sample, patch_size, idx=None):
        """
        todo Assert that shape is consistent across modalities (and label)
        todo Check that array shape is >= patch size
        """
        first_image_name = list(sample.keys())[0]
        first_image_array = sample[first_image_name]['data']
        # first_image_array should have shape (1, H, W, D)
        shape = np.array(first_image_array.shape[1:], dtype=np.uint16)
        patch_size = np.array(patch_size, dtype=np.uint16)

        # If idx is given, the patch has to include this index.
        if idx is not None:
            idx = np.array(idx, dtype=np.uint16)
            min_index = np.maximum(idx - patch_size + 1, 0)
            max_index = np.minimum(shape - patch_size + 1, idx + 1)
        else:
            min_index = np.array([0, 0, 0])
            max_index = shape-patch_size+1

        # Create valid patch boundaries.
        index_ini = np.random.randint(low=min_index, high=max_index)
        index_fin = index_ini + patch_size

        return index_ini, index_fin

    def extract_patch(self, sample, patch_size):

        # Choose label name.
        lbl_name = self.get_label(sample)
        # If lbl_name == '' choose random point, otherwise
        # sample valid patch center point inside the label.
        if lbl_name:
            lbl = sample[lbl_name]['data'].numpy()
            # Get indices inside the label.
            valid_idx = np.transpose(np.nonzero(lbl))
            num_valid = valid_idx.shape[0]
            # Sample points which lie inside the specified class.
            # Check if the sampled point is a valid center point
            # of a patch with patch_size.
            rnd = np.random.randint(0, num_valid)
            idx = np.array([valid_idx[rnd][i] for i in range(1, 4)])
            patch_min, patch_max = self.get_random_indices(sample, patch_size, idx=idx)
        else:
            patch_min, patch_max = self.get_random_indices(sample, patch_size, idx=None)

        # Extracts the patch with indices from patch_min:patch_max.
        cropped_sample = self.copy_and_crop(
            sample,
            patch_min,
            patch_max)

        cropped_sample['selected_label'] = lbl_name
        return cropped_sample


class RandomLabelSampler(LabelSampler):

        label_distribution = {'test': 0.5}

        def _label_names(self, sample):
            # Get all labels names from the sample.
            label_names = []
            for key, item in sample.items():
                if type(item) is dict:
                    if 'type' in item:
                        if item['type'] == 'label':
                            label_names.append(key)
            return label_names

        def get_label(self, sample):
            # Get all labels names for the given sample.
            label_names = self._label_names(sample)
            if not label_names:
                return ''

            # Choose random label name.
            lbl_keys = list(self.label_distribution.keys())
            lbl_d = np.array(list(self.label_distribution.values()))
            lbl_name = ''
            # Sample random value in (0,1)
            x = np.random.uniform()
            # Use x to choose label name from list.
            # e.g. lbl_d = [0.2, 0.5]
            # reveals -> label 1 prob 20%, label 2 prob 30%, random 50%
            tmp = x < lbl_d
            if np.any(tmp):
                idx = np.min(np.nonzero(tmp))
                if lbl_keys[idx] in label_names:
                    lbl_name = lbl_keys[idx]

            return lbl_name


def main():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
    path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
    path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
    path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
    path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

    # Define image dataset.
    subjects_images = [Image('img',   str(path_img), INTENSITY),
                       Image('label1', str(path_label_1), LABEL),
                       Image('label2', str(path_label_2), LABEL),
                       Image('label3', str(path_label_3), LABEL)]
    subjects_dataset = ImagesDataset([subjects_images])
    # Get single example from this dataset.
    sample = subjects_dataset[0]

    fig = plt.figure()
    aspect = 0.2
    ax = fig.add_subplot(111)
    ax.imshow(np.mean(subjects_dataset[0]['img']['data'][0, ...].numpy(), axis=1), cmap='gray', aspect=aspect)
    ax.imshow(np.mean(subjects_dataset[0]['label1']['data'][0, ...].numpy(), axis=1), alpha=0.3, aspect=aspect)
    ax.imshow(np.mean(subjects_dataset[0]['label2']['data'][0, ...].numpy(), axis=1), cmap='inferno', alpha=0.3, aspect=aspect)

    # Using the LabelSampler to extract random image patches.
    size = 100
    patch_size = size, size, size//5

    class MySampler(RandomLabelSampler):
        label_distribution = {'label1': 0.3, 'label2': 0.5}
    sampler = MySampler(sample, patch_size)

    for i in range(30):
        patch = sampler.extract_patch(sample, patch_size)
        selected_label = patch['selected_label']
        loc = patch['index_ini']

        if selected_label == 'label1':
            color = (0.3, 0.3, 0.7)
            c = 'blue'
        elif selected_label == 'label2':
            color = (0.7, 0.3, 0.3)
            c = 'red'
        else:
            color = (0.8, 0.8, 0.8)
            c = 'white'

        # plot patch rectangle with center point
        #ax.scatter([loc[2]+patch_size[2]//2], [loc[0]+patch_size[0]//2], c=c)
        rect = plt.Rectangle((loc[2], loc[0]),
                            patch_size[2],
                            patch_size[0], fill=False,
                            edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)
    plt.show()



def test():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
    path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
    path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
    path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
    path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

    # Define image dataset.
    subjects_images = [Image('img', str(path_img), INTENSITY),
                       Image('label1', str(path_label_1), LABEL),
                       Image('label2', str(path_label_2), LABEL),
                       Image('label3', str(path_label_3), LABEL)]
    subjects_dataset = ImagesDataset([subjects_images])

    # Get single example from this dataset.
    sample = subjects_dataset[0]

    # Get random patch from this sample.
    class MySampler(RandomLabelSampler):
        label_distribution = {'label1': 0.3, 'label2': 0.5}
    patch_size = 100, 100, 100
    sampler = MySampler(sample, patch_size)
    patch = sampler.extract_patch(sample, patch_size)



def test2():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
    path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
    path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
    path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
    path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

    # Define image dataset.
    subjects_images = [Image('img', str(path_img), INTENSITY),
                       Image('label1', str(path_label_1), LABEL),
                       Image('label2', str(path_label_2), LABEL),
                       Image('label3', str(path_label_3), LABEL)]
    subjects_dataset = ImagesDataset([subjects_images])

    # Get single example from this dataset.
    sample = subjects_dataset[0]
    print(sample['img']['data'].size())
    # Get random patch from this sample.
    class MySampler(RandomLabelSampler):
        label_distribution = {'label1': 0.3, 'label2': 0.5}
    patch_size = 50, 50, 1
    sampler = MySampler(sample, patch_size)
    patch = sampler.extract_patch(sample, patch_size)
    print(patch['img']['data'].size())

    plt.imshow(patch['img']['data'][0,:,:,0])
    plt.show()

if __name__ == '__main__':
    main()