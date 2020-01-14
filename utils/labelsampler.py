"""
ImageSampler to sample patches which contain foreground voxels.


"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
            while True:
                rnd = np.random.randint(0, num_valid)
                idx = np.array([valid_idx[rnd][i] for i in range(1, 4)])
                # Get patch corners.
                patch_min = np.array([0, 0, 0])
                patch_max = np.array([0, 0, 0])
                # idx defines the center of the patch volume.
                for i in range(3):
                    idx_min = idx[i] - patch_size[i]//2
                    idx_max = idx_min + patch_size[i]
                    patch_min[i] = idx_min
                    patch_max[i] = idx_max
                valid_min = np.all(np.array(patch_min) >= 0)
                valid_max = np.all(np.array(patch_max) <= lbl.shape[1:])
                # If indices are valid stop sampling.
                if valid_min and valid_max:
                    break
        else:
            patch_min, patch_max = self.get_random_indices(sample, patch_size)

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
        ax.scatter([loc[2]+patch_size[2]//2], [loc[0]+patch_size[0]//2], c=c)
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


if __name__ == '__main__':
    main()