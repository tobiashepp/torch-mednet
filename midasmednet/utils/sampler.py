import numpy as np
from copy import copy
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torchio.utils
from torchio.data.sampler import ImageSampler
from torchio.data.images import Image, ImagesDataset
from torchio import LABEL, INTENSITY


def _get_label_names(sample):
    """
    Extracts all the label keys for a sample.
    :param sample: dict
    :return:
    """

    label_names = []
    for key, item in sample.items():
        if type(item) is dict:
            if 'type' in item:
                if item['type'] == 'label':
                    label_names.append(key)
    return label_names


class LabelSampler(ImageSampler):
    """
    This iterable dataset yields patches that contain at least one voxel
    without background.
    """

    def get_label(self, sample):
        """
        Return random patch class. Overwrite this method
        to select a specific label name. Otherwise '' will
        be returned -> random patch.
        :return:
        """
        return ''

    def extract_patch_generator(self, sample, patch_size):
        while True:
            yield self.extract_patch(sample, patch_size)

    def get_random_indices(self, sample, patch_size, idx=None):
        """
        Creates (valid) max./min. corner indices of a patch.
        If a specific index is given, the patch must surround
        this index.
        todo Assert that shape is consistent across modalities (and label)
        todo Check that array shape is >= patch size
        """
        # first_image_array should have shape (1, H, W, D)
        first_image_name = list(sample.keys())[0]
        first_image_array = sample[first_image_name]['data']
        shape = np.array(first_image_array.shape[1:], dtype=np.int)
        patch_size = np.array(patch_size, dtype=np.int)

        # If idx is given, the patch has to surround this index.
        if idx is not None:
            idx = np.array(idx, dtype=np.int)
            min_index = np.maximum(idx-patch_size+1, 0)
            max_index = np.minimum(shape-patch_size+1, idx+1)
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
        # If lbl_name is '', choose random index, otherwise
        # sample valid point inside the label.
        if lbl_name:
            lbl = sample[lbl_name]['data'].numpy()
            # Get indices inside the label:
            # Sample points which lie inside the specified class.
            # Computing maximum along third axis and sample from
            # 2d-maximum map first.
            lbl_mip = np.max(lbl, axis=3)
            valid_idx = np.transpose(np.nonzero(lbl_mip))
            # Are there any nonzero label values?
            if valid_idx.size:
                rnd = np.random.randint(0, valid_idx.shape[0])
                idx = valid_idx[rnd]
                # Sample additional index along the third axis.
                valid_idx = lbl[idx[0], idx[1], idx[2], :]
                valid_idx = np.transpose(np.nonzero(valid_idx))
                rnd = np.random.randint(0, valid_idx.shape[0])
                idx = [idx[1], idx[2], valid_idx[rnd]]
                # Get patch containing idx.
                patch_min, patch_max = self.get_random_indices(sample, patch_size, idx=idx)
            else:
                # Get random patch.
                patch_min, patch_max = self.get_random_indices(sample, patch_size, idx=None)
        else:
            # Get random patch.
            patch_min, patch_max = self.get_random_indices(sample, patch_size, idx=None)

        # Extracts the patch with indices from patch_min:patch_max.
        # todo new cropping routine
        cropped_sample = self.copy_and_crop(
            sample,
            patch_min,
            patch_max)
        return cropped_sample








        def copy_and_crop(
            self,
            sample: dict,
            index_ini: np.ndarray,
            index_fin: np.ndarray,
        ) -> dict: cropped_sample = {}
        for key, value in sample.items():
            cropped_sample[key] = copy.copy(value)
            if is_image_dict(value):
                sample_image_dict = value
                cropped_image_dict = cropped_sample[key]
                cropped_image_dict[DATA] = crop(
                    sample_image_dict[DATA], index_ini, index_fin)
        # torch doesn't like uint16
        cropped_sample['index_ini'] = index_ini.astype(int)
        return cropped_sample

        def crop(
            image: Union[np.ndarray, torch.Tensor],
            index_ini: np.ndarray,
            index_fin: np.ndarray,
        ) -> Union[np.ndarray, torch.Tensor]:
        i_ini, j_ini, k_ini = index_ini
        i_fin, j_fin, k_fin = index_fin
        return image[..., i_ini:i_fin, j_ini:j_fin, k_ini:k_fin]

class RandomLabelSampler(LabelSampler):

        """
        Define the distribution for the label sampling.
        # example: label_distribution = {'a': 0.3, 'b': 0.4, 'c': 0.7}
        # probability class a: 0.3
        # probability class b: 0.1
        # probability class c: 0.3
        # probability random patch: 0.3
        label_distribution = {'test': 0.5}
        """

        def get_label(self, sample):
            # Get all labels names for the given sample.
            label_names = _get_label_names(sample)
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


class SliceSampler(Dataset):
    # todo add slice min, max
    def __init__(self,
                 subjects_dataset,
                 num_workers=0,
                 shuffle_subjects=False,
                 delete_dim=False,
                 slice_min=None,
                 slice_max=None):

        self.subjects = []
        self.index_list = []
        self.slice_min = None
        self.slice_max = None

        # Load all datasets to memory!!!
        subjects_loader = DataLoader(
            subjects_dataset,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
            shuffle=shuffle_subjects)

        # Iterate over subjects using the dataloader.
        for subject_number, sample in enumerate(tqdm(subjects_loader)):
            # Get sizes for images in sample.
            sample_img_sizes = []
            for key, value in sample.items():
                if torchio.utils.is_image_dict(value):
                    sample_img_sizes.append(value['data'].size())
            # All images should have the same size.
            assert len(set(sample_img_sizes)) == 1
            sample_img_size = sample_img_sizes[0]

            # Extract slices.
            # todo add axis selection
            slices = []
            for slice_number in range(sample_img_size[3]):
                # Create index.
                self.index_list.append((subject_number, slice_number))
                cropped_sample = {}
                # For each image in sample
                for key, value in sample.items():
                    cropped_sample[key] = {}
                    if torchio.utils.is_image_dict(value):
                        # Copy all dict entries and select slice from data field.
                        for k, tmp in value.items():
                            if k == 'data':
                                if delete_dim:
                                    cropped_sample[key]['data'] = copy(
                                        value['data'][:, :, :, slice_number])
                                else:
                                    cropped_sample[key]['data'] = copy(
                                        value['data'][:, :, :, [slice_number]])
                            else:
                                cropped_sample[key][k] = tmp
                # Torch doesn't like uint16.
                cropped_sample['index_ini'] = np.array(
                    [0, 0, slice_number]).astype(int)
                slices.append(cropped_sample)
            self.subjects.append(slices)
            del sample

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        subject_number, slice_number = self.index_list[index]
        if self.slice_min:
            if slice_number < self.slice_min:
                slice_number = self.slice_min
        if self.slice_max:
            if slice_number > self.slice_max:
                slice_number = self.slice_max
        return self.subjects[subject_number][slice_number]


class SliceSelectionSampler(Dataset):
    def __init__(self, subjects_dataset,
                 selected_slices,
                 num_workers=0,
                 shuffle_subjects=False):

        self.selected_slices = selected_slices
        self.subjects = []

        # Load all datasets to memory!!!
        subjects_loader = DataLoader(
            subjects_dataset,
            num_workers=num_workers,
            collate_fn=lambda x: x[0],
            shuffle=shuffle_subjects)

        # Iterate over subjects using the dataloader.
        for sample in tqdm(subjects_loader):
            # Extract selected slices.
            # todo add axis selection
            cropped_sample = {}
            for key, value in sample.items():
                cropped_sample[key] = {}
                if torchio.utils.is_image_dict(value):
                    for k, tmp in value.items():
                        if k == 'data':
                            cropped_sample[key]['data'] = copy(
                                value['data'][:, :, :, self.selected_slices])
                        else:
                            cropped_sample[key][k] = tmp
            self.subjects.append(cropped_sample)
            del sample

    def __getitem__(self, index):
        return self.subjects[index]

    def __len__(self):
        return len(self.subjects)

# def main():
#     train_dataset, eval_dataset = datasets.create_IXI_datasets(eval_split=10)
#     slices_train_ds = SliceSampler(eval_dataset,
#                                    num_workers=15,
#                                    shuffle_subjects=True)

#     slice_train_dl = torch.utils.data.DataLoader(
#         slices_train_ds, batch_size=batch_size, shuffle=True)

#     for sample in slice_train_dl:
#         print(*zip(sample['index_ini'], sample['img']['id']))
#         break
#     return


# if __name__ == '__main__':
#     main()

# def main():
#     # todo new examples
#     work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
#     path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
#     path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
#     path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
#     path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

#     # Define image dataset.
#     subjects_images = [Image('img',   str(path_img), INTENSITY),
#                        Image('label1', str(path_label_1), LABEL),
#                        Image('label2', str(path_label_2), LABEL),
#                        Image('label3', str(path_label_3), LABEL)]
#     subjects_dataset = ImagesDataset([subjects_images], transform=StoreValidLabelIndices())
#     # Get single example from this dataset.
#     sample = subjects_dataset[0]

#     fig = plt.figure()
#     aspect = 0.2
#     ax = fig.add_subplot(111)
#     ax.imshow(np.mean(sample['img']['data'][0, ...].numpy(), axis=1), cmap='gray', aspect=aspect)
#     ax.imshow(np.mean(sample['label1']['data'][0, ...].numpy(), axis=1), alpha=0.3, aspect=aspect)
#     ax.imshow(np.mean(sample['label2']['data'][0, ...].numpy(), axis=1), cmap='inferno', alpha=0.3, aspect=aspect)

#     # Using the LabelSampler to extract random image patches.
#     size = 100
#     patch_size = size, size, size//5

#     class MySampler(RandomLabelSampler):
#         label_distribution = {'label1': 0.3, 'label2': 0.5}
#     sampler = MySampler(sample, patch_size)

#     for i in range(30):
#         print(f'h{i}')
#         patch = sampler.extract_patch(sample, patch_size)
#         selected_label = patch['selected_label']
#         loc = patch['index_ini']

#         if selected_label == 'label1':
#             color = (0.3, 0.3, 0.7)
#             c = 'blue'
#         elif selected_label == 'label2':
#             color = (0.7, 0.3, 0.3)
#             c = 'red'
#         else:
#             color = (0.8, 0.8, 0.8)
#             c = 'white'

#         # plot patch rectangle with center point
#         #ax.scatter([loc[2]+patch_size[2]//2], [loc[0]+patch_size[0]//2], c=c)
#         rect = plt.Rectangle((loc[2], loc[0]),
#                             patch_size[2],
#                             patch_size[0], fill=False,
#                             edgecolor=color, linewidth=2.5)
#         ax.add_patch(rect)
#     plt.show()


# def test():
#     work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
#     path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
#     path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
#     path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
#     path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

#     # Define image dataset.
#     subjects_images = [Image('img', str(path_img), INTENSITY),
#                        Image('label1', str(path_label_1), LABEL),
#                        Image('label2', str(path_label_2), LABEL),
#                        Image('label3', str(path_label_3), LABEL)]
#     subjects_dataset = ImagesDataset([subjects_images])

#     # Get single example from this dataset.
#     sample = subjects_dataset[0]

#     # Get random patch from this sample.
#     class MySampler(RandomLabelSampler):
#         label_distribution = {'label1': 0.3, 'label2': 0.5}
#     patch_size = 100, 100, 100
#     sampler = MySampler(sample, patch_size)
#     patch = sampler.extract_patch(sample, patch_size)


# def test2():
#     work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/0002')
#     path_img = work_dir.joinpath('sequences/002_ct.nii.gz')
#     path_label_1 = work_dir.joinpath('labels/002_liver.nii.gz')
#     path_label_2 = work_dir.joinpath('labels/002_spleen.nii.gz')
#     path_label_3 = work_dir.joinpath('labels/002_spine_filled.nii.gz')

#     # Define image dataset.
#     subjects_images = [Image('img', str(path_img), INTENSITY),
#                        Image('label1', str(path_label_1), LABEL),
#                        Image('label2', str(path_label_2), LABEL),
#                        Image('label3', str(path_label_3), LABEL)]
#     subjects_dataset = ImagesDataset([subjects_images])

#     # Get single example from this dataset.
#     sample = subjects_dataset[0]
#     print(sample['img']['data'].size())
#     # Get random patch from this sample.
#     class MySampler(RandomLabelSampler):
#         label_distribution = {'label1': 0.3, 'label2': 0.5}
#     patch_size = 50, 50, 1
#     sampler = MySampler(sample, patch_size)
#     patch = sampler.extract_patch(sample, patch_size)
#     print(patch['img']['data'].size())

#     plt.imshow(patch['img']['data'][0,:,:,0])
#     plt.show()


# if __name__ == '__main__':
#     main()
