from pathlib import Path
import torch
import time
import zarr
from pathlib import Path
import nibabel as nib
import numpy  as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import midasmednet.dataset
from midasmednet.dataset import GridPatchSampler
from midasmednet.prediction import test_model
import logging

zarr_path = '/mnt/share/raheppt1/data/aortath/interim/zarr/all_mra_preprocessed.zarr'

#subject_keys = ['100099']
subject_key_file = '/mnt/share/raheppt1/data/aortath/interim/crossval/validation.dat'
with open(subject_key_file, 'r') as f:
    subject_keys = [key.strip() for key in f.readlines()]

subject_keys = subject_keys[:10]

print('inference')
model_path = '/mnt/share/raheppt1/data/aortath/processed/models/segm_first_model.pt'
results = test_model(input_data_path=zarr_path,
                     input_group='images_norm',
                     model_path=model_path,
                     subject_keys=subject_keys,
                     patch_size=[64, 64, 64],
                     patch_overlap=[5, 5, 5],
                     batch_size=4,
                     out_channels=2,
                     in_channels=1,
                     fmaps=64,
                     num_workers=0,
                     one_hot_encoded=False,
                     data_reader=midasmednet.dataset.read_zarr)

print('write results to zarr file')

# write results to zarr file
zarr_path = '/mnt/share/raheppt1/data/aortath/processed/predictions/zarr/all_mra_predictions.zarr'
with zarr.open(zarr_path, 'a') as zf:
    gr = zf.require_group('mask_predictions')
    for key in subject_keys:
        print(key)
        output = results[key].astype(np.float16)
        ds = gr.require_dataset(key, output.shape, dtype=output.dtype)
        ds[:] = output
