from pathlib import Path
import torch
import time
import zarr
import re
import logging
from pathlib import Path
import nibabel as nib
import numpy  as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import midasmednet.dataset
from midasmednet.dataset import GridPatchSampler
from midasmednet.prediction import test_model
from midasmednet.utils.export import export_results_to_zarr
import logging
from sacred import Experiment
from sys import stdout

ex = Experiment('mednet_prediction')
#ex.add_config('/home/raheppt1/projects/mednet/config/aortath_landmarks.yaml')


@ex.config
def prediction_config():
    zarr_path = '/mnt/share/raheppt1/data/aortath/interim/zarr/full_1_5_mm_LAS_preprocessed.zarr'
    zarr_out_path = '/mnt/share/raheppt1/data/aortath/processed/predictions/zarr/full_1_5_mm_LAS_predictions.zarr'
    model_path = '/mnt/share/raheppt1/data/aortath/processed/models/aorta_segmentation12_200411_025329_model.pt'

    # load subjects
    with zarr.open(zarr_path, 'r') as zf:
        all_subject_keys = [key for key in zf['images_norm']]
    chunk_range = range(8,23)
    chunk_size = 500

    input_group = 'images_norm'
    output_group = 'label_predictions'
    output_dtype = np.uint8
    patch_size = [64, 64, 64]
    patch_overlap = [5, 5, 5]
    batch_size = 8
    out_channels = 2
    in_channels = 1
    fmaps = 64

@ex.automain
def main(zarr_path,
        zarr_out_path,
        model_path,
        all_subject_keys,
        chunk_range,
        chunk_size,
        input_group,
        output_group,
        output_dtype,
        patch_size,
        patch_overlap,
        batch_size,
        out_channels,
        in_channels,
        fmaps):

    print('inference')
    t = time.perf_counter()
    for t in chunk_range:
        # select <chunk_size> subject keys 
        start_subj = t*chunk_size
        end_subj = min((t+1)*chunk_size, len(all_subject_keys))
        subject_keys = all_subject_keys[start_subj:end_subj]
        print(f'processing subjects {start_subj} to {end_subj}')
        # predict results
        results = test_model(input_data_path=zarr_path,
                            input_group=input_group,
                            model_path=model_path,
                            subject_keys=subject_keys,
                            patch_size=patch_size,
                            patch_overlap=patch_overlap,
                            batch_size=batch_size,
                            out_channels=out_channels,
                            in_channels=in_channels,
                            fmaps=fmaps,
                            num_workers=0,
                            one_hot_encoded=False,
                            data_reader=midasmednet.dataset.read_zarr)
        print(f'finished {t - time.perf_counter():.3f}s')
        # write results to zarr file
        print('write results to zarr file')
        export_results_to_zarr(zarr_out_path,
                            results,
                            output_group,
                            subject_keys,
                            dtype=output_dtype)
        print(f'finished {t - time.perf_counter():.3f}s')

    # copy 'affines' group from zarr to zarr
    print('copy affines')
    with zarr.open(zarr_path, 'r') as source:
        with zarr.open(zarr_out_path, 'a') as destination:
            zarr.copy(source['affines'], destination, log=stdout)
    print(f'finished {t - time.perf_counter():.3f}s')


if __name__ == '__main__':
    main()
