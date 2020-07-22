import os
from pathlib import Path

import click
import h5py
import zarr
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import nilearn.image
load_dotenv()


@click.command()
@click.option('--data_path', required=True)
@click.option('--data_group', default='images')
@click.option('--export_dir', required=True)
@click.option('--sum_channels', default=False)
@click.option('--test_keys', default=None)
@click.option('--select_channels', default='all', type=click.Choice(['heatmaps', 'mask', 'all'], case_sensitive=False))
@click.option('--dtype', default='float', type=click.Choice(['float', 'int'], case_sensitive=False))
def export_to_nii(data_path,
                data_group,
                export_dir,
                select_channels=None,
                sum_channels=False,
                test_keys=None,
                dtype='float'):
    data_path = Path(data_path)
    export_dir = Path(export_dir)
    out_dir = export_dir/data_path.stem/data_group
    out_dir.mkdir(exist_ok=True, parents=True)

    assert dtype in ['float', 'int']
    if dtype == 'float':
        _dtype = np.float32 
    elif dtype == 'int':
        _dtype = np.uint8

    # open file storage
    print(data_path.suffix)
    assert data_path.suffix in ['.h5', '.zip', '.zarr']
    if data_path.suffix == '.h5':
        hf = h5py.File(data_path, 'r')
    else:
        if data_path.suffix == '.zarr':
            store = zarr.DirectoryStore(data_path)
        elif data_path.suffix == '.zip':
            store = zarr.ZipStore(data_path, mode='r')
        hf = zarr.open(store=store, mode='r')

    # load test keys
    if test_keys:
        with open(test_keys, 'r') as f:
            keys = [f.strip() for f in f.readlines()]
    else:
        keys = list(hf[data_group])

    # export files
    for key in tqdm(keys):
        ds = hf[f'{data_group}/{key}']

        assert select_channels in ['heatmaps', 'mask', 'all']
        if sum_channels:
            if select_channels == 'all':
                img = ds[:]
            elif select_channels == 'heatmaps':
                img = ds[:-1]
            elif select_channels == 'mask':
                img = ds[-1:]
            img = img.astype(_dtype)
            img = img.sum(axis=0)
            affine = ds.attrs['affine']
            nii_img = nib.Nifti1Image(img, affine)
            nib.save(nii_img, out_dir/f'{key}_{data_group}_{select_channels}_sum.nii.gz')
        else:
            channels = range(ds.shape[0])
            for c in channels:
                img = ds[c, ...].astype(_dtype)
                affine = ds.attrs['affine']
                nii_img = nib.Nifti1Image(img, affine)
                nib.save(nii_img, out_dir/f'{key}_{data_group}_c{c}.nii.gz')
    
    # close file storage
    if data_path.suffix == '.h5':
        hf.close()
    else:
        store.close()


if __name__ == '__main__':
    export_to_nii()