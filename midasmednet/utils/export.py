import zarr
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from sys import stdout

# todo modify prints 
def export_zarr_to_nifti(zarr_path,
                         out_dir,
                         group,
                         subject_keys,
                         dtype=np.float32):
    """Exports image group from zarr to nifti files.

    Specify an image group of the zarr data file.
    A additional group, named affine, including the 
    corresponing affine transformation matrices is required.

    Args:
        zarr_path (str/Path): path to zarr data
        out_dir (str/Path): directory to store created nifti files
        group (str): zarr group which should be exported
        subject_keys (list): list of subject keys. 
        dtype (type, optional): data type for the exported data. Defaults to np.float32.
    """
    logger = logging.getLogger(__name__)
    with zarr.open(str(zarr_path), 'r') as zf:
        for key in subject_keys:
            affine = zf[f'affines/{key}']
            data = zf[f'{group}/{key}']
            channels = data.shape[0]
            logger.debug(f'subject {key}, shape {data.shape}')
            for c in range(channels):
                channel_data = data[c,...]
                channel_data = channel_data.astype(dtype)
                img = nib.Nifti1Image(channel_data, affine)
                save_path = Path(out_dir).joinpath(f'{key}_{group}_{c}.nii.gz')
                nib.save(img, save_path)


def export_results_to_zarr(zarr_path,
                           results,
                           group,
                           subject_keys,
                           dtype=np.uint8):
    """Export result list with arrays to zarr dataset.
    
    Args:
        zarr_path (str/Path): path to zarr data
        results (list): list of array to export
        group (str): zarr group, which should be exported 
        subject_keys (list):  list of subject keys.
        dtype (type, optional): data type for the exported data. Defaults to np.float32.
    """
    logger = logging.getLogger(__name__)
    with zarr.open(zarr_path, 'a') as zf:
        gr = zf.require_group(group)
        for key in subject_keys:
            output = results[key].astype(dtype)
            ds = gr.require_dataset(key, output.shape, dtype=output.dtype)
            ds[:] = output
            logger.debug(f'exporting {key}, shape {output.shape}, type {output.dtype}')


def copy_affines(zarr_source_path,
                 zarr_destination_path):

    with zarr.open(zarr_source_path, 'r') as source:
        with zarr.open(zarr_destination_path, 'a') as destination:
            zarr.copy(source['affines'], destination, log=stdout)
