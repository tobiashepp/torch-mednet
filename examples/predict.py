import logging
from pathlib import Path

import torch
import zarr 
import h5py
import hydra
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader

from midasmednet.dataset import GridPatchSampler, DataReaderHDF5
from midasmednet.segmentation import SegmentationNet
from midasmednet.landmarks import LandmarkNet


@hydra.main(config_path='/home/raheppt1/projects/tumorvolume/config/petct.yaml', strict=False)
def predict(cfg):
    # copy over
    data_path = cfg.base.data
    image_group = cfg.base.image_group
    test_set = cfg.prediction.test_set
    patch_size = cfg.prediction.patch_size
    patch_overlap = cfg.prediction.patch_overlap
    channel_selection = cfg.prediction.channel_selection
    num_heatmaps = len(cfg.base.sigma or [])
    batch_size = cfg.prediction.batch_size
    prediction_path = cfg.prediction.data 
    prediction_group = cfg.prediction.group 
    checkpoint_path = cfg.prediction.checkpoint
    chunk_size = cfg.prediction.chunk_size
    model_name = cfg.prediction.model

    # load validation and training key selection
    with open(test_set, 'r') as keyfile:
        test_keys = [l.strip() for l in keyfile.readlines()]
    print(f'total number of keys {len(test_keys)}')
    chunk_num = len(test_keys)//chunk_size
    chunks = np.array_split(np.array(test_keys), chunk_num)

    # load model
    print('loading model ...')
    if model_name == 'LandmarkNet':
        model = LandmarkNet.load_from_checkpoint(checkpoint_path)
    elif model_name == 'SegmentationNet':
        model = SegmentationNet.load_from_checkpoint(checkpoint_path)
    model.freeze()

    for c, chunk in enumerate(chunks):
        print(f'chunk {c}/{chunk_num}')
        chunk_keys = list(chunk)

        # data loading ...
        print('loading data ...')
        patch_dataset = GridPatchSampler(
                            data_path,
                            chunk_keys,
                            patch_size,
                            patch_overlap,
                            out_channels=num_heatmaps+1,
                            out_dtype=np.uint8,
                            channel_selection=channel_selection,
                            image_group=image_group,
                            ReaderClass=DataReaderHDF5,
                            pad_args={'mode': 'constant'})

        patch_loader = DataLoader(patch_dataset, 
                                batch_size=batch_size, 
                                num_workers=0)

        # use cuda device 
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'using {device}')
        model = model.to(device)

        # inference ...
        print('processing patches ...')
        model.eval()
        with torch.no_grad():
            for patch in patch_loader:
                # forward propagation only
                inputs = patch['data']
                inputs = inputs.float().to(device)
                logits = model(inputs)
                output_class =  logits[:, num_heatmaps:, ...]
                output_class = F.softmax(output_class, dim=1)
                output_class = torch.argmax(output_class, dim=1, keepdim=True).cpu().numpy()
                output_heatmaps = logits[:, :num_heatmaps, ...].cpu().numpy()
                output_heatmaps = np.clip(output_heatmaps, 0.0, 255.0)
                output = np.concatenate([output_heatmaps.astype(np.uint8),
                                        output_class.astype(np.uint8)], axis=1)
                # aggregate processed patches
                patch['data'] = output
                patch_dataset.add_processed_batch(patch)

        # assemble patches to original shapes
        results = patch_dataset.get_assembled_data()
        # save results
        if prediction_path:
            if Path(prediction_path).stem == '.h5':
                #save to hdf5
                with h5py.File(prediction_path, 'a') as hf:
                    gr = hf.require_group(prediction_group)
                    zarr.convenience.copy_all(results, gr)
            else:
                # save to zarr
                # TODO select storage
                store = zarr.ZipStore(prediction_path, mode='a')
                with zarr.open(store=store, mode='a') as zf:
                    gr = zf.require_group(prediction_group)
                    zarr.convenience.copy_all(results, gr)
                store.close()


if __name__ == '__main__':
    predict()