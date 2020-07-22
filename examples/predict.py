import torch
import numpy as np
from pytorch_lightning import Trainer, LightningModule
from midasmednet.dataset import GridPatchSampler, DataReaderHDF5
from midasmednet.unet.loss import DiceLoss
from midasmednet.segmentation_lightning import SegmentationNet
from midasmednet.landmarks_lightning import LandmarkNet
from torch.utils.data import DataLoader
import logging
import zarr, h5py
from pathlib import Path
from torch.nn import functional as F
from configargparse import ArgumentParser
from pytorch_lightning import loggers
from pytorch_lightning.logging.neptune import NeptuneLogger
from midasmednet.utils.misc import _LOG_LEVEL_STRINGS, _log_level_string_to_int
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform, ContrastAugmentationTransform

# arg parsing
parser = ArgumentParser()
# add experiment level args
parser.add_argument('-c', '--config', is_config_file=True, default='/mnt/share/raheppt1/data/tumorvolume/ctorgans/config/ctorgans_prediction.yaml')
parser.add_argument('--data_path', type=str)
parser.add_argument('--image_group', type=str, default='images')
parser.add_argument('--test_set', type=str)
parser.add_argument('--patch_size', type=int, nargs='+', default=[96, 96, 96])
parser.add_argument('--patch_overlap', type=int, nargs='+', default=[10, 10, 10])
parser.add_argument('--channel_selection', type=int, nargs='+', default=[0])
parser.add_argument('--num_heatmaps', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--prediction_path', type=str)
parser.add_argument('--prediction_group', type=str, default="prediction")
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--chunk_size', type=int, default=10)
parser.add_argument('--model', type=str, choices=['LandmarkNet', 'SegmentationNet'], default='LandmarkNet')
hparams = parser.parse_args()

# load validation and training key selection
with open(hparams.test_set, "r") as keyfile:
    test_keys = [l.strip() for l in keyfile.readlines()]
print(f'total number of keys {len(test_keys)}')
chunk_num = len(test_keys)//hparams.chunk_size
chunks = np.array_split(np.array(test_keys), chunk_num)

# load model
print('loading model ...')
if hparams.model == 'LandmarkNet':
    model = LandmarkNet.load_from_checkpoint(hparams.checkpoint_path)
elif hparams.model == 'SegmentationNet':
    model = SegmentationNet.load_from_checkpoint(hparams.checkpoint_path)
model.freeze()

for c, chunk in enumerate(chunks):
    print(f'chunk {c}/{chunk_num}')
    chunk_keys = list(chunk)

    # data loading ...
    print('loading data ...')
    patch_dataset = GridPatchSampler(
                        hparams.data_path,
                        chunk_keys,
                        hparams.patch_size,
                        hparams.patch_overlap,
                        out_channels=hparams.num_heatmaps+1,
                        out_dtype=np.uint8,
                        channel_selection=hparams.channel_selection,
                        image_group=hparams.image_group,
                        ReaderClass=DataReaderHDF5,
                        pad_args={'mode': 'constant'})

    patch_loader = DataLoader(patch_dataset, 
                            batch_size=hparams.batch_size, 
                            num_workers=0)

    # use cuda device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            output_class =  logits[:, hparams.num_heatmaps:, ...]
            output_class = F.softmax(output_class, dim=1)
            output_class = torch.argmax(output_class, dim=1, keepdim=True).cpu().numpy()
            output_heatmaps = logits[:, :hparams.num_heatmaps, ...].cpu().numpy()
            output_heatmaps = np.clip(output_heatmaps, 0.0, 255.0)
            output = np.concatenate([output_heatmaps.astype(np.uint8),
                                    output_class.astype(np.uint8)], axis=1)
            # aggregate processed patches
            patch['data'] = output
            patch_dataset.add_processed_batch(patch)

    # assemble patches to original shapes
    results = patch_dataset.get_assembled_data()
    # save results
    if hparams.prediction_path:
        
        if Path(hparams.prediction_path).stem == '.h5':
            #save to hdf5
            with h5py.File(hparams.h5_path, 'a') as hf:
                gr = hf.require_group(hparams.prediction_group)
                zarr.convenience.copy_all(results, gr)
        else:
            # save to zarr
            # TODO select storage
            store = zarr.ZipStore(hparams.prediction_path, mode='a')
            with zarr.open(store=store, mode='a') as zf:
                gr = zf.require_group(hparams.prediction_group)
                zarr.convenience.copy_all(results, gr)
            store.close()
