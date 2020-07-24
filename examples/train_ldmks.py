import os
import logging

import torch
import numpy as np
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from configargparse import ArgumentParser
from pytorch_lightning import loggers
from pytorch_lightning.logging.neptune import NeptuneLogger
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform, ContrastAugmentationTransform

from midasmednet.utils.misc import _LOG_LEVEL_STRINGS, _log_level_string_to_int
from midasmednet.dataset import MedDataset, DataReaderHDF5
from midasmednet.unet.loss import DiceLoss
from midasmednet.landmarks import LandmarkNet

load_dotenv()

# replace $DATA and $MODEL in paths
# by the values of the env variables
DATA = os.getenv('DATA')
MODEL = os.getenv('MODEL')
def _replace_env(path_str):
    path_str = str(path_str)
    mod_str = path_str.replace('$DATA', DATA)
    mod_str = path_str.replace('$MODEL', MODEL)
    return mod_str

# arg parsing
parser = ArgumentParser()
# add experiment level args
parser.add_argument('-c', '--config', is_config_file=True, default="/mnt/share/raheppt1/data/vessel/config/aorth_ldmks.yaml")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--neptune_project', type=str, default='lab-midas/mednet')
parser.add_argument('--experiment_name', type=str, default="aorth_ldmks")
parser.add_argument('--data_path', type=_replace_env)
parser.add_argument('--image_group', type=str, default='images')
parser.add_argument('--label_group', type=str, default='labels')
parser.add_argument('--heatmap_group', type=str, default='heatmaps')
parser.add_argument('--train_set', type=str)
parser.add_argument('--val_set', type=str)
parser.add_argument('--model_dir', type=_replace_env)
parser.add_argument('--log_dir', type=_replace_env)
parser.add_argument('--patch_size', type=int, nargs='+', default=[96, 96, 96])
parser.add_argument('--class_probabilities', type=float, nargs='+', default=None)
parser.add_argument('--patches_per_subject', type=int, default=10)
parser.add_argument('--data_augmentation', action="store_true")
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--preload', action='store_true')
parser.add_argument('--resume', type=str)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--log_level', type=str, default='INFO')

# add model specific args
parser = LandmarkNet.add_model_specific_args(parser)
hparams = parser.parse_args()

# set seeds
torch.manual_seed(hparams.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # False
np.random.seed(hparams.seed)

# logging
# in order to use neptune logging:
# export NEPTUNE_API_TOKEN = '...' !!! 
logging.getLogger().setLevel('INFO')
source_files = [__file__]
if hparams.config:
    source_files.append(hparams.config)
neptune_logger = NeptuneLogger(
    project_name=hparams.neptune_project,
    params=vars(hparams),
    experiment_name=hparams.experiment_name,
    tags=[hparams.experiment_name],
    upload_source_files=source_files)
tb_logger = loggers.TensorBoardLogger(hparams.log_dir)

transform = Compose([BrightnessTransform(mu=0.0, sigma=0.3, data_key='data'),
                     GammaTransform(gamma_range=(0.7, 1.3), data_key='data'),
                     ContrastAugmentationTransform(contrast_range=(0.3, 1.7), data_key='data')])

with open(hparams.train_set, 'r') as keyfile:
    train_keys = [l.strip() for l in keyfile.readlines()]
print(train_keys)

with open(hparams.val_set, 'r') as keyfile:
    val_keys = [l.strip() for l in keyfile.readlines()]
print(val_keys)

train_ds = MedDataset(hparams.data_path,
                 train_keys,
                 hparams.patches_per_subject,
                 hparams.patch_size,
                 image_group=hparams.image_group,
                 label_group=hparams.label_group,
                 heatmap_group=hparams.heatmap_group,
                 class_probabilities=hparams.class_probabilities,
                 transform=transform,
                 ReaderClass=DataReaderHDF5)

val_ds = MedDataset(hparams.data_path,
                 val_keys, 
                 hparams.patches_per_subject,
                 hparams.patch_size,
                 image_group=hparams.image_group,
                 label_group=hparams.label_group,
                 heatmap_group=hparams.heatmap_group,
                 class_probabilities=None,
                 ReaderClass=DataReaderHDF5)

model = LandmarkNet(hparams,
                    train_ds, val_ds)

kwargs = {}
if hparams.resume:
    print('loading checkpoint ...')
    kwargs['resume_from_checkpoint'] = hparams.resume

trainer = Trainer(gpus=hparams.gpus,
                #precision=16, amp_level='O2',
                max_epochs=hparams.max_epochs,
                default_root_dir=hparams.model_dir,
                logger=[tb_logger, neptune_logger],
                **kwargs)
trainer.fit(model)

