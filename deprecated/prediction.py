
import datetime
import time
import shutil
import visdom
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import midasmednet.unet as unet
import midasmednet.unet.model
import midasmednet.unet.loss
from midasmednet.utils.misc import heatmap_plot, class_plot
from midasmednet.dataset import LandmarkDataset, SegmentationDataset, GridPatchSampler
from midasmednet.unet.loss import expand_as_one_hot
import random


# TODO comments
# TODO heatmap and label separation / (output logits)
# todo add timing (s/patch, s/subject)
def test_model(input_data_path,
                input_group,
                model_path,
                subject_keys,
                patch_size, 
                patch_overlap,
                batch_size,
                out_channels,
                in_channels,
                fmaps,
                num_workers,
                one_hot_encoded=True,
                softmax_output=True,
                ReaderClass=midasmednet.dataset.DataReaderHDF5):

    logger = logging.getLogger(__name__)
    # set parameters
    patch_size = patch_size
    batch_size = batch_size
    subject_keys = subject_keys
    
    # check cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'using {device}')

    # create model and send it to GPU
    logger.info(f'creating U-net model (inputs {in_channels}, outputs {out_channels})')
    net = unet.model.ResidualUNet3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    final_sigmoid=False,
                                    f_maps=fmaps)

    # restore checkpoint from file
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    finished_epochs = checkpoint['epoch']
    logger.info(f'restoring model, epoch: {finished_epochs}')
    net.to(device)

    logger.info(f'inference ...')
    # create patched dataset
    if not one_hot_encoded:
        out_channels = 1
    patch_dataset = GridPatchSampler(input_data_path,
                                    subject_keys,
                                    patch_size, patch_overlap,
                                    image_group=input_group,
                                    out_channels=out_channels,
                                    out_dtype=np.float16)

    patch_loader = DataLoader(patch_dataset, batch_size=batch_size, num_workers=num_workers)

    # inference ...
    net.eval()
    with torch.no_grad():
        for patch in patch_loader:
            # forward propagation only
            inputs = patch['data']
            inputs = inputs.float().to(device)
            logits = net(inputs)
            if softmax_output:
                outputs = nn.Softmax(dim=1)(logits)
            else:
                outputs = logits

            # aggregate processed patches
            patch['data'] = outputs.cpu().numpy().astype(np.float16)
            if not one_hot_encoded:
                patch['data'] = np.expand_dims(np.argmax(patch['data'], axis=1), axis=1)
            patch_dataset.add_processed_batch(patch)
    
    results = patch_dataset.get_assembled_data()
    return results


