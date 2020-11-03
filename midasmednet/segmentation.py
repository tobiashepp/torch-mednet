import os
import logging
import numpy as np
from configargparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import tempfile
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import torch 
import time
import matplotlib.pyplot as plt
from midasmednet.unet.model import ResidualUNet3D
from midasmednet.unet.loss import DiceLoss, WeightedCrossEntropyLoss, dice_metric
from midasmednet.unet.loss import expand_as_one_hot
from midasmednet.utils.plots import vis_logimages, vis_loglabels
from torchvision.utils import make_grid

class SegmentationNet(ResidualUNet3D):

    def __init__(self,
                 hparams,
                 training_dataset=None,
                 validation_dataset=None):

        # create model
        super(SegmentationNet, self).__init__(hparams.in_channels, hparams.out_channels,
                                                 final_sigmoid=False, f_maps= hparams.fmaps)
        # copy over
        self.hparams = hparams
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = hparams.learning_rate
        self.num_workers =  hparams.num_workers
        self.batch_size = hparams.batch_size
        self.out_channels =  hparams.out_channels
        self.in_channels =  hparams.in_channels

        # set loss criterion
        if hasattr(hparams, 'loss'):
            assert hparams.loss in ['DICE', 'CE']
            loss_weight = torch.tensor(hparams.loss_weight)
            if hparams.loss == 'DICE':
                self.loss = DiceLoss(weight=loss_weight)
            elif hparams.loss == 'CE':
                self.loss = torch.nn.CrossEntropyLoss(weight=loss_weight)

        # optional
        self.log_interval = hparams.log_interval if hasattr(hparams, 'log_interval') else 5
        self.log_vis_mip = hparams.log_vis_mip if hasattr(hparams, 'log_vis_mip') else 'mean'
        
        # initialization
        self.logger = logging.getLogger(__name__)

    def training_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        # output of the network is assumed to be un-normalized
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def log_samples(self, batch, outputs, batch_id):
        # extract data
        inputs = batch['data'].float().cpu().numpy()
        labels = batch['label'][:, -1, ...].long().cpu().numpy()
        prediction = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(prediction, dim=1).cpu().numpy()

        with tempfile.TemporaryDirectory() as test:
            png_path = str(Path(test)/'tmp.png')

            # images
            fig, ax = vis_logimages(inputs[0, ...])
            plt.title(f"epoch {self.current_epoch} batch {batch_id}")
            plt.savefig(png_path, bbox_inches='tight',pad_inches = 0, dpi = 200) 
            plt.close(fig)
            self.logger[1].experiment.log_image('images', png_path)

            # labels
            fig, ax = vis_loglabels(labels[0, ...], pred_class[0, ...],
                                    inputs=inputs[0, 0, ...],
                                    projection_type=self.log_vis_mip)
            plt.title(f"epoch {self.current_epoch} batch {batch_id}")
            plt.savefig(png_path, bbox_inches='tight',pad_inches = 0, dpi = 200) 
            plt.close(fig)
            self.logger[1].experiment.log_image('labels', png_path)
        return

    def validation_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        # output of the network is assumed to be un-normalized
        outputs = self(inputs)
        # log samples 
        if batch_nb%self.log_interval == 0:
            self.log_samples(batch, outputs, batch_id=batch_nb)
        # metrics
        loss = self.loss(outputs, labels)
        per_channel_dice = dice_metric(outputs, labels)
        # store results to dictionary
        results = {'val_loss': loss}
        for c in range(self.out_channels):
            results[f'val_dice{c}'] = per_channel_dice[c]
        return results

    def validation_epoch_end(self, outputs):
        # average metrics over epoch
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        for c in range(self.out_channels):
            logs[f"val_dice{c}"] = torch.stack([x[f"val_dice{c}"] for x in outputs]).mean()
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, 
                        batch_size=self.batch_size, 
                        num_workers=self.num_workers,
                        shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, 
                        batch_size=self.batch_size, 
                        num_workers=self.num_workers,
                        shuffle=False)