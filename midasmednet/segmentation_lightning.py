import os
import logging
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import torch 
import time
from midasmednet.unet.model import ResidualUNet3D
from midasmednet.unet.loss import DiceLoss, WeightedCrossEntropyLoss, dice_metric
from midasmednet.unet.loss import expand_as_one_hot

class SegmentationTrainer(ResidualUNet3D):

    def __init__(self,
                 training_dataset,
                 validation_dataset,
                 batch_size,
                 num_workers,
                 in_channels,
                 out_channels,
                 f_maps,
                 loss_criterion,
                 learning_rate,
                 hparams=None):

        # create model
        super(SegmentationTrainer, self).__init__(in_channels, out_channels,
                                                 final_sigmoid=False, f_maps=f_maps)
        # copy over
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.loss_criterion = loss_criterion

        # initialization
        self.logger = logging.getLogger(__name__)

    def training_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        outputs = self(inputs)
        loss = self.loss_criterion(outputs, labels)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        outputs = self(inputs)
        # metrics
        loss = self.loss_criterion(outputs, labels)
        per_channel_dice = dice_metric(outputs, labels)
        # store results to dictionary
        results = {'val_loss': loss}
        for c in range(self.out_channels):
            results[f'val_dice{c}'] = per_channel_dice[c]
        return results

    def validation_epoch_end(self, outputs):
        # average metrics over epoch
        # TODO plot images
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
                        shuffle=True)