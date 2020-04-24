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
from midasmednet.unet.loss import DiceLoss, WeightedCrossEntropyLoss, compute_per_channel_dice
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

        # TODO validation and training data to experiment
        # including the data augmentation

        # copy over
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.in_channels = in_channels

        # initialization
        self.logger = logging.getLogger(__name__)

        # define loss function
        # TODO move to experiment section ?
        if loss_criterion == 'CE':
            self.criterion = WeightedCrossEntropyLoss()
        elif loss_criterion == 'DICE':
            self.criterion = DiceLoss(sigmoid_normalization=False)

    def training_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        labels = batch['label'][:, -1, ...].long()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        # metrics
        # TODO per channel dice to external file
        results = {'val_loss': loss}
        outputs = nn.Softmax(dim=1)(logits)
        labels = expand_as_one_hot(labels, C=outputs.size()[1])
        per_channel_dice = compute_per_channel_dice(outputs, labels)
        for c in range(self.out_channels):
            results[f'val_dice{c}'] = per_channel_dice[c]
        # TODO define in external file
        return results

    def validation_epoch_end(self, outputs):
        # TODO per channel dice
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()
        for c in range(self.out_channels):
            results[f'val_dice{c}'] = per_channel_dice[c]
        logs = {'val_loss': avg_loss, 'val_mae': avg_mae}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

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