import os
import logging
import numpy as np
from configargparse import ArgumentParser
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import tempfile
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import torch 
import time
from pathlib import Path
import matplotlib.pyplot as plt
from midasmednet.unet.model import ResidualUNet3D
from midasmednet.unet.loss import DiceLoss, WeightedCrossEntropyLoss, dice_metric
from midasmednet.unet.loss import expand_as_one_hot
from midasmednet.utils.plots import vis_logimages, vis_loglabels, vis_logheatmaps
from torchvision.utils import make_grid

class LandmarkNet(ResidualUNet3D):

    def __init__(self,
                 hparams,
                 training_dataset=None,
                 validation_dataset=None):

        # create model
        super(LandmarkNet, self).__init__(hparams.in_channels, hparams.out_channels,
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
        if hasattr(hparams, 'loss_class'):
            assert hparams.loss_class in ['DICE', 'CE']
            loss_class_weight = torch.tensor(hparams.loss_class_weight)
            if hparams.loss_class == 'DICE':
                self.loss_class = DiceLoss(weight=loss_class_weight)
            elif hparams.loss_class == 'CE':
                self.loss_class = torch.nn.CrossEntropyLoss(weight=loss_class_weight)

            assert hparams.loss_regression in ['L2', 'L1']
            if hparams.loss_regression == 'L2':
                self.loss_regression = torch.nn.MSELoss()
            elif hparams.loss_regression == 'L1':
                self.loss_regression = torch.nn.L1Loss()
            self.loss_regression_weight = hparams.loss_regression_weight
            self.num_heatmaps = len(self.loss_regression_weight)

        # optional
        self.log_interval = hparams.log_interval if hasattr(hparams, 'log_interval') else 5
        self.log_vis_mip = hparams.log_vis_mip if hasattr(hparams, 'log_vis_mip') else 'mean'

        # initialization
        self.logger = logging.getLogger(__name__)

    def training_step(self, batch, batch_nb):
        inputs = batch['data'].float()
        heatmaps = batch['label'][:, :-1, ...].float()
        num_heatmaps = heatmaps.shape[1] 
        labels = batch['label'][:, -1, ...].long()

        # output of the network is assumed to be un-normalized
        outputs = self(inputs)
        output_labels =  outputs[:, num_heatmaps:, ...]
        output_heatmaps = outputs[:, :num_heatmaps, ...]

        # calculate loss
        loss, class_loss, regression_loss = \
            self.loss(output_labels, output_heatmaps, labels, heatmaps)
        tensorboard_logs = {"train_loss": loss.item(), 
                            "class_loss": class_loss.item(),
                            "regression_loss": regression_loss.item()}
        return {'loss': loss, 'log': tensorboard_logs}

    def log_samples(self, batch, outputs, batch_id):
        # extract data
        inputs = batch['data'].float().cpu().numpy()
        heatmaps = batch['label'][:, :-1, ...].float().cpu().numpy()
        num_heatmaps = heatmaps.shape[1] 
        labels = batch['label'][:, -1, ...].long().cpu().numpy()
        outputs_labels =  outputs[:, num_heatmaps:, ...]
        pred_class = F.softmax(outputs_labels, dim=1)
        pred_class = torch.argmax(pred_class, dim=1).cpu().numpy()
        outputs_heatmaps = outputs[:, :num_heatmaps, ...].cpu().numpy()
        
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

            # heatmaps
            fig, ax = vis_logheatmaps(inputs[0, 0,...], 
                                      outputs_heatmaps[0,...], heatmaps[0,...],
                                      projection_type=self.log_vis_mip)
            plt.title(f"epoch {self.current_epoch} batch {batch_id}")
            plt.savefig(png_path, bbox_inches='tight',pad_inches = 0, dpi = 200) 
            plt.close(fig)
            self.logger[1].experiment.log_image('heatmaps', png_path)
        return

    def loss(self, output_labels, output_heatmaps,
                   labels, heatmaps):
        class_loss = self.loss_class(output_labels, labels)
        regression_loss = torch.tensor(0.0).type_as(output_labels)
        for c in range(self.num_heatmaps):
            regression_loss += self.loss_regression_weight[c]* \
                    self.loss_regression(output_heatmaps[:, c, ...], 
                                         heatmaps[:, c, ...])
        loss = regression_loss + class_loss
        return loss, class_loss, regression_loss

    def validation_step(self, batch, batch_nb):
        # extract data
        inputs = batch['data'].float()
        heatmaps = batch['label'][:, :-1, ...].float()
        labels = batch['label'][:, -1, ...].long()

        # output of the network is assumed to be un-normalized
        outputs = self(inputs)
        output_labels =  outputs[:, self.num_heatmaps:, ...]
        output_heatmaps = outputs[:, :self.num_heatmaps, ...]

        # log samples 
        if batch_nb%self.log_interval == 0:
            self.log_samples(batch, outputs, batch_nb)

        # calculate loss & metrics
        loss, class_loss, regression_loss = \
            self.loss(output_labels, output_heatmaps, labels, heatmaps)
        per_channel_dice = dice_metric(output_labels, labels)

        # store results to dictionary
        results = {'val_loss': loss,
                   'val_class_loss': class_loss,
                   'val_regression_loss': regression_loss }
        for c in range(self.out_channels - self.num_heatmaps):
            results[f'val_dice{c}'] = per_channel_dice[c]
        return results

    def validation_epoch_end(self, outputs):
        # average metrics over epoch
        avg_class_loss = torch.stack([x['val_class_loss'] for x in outputs]).mean()
        avg_regression_loss = torch.stack([x['val_regression_loss'] for x in outputs]).mean()
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss,
                'val_class_loss': avg_class_loss,
                'val_regression_loss': avg_regression_loss }
        for c in range(self.out_channels - self.num_heatmaps):
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
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--fmaps", type=int, default=64)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--in_channels", type=int, default=1)
        parser.add_argument("--out_channels", type=int, default=1)
        parser.add_argument("--log_interval", type=int, default=5)
        parser.add_argument("--log_vis_mip", type=str, choices=['mean', 'max'], default='mean')
        parser.add_argument('--loss_class', choices=['DICE', 'CE'], default='DICE')
        parser.add_argument('--loss_class_weight', nargs='+', type=float, default=[0.05, 1.0])
        parser.add_argument('--loss_regression', choices=['L2', 'L1'], default='L2')
        parser.add_argument("--loss_regression_weight", type=float, nargs='+', default=[0.001, 0.015, 0.015, 0.015, 0.001, 0.001])
        return parser