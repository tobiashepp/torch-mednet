import datetime
import time
import shutil
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
from midasmednet.dataset import LandmarkDataset, SegmentationDataset
import random
# todo reweighted loss
# todo data augmentation to config file

class LandmarkTrainer:

    def __init__(self,
                 config_path,
                 b_restore=False,
                 b_shuffle_subjects=False):

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        # define parameters from config dict
        self.run_name = self.config['run']
        self.log_dir = self.config['log_dir']
        self.max_epochs = self.config['max_epochs']
        self.print_interval = self.config['print_interval']
        self.model_path = self.config['model_path']

        # create training and validation datasets
        training_subject_keys = self.config['training_subject_keys']
        validation_subject_keys = self.config['validation_subject_keys']
        self.training_ds = self._create_dataset(training_subject_keys)
        self.validation_ds = self._create_dataset(validation_subject_keys)
        
        self.dataloader_training = DataLoader(self.training_ds, shuffle=True, 
                                           batch_size=self.config['batch_size'],
                                           num_workers=self.config['num_workers'])
        self.dataloader_validation = DataLoader(self.validation_ds, shuffle=True,
                                            batch_size=self.config['batch_size'],
                                            num_workers=self.config['num_workers'])

        # initialize tensorboard writer
        self.writer = self._init_writer()

        # check cuda device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device}')

        # create model and send it to GPU
        self.net = midasmednet.unet.model.ResidualUNet3D(in_channels=self.config['model']['in_channels'],
                                                         out_channels=self.config['model']['out_channels'],
                                                         final_sigmoid=False,
                                                         f_maps=self.config['model']['f_maps'])
        self.net.to(self.device)

        # initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                          lr=self.config['learning_rate'])
       
        self.criterion = midasmednet.unet.loss.LandmarkLoss()

        # Restore from checkpoint?
        self.start_epoch = 0
        self.val_loss_min = None
        if b_restore:
            self.start_epoch, self.val_loss_min = self._restore_model()

    def _create_dataset(self, subject_key_file):
        # todo add data augmentation
        with open(subject_key_file, 'r') as f:
            subject_keys = [key.strip() for key in f.readlines()]
        print(subject_keys)
        # define dataset
        ds = LandmarkDataset(path_h5data=self.config['path_h5data'],
                             subject_keys=subject_keys,
                             samples_per_subject=self.config['samples_per_subject'],
                             patch_size=self.config['patch_size'],
                             class_probabilities=self.config['class_probabilities'], 
                             transform=None, 
                             verbose=True,
                             #heatmap_treshold=self.config['heatmap_treshold'],
                             h5_image_group=self.config['h5_image_group'],
                             h5_heatmap_group=self.config['h5_heatmap_group'])
        return ds

    def _init_writer(self):
        ts = datetime.datetime.now().timestamp()
        readable = datetime.datetime.fromtimestamp(ts).isoformat()
        log_dir = Path(self.log_dir)
        log_dir.mkdir(exist_ok=True)
        if self.run_name:
            log_dir = log_dir.joinpath('log_' + self.run_name)
        else:
            log_dir = log_dir.joinpath('log_' + readable)
        writer = SummaryWriter(log_dir)
        return writer

    def _save_model(self, epoch, loss):
        print('Saving new checkpoint ...')
        model_path = Path(self.model_path)
        model_path = model_path.joinpath(self.run_name+'_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss},
            str(model_path))

    def _restore_model(self):
        print('Loading checkpoint ...')
        model_path = Path(self.model_path)
        model_path = model_path.joinpath(self.run_name+'_model.pt')
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        val_loss_min = checkpoint['loss']
        return start_epoch, val_loss_min

    def _train_epoch(self, epoch):
        print_interval = self.print_interval
        # Training loop ...
        self.net.train()
        running_loss = 0.0
        for step, batch in enumerate(self.dataloader_training):

            # load input and targets from batch and send them to GPU
            inputs =  batch['data'].float()
            labels = batch['label'][:,-1,...].long()
            heatmaps = batch['label'][:, :-1, ...].float()
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            heatmaps = heatmaps.to(self.device)

            # training stept
            # forward pass
            self.optimizer.zero_grad()
            logits = self.net(inputs)
            # backpropagation
            #loss_mse = self.criterion(logits, heatmaps)
            loss = torch.nn.MSELoss()
            mse = []
            weights = [1.0, 15.0, 15.0, 15.0, 1.0, 1.0]
            for c in range(len(weights)):
                mse.append(weights[c]*loss(logits[:,c,...], heatmaps[:,c,...]))
            loss_mse = sum(mse)
            loss = loss_mse
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if step % print_interval == (print_interval - 1):
                print('[%d, %4d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_interval))

                global_step = epoch * len(self.dataloader_training) + (step + 1)

                self.writer.add_scalar('Loss/training',
                                       running_loss / print_interval,
                                       global_step=global_step)

                self.writer.add_figure('Sample', heatmap_plot(inputs[0], logits[0], heatmaps[0]),
                                       global_step=global_step)
                running_loss = 0.0

    def _test(self, epoch):
        running_loss = 0.0
        mse_loss = 0.0
        self.net.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dataloader_validation)):
                # load input and targets from batch and send them to GPU
                inputs =  batch['data'].float()
                labels = batch['label'][:,-1,...].long()
                heatmaps = batch['label'][:, :-1, ...].float()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                heatmaps = heatmaps.to(self.device)
            
                # Forward propagation only.
                logits = self.net(inputs)

                # Calculate metrics.
                loss = torch.nn.MSELoss()
                mse = []
                weights = [1.0, 15.0, 15.0, 15.0, 1.0, 1.0]
                for c in range(len(weights)):
                    mse.append(weights[c]*loss(logits[:,c,...], heatmaps[:,c,...]))
                loss_mse = sum(mse)
                loss = loss_mse

                running_loss += loss
                mse_loss += loss_mse

        # Compute mean loss/dice metrics and log to tensorboard.
        validation_loss = running_loss / len(self.dataloader_validation)
        self.writer.add_scalar('Loss/validation',
                               validation_loss,
                               global_step=epoch + 1)

        mse_loss = mse_loss / len(self.dataloader_validation)
        self.writer.add_scalar('Loss/valmse',
                               mse_loss,
                               global_step=epoch + 1)
        return validation_loss

    def run(self):
        # Parameters
        max_epochs = self.max_epochs

        # Variables
        start_epoch = self.start_epoch
        val_loss_min = self.val_loss_min

        print(f'Training started ...')
        start = time.time()
        for epoch in range(start_epoch, max_epochs):
            start_time = time.time()

            # Train for one epoch.
            print('Train ...')
            self._train_epoch(epoch)

            # Evaluate ...
            print('Validate ...')
            validation_loss = self._test(epoch)

            end_time = time.time()
            print("Epoch {}, time {:.2f}".format(
                epoch + 1, end_time - start_time))

            # Save checkpoint if the current eval_loss is the lowest.
            if not val_loss_min:
                val_loss_min = validation_loss
            if validation_loss < val_loss_min or epoch == 0:
                val_loss_min = validation_loss
                self._save_model(epoch + 1, validation_loss)

        print('Time:', int(time.time() - start), 'seconds')


def main():
    trainer = LandmarkTrainer(
        '/home/raheppt1/projects/mednet/config/aortath_landmarks.yaml')
    trainer.run()


if __name__ == "__main__":
    main()
