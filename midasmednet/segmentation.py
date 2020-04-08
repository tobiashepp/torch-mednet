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

# global todo list
# todo reweighted loss
# todo data augmentation to config file


class SegmentationTrainer:

    def __init__(self,
                 run_name,
                 log_dir,
                 model_path,
                 print_interval,
                 max_epochs,
                 learning_rate,
                 data_path,
                 training_subject_keys,
                 validation_subject_keys,
                 image_group,
                 label_group,
                 samples_per_subject,
                 class_probabilities,
                 patch_size, batch_size,
                 num_workers,
                 in_channels,
                 out_channels,
                 f_maps,
                 data_reader=midasmednet.dataset.read_zarr,
                 b_restore=False):

        # define parameters
        self.logger = logging.getLogger(__name__)
        self.run_name = run_name
        self.log_dir = log_dir
        self.print_interval = print_interval
        self.model_path = model_path
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.training_subject_keys = training_subject_keys
        self.validation_subject_keys = validation_subject_keys
        self.image_group = image_group
        self.label_group = label_group
        self.samples_per_subject = samples_per_subject
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_maps = f_maps
        self.patch_size = patch_size
        self.class_probabilities = class_probabilities
        self.data_reader = data_reader
        self.transform = None

        # create training and validation datasets
        self.logger.info('copying training data to memory ...')
        self.training_ds = self._create_dataset(training_subject_keys)
        self.logger.info('copying validation data to memory ...')
        self.validation_ds = self._create_dataset(validation_subject_keys)

        self.dataloader_training = DataLoader(self.training_ds, shuffle=True,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers)
        
        self.dataloader_validation = DataLoader(self.validation_ds, shuffle=True,
                                                batch_size=self.batch_size,
                                                num_workers=self.num_workers)

        # initialize tensorboard writer
        self.writer = self._init_writer()

        # check cuda device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device}')

        # create model and send it to GPU
        self.net = midasmednet.unet.model.ResidualUNet3D(in_channels=self.in_channels,
                                                         out_channels=self.out_channels,
                                                         final_sigmoid=False,
                                                         f_maps=self.f_maps)
        self.net.to(self.device)

        # initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                          lr=self.learning_rate)

        self.loss_criterion = 'DICE'
        if self.loss_criterion == 'CE':
            self.criterion = midasmednet.unet.loss.WeightedCrossEntropyLoss()
        elif self.loss_criterion == 'DICE':
            self.criterion = midasmednet.unet.loss.DiceLoss(sigmoid_normalization=False)

        # Restore from checkpoint?
        self.start_epoch = 0
        self.val_loss_min = None
        if b_restore:
            self.start_epoch, self.val_loss_min = self._restore_model()

    def _create_dataset(self, subject_key_file):
        # read subject keys from file
        with open(subject_key_file, 'r') as f:
            subject_keys = [key.strip() for key in f.readlines()]
        # define dataset
        ds = SegmentationDataset(data_path=self.data_path,
                                 subject_keys=subject_keys,
                                 samples_per_subject=self.samples_per_subject,
                                 patch_size=self.patch_size,
                                 class_probabilities=self.class_probabilities,
                                 transform=None,
                                 data_reader=self.data_reader,
                                 image_group=self.image_group,
                                 label_group=self.label_group)
        return ds

    def _init_writer(self):
        ts = datetime.datetime.now().timestamp()
        readable = datetime.datetime.fromtimestamp(ts).isoformat()
        log_dir = Path(self.log_dir)
        log_dir.mkdir(exist_ok=True)
        if self.run_name:
            log_dir = log_dir.joinpath('log_' + self.run_name + readable)
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
            inputs = batch['data'].float().to(self.device)
            labels = batch['label'][:, -1, ...].long().to(self.device)
            # training step
            # forward pass
            self.optimizer.zero_grad()
            logits = self.net(inputs)
            # backpropagation
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if step % print_interval == (print_interval - 1):
                print('[%d, %4d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_interval))

                global_step = epoch * \
                    len(self.dataloader_training) + (step + 1)

                self.writer.add_scalar('Loss/training',
                                       running_loss / print_interval,
                                       global_step=global_step)

                self.writer.add_figure('Segementation', class_plot(inputs[0], logits[0], labels[0]),
                                       global_step=global_step)
                running_loss = 0.0

    def _test(self, epoch):
        running_loss = 0.0
        per_channel_dice = 0.0
        self.net.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dataloader_validation)):
                # load input and targets from batch and send them to GPU
                inputs = batch['data'].float().to(self.device)
                labels = batch['label'][:, -1, ...].long().to(self.device)
                # Forward propagation only.
                logits = self.net(inputs)
                # Calculate metrics.
                loss = self.criterion(logits, labels)
                running_loss += loss.item()
                # expand as on hot for dice loss
                outputs = nn.Softmax(dim=1)(logits)
                labels = expand_as_one_hot(labels, C=outputs.size()[1])
                per_channel_dice += midasmednet.unet.loss.compute_per_channel_dice(
                                                    outputs, labels)

        # Compute mean loss/dice metrics and log to tensorboard.
        validation_loss = running_loss / len(self.dataloader_validation)
        self.writer.add_scalar('Loss/validation',
                               validation_loss,
                               global_step=epoch + 1)

        per_channel_dice = per_channel_dice/len(self.dataloader_validation)
        for k in range(per_channel_dice.size()[0]):
            self.writer.add_scalar(f'Dice/label{k}',
                                   per_channel_dice[k],
                                   global_step=epoch+1)
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

# todo documentation
def test_segmentation(input_data_path,
                     input_group,
                     output_data_path,
                     output_group,
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
                     data_reader=midasmednet.dataset.read_zarr):

    logger = logging.getLogger(__name__)
    # set parameters
    patch_size = patch_size
    batch_size = batch_size
    subject_keys = subject_keys
    
    # check cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using {device}')

    # create model and send it to GPU
    logger.info(f'unet: inputs {in_channels} outpus {out_channels}')
    net = unet.model.ResidualUNet3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    final_sigmoid=False,
                                    f_maps=fmaps)

    # restore checkpoint from file
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    finished_epochs = checkpoint['epoch']
    logger.info(f'Restoring model, epoch: {finished_epochs}')
    net.to(device)

    # create patch sample
    if not one_hot_encoded:
        out_channels = 1
    patch_dataset = GridPatchSampler(input_data_path,
                                        subject_keys,
                                        patch_size, patch_overlap,
                                        image_group=input_group,
                                        out_channels=out_channels,
                                        out_dtype=np.uint8)

    patch_loader = DataLoader(patch_dataset, batch_size=batch_size, num_workers=num_workers)

    net.eval()
    with torch.no_grad():
        for patch in patch_loader:
            print(patch['subject_key'][0], str(int(patch['count'][0])).zfill(6), patch['data'][0].shape)
            inputs = patch['data']
            
            # forward propagation only
            inputs = inputs.float().to(device)
            logits = net(inputs)
            outputs = nn.Softmax(dim=1)(logits)

            # aggregate processed patches
            patch['data'] = outputs.cpu().numpy().astype(np.uint8)
            if not one_hot_encoded:
                patch['data'] = np.expand_dims(np.argmax(patch['data'], axis=1), axis=1)
            patch_dataset.add_processed_batch(patch)
    
    results = patch_dataset.get_assembled_data()
    return results
    # todo condense to class label array
