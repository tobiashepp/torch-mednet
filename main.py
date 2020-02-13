import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchio.data.sampler.label import RandomLabelSampler, LabelSampler
from torchio.data.images import Image, ImagesDataset
from torchio.data.queue import Queue
from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)

import unet.model
import unet.loss
from utils.misc import  matplotlib_imshow
from config import Config
import random
# todo reweighted loss

class Trainer:

    def __init__(self,
                 config=None,
                 b_restore=False,
                 b_shuffle_subjects=False):

        if not config:
            self.config = Config()
        else:
            self.config = config

        self.run_name = self.config.run

        # Parse subjects.
        subjects_parser = self.config.parse_subjects(self.config.train_dir)
        subjects_list = subjects_parser['subjects_list']
        if b_shuffle_subjects:
            subjects_list = random.shuffle(subjects_list)
        self.label_distribution = subjects_parser['label_distribution']
        self.img_names = subjects_parser['names']['images']
        self.label_names = subjects_parser['names']['labels']

        # Create training and evaluation datasets, queues.
        train_dataset, eval_dataset = self._create_datasets(subjects_list,
                                                            self.config.validation_split)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._generate_queues()

        # Initialize tensorboard writer.
        self.writer = self._init_writer()

        # Check cuda device.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device}')

        # Create model and send it to GPU.
        self.net = unet.model.ResidualUNet3D(in_channels=self.config.model['in_channels'],
                                             out_channels=self.config.model['out_channels'],
                                             final_sigmoid=False,
                                             f_maps=self.config.model['f_maps'])
        self.net.to(self.device)

        # Initialize optimizer and loss function.
        self.optimizer = torch.optim.Adam(params=self.net.parameters(),
                                          lr=self.config.learning_rate)
        if self.config.loss == 'CE':
            self.criterion = unet.loss.WeightedCrossEntropyLoss()
        else:
            self.criterion = unet.loss.DiceLoss(sigmoid_normalization=False)

        # Restore from checkpoint?
        self.start_epoch = 0
        self.eval_loss_min = None
        if b_restore:
            self.start_epoch, self.eval_loss_min = self._restore_model()

    @staticmethod
    def _create_datasets(subjects_list, validation_split):

        # todo add data augmentation to config
        # Define transforms for data normalization and augmentation.
        transforms = (
            ZNormalization(),
            RandomNoise(std_range=(0, 0.25)),
            RandomFlip(axes=(0,)),
            RandomAffine(scales=(0.9, 1.1), degrees=5))
        transform = Compose(transforms)

        # Define datasets.
        train_dataset = ImagesDataset(subjects_list[:validation_split],
                                      transform)
        eval_dataset = ImagesDataset(subjects_list[validation_split:],
                                     transform=ZNormalization())

        return train_dataset, eval_dataset

    def _generate_queues(self):
        """
        Create queues filled with patches from the train/eval dataset.
        Define dataloaders for training and evaluation. Parameter configuration
        can be done with self.config.
        :return:
        """

        # Random patch sampling (define the frequency for each label class)
        class PatchSampler(RandomLabelSampler):
            label_distribution = self.label_distribution

        train_queue = Queue(
            self.train_dataset,
            max_length=self.config.queue['max_length'],
            samples_per_volume=self.config.queue['samples_per_volume'],
            patch_size=self.config.patch_size,
            sampler_class=PatchSampler,
            num_workers=self.config.queue['num_workers'],
            shuffle_subjects=self.config.queue['shuffle_subjects'],
            shuffle_patches=self.config.queue['shuffle_patches']
        )
        self.train_loader = DataLoader(train_queue, batch_size=self.config.train_batchsize)

        eval_queue = Queue(
            self.eval_dataset,
            max_length=self.config.queue['max_length'],
            samples_per_volume=self.config.queue['samples_per_volume'],
            patch_size=self.config.patch_size,
            sampler_class=PatchSampler,
            num_workers=self.config.queue['num_workers'],
            shuffle_subjects=self.config.queue['shuffle_subjects'],
            shuffle_patches=self.config.queue['shuffle_patches']
        )
        self.eval_loader = DataLoader(eval_queue, batch_size=self.config.eval_batchsize)

    def _init_writer(self):
        ts = datetime.datetime.now().timestamp()
        readable = datetime.datetime.fromtimestamp(ts).isoformat()
        log_dir = Path(self.config.log_dir)
        if self.run_name:
            log_dir = log_dir.joinpath('log_' + self.run_name)
        else:
            log_dir = log_dir.joinpath('log_' + readable)
        writer = SummaryWriter(str(log_dir))
        return writer

    def _save_model(self, epoch, loss):
        print('Saving new checkpoint ...')
        model_path = self.config.model_path
        model_path = Path(model_path)
        model_path = model_path.joinpath(self.run_name+'_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss},
            str(model_path))

    def _restore_model(self):
        print('Loading checkpoint ...')
        model_path = self.config.model_path
        model_path = Path(model_path)
        model_path = model_path.joinpath(self.run_name+'_model.pt')
        checkpoint = torch.load(str(model_path))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        eval_loss_min = checkpoint['loss']
        return start_epoch, eval_loss_min

    def _extract_tensors(self, batch):
        # Concat inputs and targets along the channel dimensions
        targets = torch.cat([batch[key]['data'] for key in self.label_names], dim=1)
        inputs = torch.cat([batch[key]['data'] for key in self.img_names], dim=1)

        # Add background label as first channel.
        targets_background = torch.max(targets, dim=1, keepdim=True)[0]
        targets_background = (-1) * (targets_background - 1)
        targets = torch.cat([targets_background, targets], dim=1)
        return inputs, targets

    def _train_epoch(self, epoch):
        print_interval = self.config.print_interval

        # Training loop ...
        self.net.train()
        running_loss = 0.0
        for step, batch in enumerate(self.train_loader):
            # Load input and targets from batch and send them to GPU.
            inputs, targets = self._extract_tensors(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Training step.
            self.optimizer.zero_grad()
            logits = self.net(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if step % print_interval == (print_interval - 1):
                print('[%d, %4d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_interval))

                global_step = epoch * len(self.train_loader) + (step + 1)

                self.writer.add_scalar('Loss/train',
                                       running_loss / print_interval,
                                       global_step=global_step)

                self.writer.add_figure('Sample', matplotlib_imshow(inputs, logits, targets),
                                       global_step=global_step)
                running_loss = 0.0

    def _evaluate(self, epoch):
        running_loss = 0.0
        per_channel_dice = 0.0
        self.net.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.eval_loader)):
                # Load input and targets from batch.
                inputs, targets = self._extract_tensors(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Forward propagation only.
                logits = self.net(inputs)
                # Calculate loss.
                outputs = nn.Softmax(dim=1)(logits)
                per_channel_dice += unet.loss.compute_per_channel_dice(outputs, targets)
                # Calculate metrics.
                loss = self.criterion(logits, targets)
                running_loss += loss.item()

        # Compute mean loss/dice metrics and log to tensorboard.
        eval_loss = running_loss / len(self.eval_loader)
        self.writer.add_scalar('Loss/eval',
                               eval_loss,
                               global_step=epoch + 1)

        per_channel_dice = per_channel_dice/len(self.eval_loader)
        for k in range(per_channel_dice.size()[0]):
            self.writer.add_scalar(f'Dice/label{k}',
                                   per_channel_dice[k],
                                   global_step=epoch+1)
        return eval_loss

    def run(self):
        # Parameters
        max_epochs = self.config.max_epochs

        # Variables
        start_epoch = self.start_epoch
        eval_loss_min = self.eval_loss_min

        print(f'Training started ...')
        start = time.time()
        for epoch in range(start_epoch, max_epochs):
            start_time = time.time()

            # Train for one epoch.
            print('Train ...')
            self._train_epoch(epoch)

            # Evaluate ...
            print('Evaluate ...')
            eval_loss = self._evaluate(epoch)

            end_time = time.time()
            print("Epoch {}, time {:.2f}".format(epoch + 1, end_time - start_time))

            # Save checkpoint if the current eval_loss is the lowest.
            if not eval_loss_min:
                eval_loss_min = eval_loss
            if eval_loss < eval_loss_min or epoch == 0:
                eval_loss_min = eval_loss
                self._save_model(epoch + 1, eval_loss)

        print('Time:', int(time.time() - start), 'seconds')


def main():
    b_restore = False
    config = Config(conf='./config/ctorgan_config.yaml')
    trainer = Trainer(config, b_restore)
    trainer.run()


if __name__ == "__main__":
    main()