import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import DataLoader
from torchio.data.images import Image, ImagesDataset
from torchio.transforms import transform
from torchio import INTENSITY, LABEL
from torchio.data.queue import Queue
from torchio.data.inference import GridAggregator, GridSampler
import torch
import torchio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchio.data.sampler import ImageSampler
import time
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
from torchio.data.sampler.label import RandomLabelSampler, LabelSampler


from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)
from torchio.utils import create_dummy_dataset
import torch.nn as nn
import numpy as np
import torchvision
from torchvision.transforms import Compose
import unet.model
import unet.loss
import yaml


def matplotlib_imshow(inputs, outputs, labels):

    # todo move to utils.misc
    # Tensor format: BCHWD
    num_plots = inputs.size()[1] + outputs.size()[1] + labels.size()[1]
    fig, ax = plt.subplots(1, num_plots)

    def _subplot_slice(n, img, title, cmap='gray', vmin=0.0, vmax=1.0):
        img = img.cpu().detach()
        # Select the slice in the middle ox the patch.
        #slice = img.size()[2] // 2
        #npimg = img[:, :, slice].numpy()
        npimg = img[:, :, :].numpy()
        npimg = np.mean(npimg, axis=2)
        ax[n].imshow(npimg, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[n].axis('off')
        ax[n].set_title(title)

    # Apply softmax activation.
    normalization = nn.Softmax(dim=1)
    outputs = normalization(outputs)

    i = 0
    for k in range(inputs.size()[1]):
        _subplot_slice(i, inputs[0, k, ...], cmap='gray', title=f'input {k}', vmin=-3.0, vmax=3.0)
        i = i + 1

    for k in range(labels.size()[1]):
        _subplot_slice(i, labels[0, k, ...], cmap='viridis', title=f'label {k}', vmin=0.0, vmax=1.0)
        i = i + 1

    for k in range(outputs.size()[1]):
        _subplot_slice(i, outputs[0, k, ...], cmap='viridis', title=f'output {k}', vmin=0.0, vmax=1.0)
        i = i + 1

    plt.tight_layout()

    return fig

# class AttrDict(dict):
#     def __init__(self, *args, **kwargs):
#         super(AttrDict, self).__init__(*args, **kwargs)
#         self.__dict__ = self
#
#
# if __name__ == '__main__':
#     config = AttrDict()
#     config.learning_rate = 0.0001
#     config.cv = 'CA'
#     config.method = ''
#     config.exp_name = 'exp_1'
#     config.batch_size = 8
#     config.max_iter = 10000
#     config.image_size = [124] * 3
#     config.image_spacing = [1] * 3
#
#     config.test_iter = 100
#     config.disp_iter = 10
#     config.snapshot_iter = 1000

def _extract_tensors(batch):

    # todo move definitions elsewhere
    img_names = ['mri']
    label_names = ['label', 'label2']

    # Concat inputs and targets along the channel dimensions
    targets = torch.cat([batch[key]['data'] for key in label_names], dim=1)
    inputs = torch.cat([batch[key]['data'] for key in img_names], dim=1)

    # Add background label as first channel.
    targets_background = torch.max(targets, dim=1, keepdim=True)[0]
    targets_background = (-1) * (targets_background - 1)
    targets = torch.cat([targets_background, targets], dim=1)

    return inputs, targets

def _get_subjects():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/')

    path_labels = list(work_dir.glob('**/*liver_3mm_low*'))
    path_labels.sort()
    path_labels2 = list(work_dir.glob('**/*spleen_3mm_low*'))
    path_labels2.sort()
    path_images = list(work_dir.glob('**/*ct_3mm_low*'))
    path_images.sort()

    paths = zip(path_images, path_labels, path_labels2)

    subjects_list = []

    for path in paths:
        subjects_list.append([
            Image('mri', str(path[0]), INTENSITY),
            Image('label', str(path[1]), LABEL),
            Image('label2', str(path[2]), LABEL)
        ])
    return subjects_list

def _create_datasets():
    subjects_list = _get_subjects()

    # Define transforms for data normalization and augmentation.
    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomFlip(axes=(0,)))
        #RandomAffine(scales=(0.9, 1.1), degrees=10))
    transform = Compose(transforms)

    # Define datasets.
    train_dataset = ImagesDataset(subjects_list[:50], transform)
    eval_dataset = ImagesDataset(subjects_list[50:], transform=ZNormalization())

    return train_dataset, eval_dataset


# Random patch sampling (define the frequency for each label class).
class PatchSampler(RandomLabelSampler):
    label_distribution = {'label': 0.3, 'label2': 0.6}


def train():
    config = {}
    config['print_interval'] = 10
    config['log_dir'] = './runs'
    config['patch_size'] = (96, 96, 96)
    config['max_epochs'] = 300
    config['learning_rate'] = 0.0001
    config['model_path'] = '/mnt/share/raheppt1/pytorch_models'
    config['train'] = {'batch_size': 4, 'queue': {}}
    config['train']['queue'] = {'max_length': 300,
                                'samples_per_volume': 30,
                                'num_workers': 0,
                                'shuffle_subjects': False,
                                'shuffle_patches': True}

    config['eval'] = {'batch_size': 4,
                      'queue': {}}
    config['eval']['queue'] = {'max_length': 300,
                               'samples_per_volume': 30,
                               'num_workers': 0,
                               'shuffle_subjects': False,
                               'shuffle_patches': True}

    config['model'] = {'in_channels': 1,
                       'out_channels': 3,
                       'fmaps': 64}

    with open("./config.yaml", "w") as f:
        yaml.dump(config, f)

    model_path = Path(config['model_path'])
    patch_size = config['patch_size']
    print_interval = config['print_interval']
    log_dir = Path(config['log_dir'])

    # Initialize tensorboard writer.
    import datetime
    ts = datetime.datetime.now().timestamp()
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    log_dir = str(log_dir.joinpath('log_' + readable))
    writer = SummaryWriter(log_dir)

    # Create training and evaluation datasets.
    train_dataset, eval_dataset = _create_datasets()

    # todo move to _create datasets
    # Define the dataset as a queue of patches.
    train_queue = Queue(
        train_dataset,
        max_length=config['train']['queue']['max_length'],
        samples_per_volume=config['train']['queue']['samples_per_volume'],
        patch_size=patch_size,
        sampler_class=PatchSampler,
        num_workers=config['train']['queue']['num_workers'],
        shuffle_subjects=config['train']['queue']['shuffle_subjects'],
        shuffle_patches=config['train']['queue']['shuffle_patches']
    )
    train_loader = DataLoader(train_queue, batch_size=config['train']['batch_size'])

    eval_queue = Queue(
        eval_dataset,
        max_length=config['eval']['queue']['max_length'],
        samples_per_volume=config['eval']['queue']['samples_per_volume'],
        patch_size=patch_size,
        sampler_class=PatchSampler,
        num_workers=config['eval']['queue']['num_workers'],
        shuffle_subjects=config['eval']['queue']['shuffle_subjects'],
        shuffle_patches=config['eval']['queue']['shuffle_patches']
    )
    eval_loader = DataLoader(eval_queue, batch_size=config['eval']['batch_size'])

    # Check cuda device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # todo restore
    b_restore = False
    if b_restore:
        print('Loading checkpoint ...')

    # Create model and send it to GPU.
    net = unet.model.ResidualUNet3D(in_channels=1, out_channels=3,
                                    final_sigmoid=False,
                                    f_maps=64)
    net.to(device)

    # Initialize optimizer and loss function.
    optimizer = torch.optim.Adam(params=net.parameters(), lr=config['learning_rate'])
    criterion = unet.loss.DiceLossB(sigmoid_normalization=False)

    max_epochs = config['max_epochs']
    finished_epochs = 0
    max_epochs = max_epochs - finished_epochs

    print(f'Training started with epoch {finished_epochs+1}...')
    start = time.time()
    for epoch in range(finished_epochs, max_epochs):
        # Initialize parameters.
        start_time = time.time()
        eval_loss_min = 0.0
        step = 0

        # Training loop ...
        net.train()
        running_loss = 0.0
        for batch in train_loader:
            # Load input and targets from batch and
            # send them to GPU.
            inputs, targets = _extract_tensors(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Training step.
            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step = step + 1

            if step % print_interval == (print_interval-1):
                # todo move to function
                print('[%d, %4d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / print_interval))

                global_step = epoch * len(train_loader) + (step + 1)

                writer.add_scalar('training loss',
                                  running_loss / print_interval,
                                  global_step=global_step)

                writer.add_figure('training_sample', matplotlib_imshow(inputs, logits, targets),
                                  global_step=global_step)
                running_loss = 0.0

        end_time = time.time()
        print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch+1, running_loss / step,
                                                                   end_time - start_time))

        # Evaluation loop.
        net.eval()
        running_loss = 0.0
        step = 0
        print('Evaluating ...')
        with torch.no_grad():
            for batch in tqdm(eval_loader):
                # Load input and targets from batch.
                inputs, targets = _extract_tensors(batch)
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Forward propagation only.
                logits = net(inputs)
                loss = criterion(logits, targets)
                running_loss += loss.item()
                step = step + 1

            eval_loss = running_loss / step
            writer.add_scalar('evaluation loss',
                              eval_loss,
                              global_step=epoch+1)

        # Save checkpoint if the current eval_loss is the lowest.
        if eval_loss < eval_loss_min:
            print('Saving new checkpoint ...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / step},
                str(model_path.joinpath('chkpt_model.pt')))

    print('Time:', int(time.time() - start), 'seconds')

def main():
    train()

if __name__ == "__main__":
    main()