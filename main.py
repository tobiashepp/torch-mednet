import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import DataLoader
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Queue
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
from utils.labelsampler import RandomLabelSampler, LabelSampler

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

import utils.misc

def matplotlib_imshow(inputs, outputs, labels):

    # tensor format BCHWD

    num_plots = inputs.size()[1] + outputs.size()[1] + labels.size()[1]
    fig, ax = plt.subplots(1, num_plots)

    def _subplot_slice(n, img, title, cmap='gray'):
        img = img.cpu().detach()
        # Select the slice in the middle ox the patch.
        slice = img.size() // 2
        npimg = img[:, :, slice].numpy()
        ax[n].imshow(npimg, cmap=cmap)
        ax[n].axis('off')
        ax[n].set_title(title)

    i = 0
    for k in range(inputs.size()[1]):
        _subplot_slice(i, inputs[0, k, ...], cmap='gray', title=f'input {k}')
        i = i + 1

    for k in range(labels.size()[1]):
        _subplot_slice(i, labels[0, k, ...], cmap='inferno', title=f'label {k}')
        i = i + 1

    for k in range(outputs.size()[1]):
        _subplot_slice(i, outputs[0, k, ...], cmap='inferno', title=f'output {k}')
        i = i + 1

    plt.tight_layout()

    return fig

def test():
    return


def predict():
    subjects_list = get_subjects()

    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), degrees=10),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    img = subjects_dataset[0]['mri']['data'].numpy()

    patch_size = 96, 96, 96
    patch_overlap = 2, 2, 2
    batch_size = 4
    sample = subjects_dataset[0]
    sampler = LabelSampler(sample, patch_size)
    patch = sampler.extract_patch(sample, patch_size)

    writer = SummaryWriter('runs/test')
    #grid = torchvision.utils2.make_grid(patch['mri']['data'][0, ...].max(dim=0)[0])
    #writer.add_image('test', patch['mri']['data'][0, :, 50, :], 0, dataformats='HW')
    writer.add_figure('new', matplotlib_imshow(patch))
    writer.close()

    print()
    plt.imshow(patch['mri']['data'][0, ...].mean(dim=0))
    plt.show()
    plt.imshow(np.max(patch['label']['data'][0, ...].numpy(), axis=0))
    plt.show()

    return
    grid_sampler = GridSampler(img[0, :, :, :], patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)

    patch = next(iter(patch_loader))


    #print(patch['image'].shape)
    plt.imshow(patch['image'][0, 0, :, 50, :])
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)



    #unet = model.UNet3D(in_channels=1, out_channels=1,
    #                    final_sigmoid=True,
    #                    f_maps=64)

    # use multiple gpus
    # if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    unet = nn.DataParallel(unet)

    #unet.to(device)

    #input = patch['image'].to(device)

def evaluate():

    subjects_list = get_subjects()

    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), degrees=10),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    img = subjects_dataset[0]['mri']['data'].numpy()

    patch_size = 96, 96, 96
    patch_overlap = 4, 4, 4
    batch_size = 6
    CHANNELS_DIMENSION = 1

    net = unet.model.ResidualUNet3D(in_channels=1, out_channels=1,
                            final_sigmoid=True,
                            f_maps=64)


    checkpoint = torch.load('/mnt/share/raheppt1/pytorch_models/model2.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    loss = checkpoint['loss']
    net.eval()

    # Select GPU with CUDA_VISIBLE_DEVICES=x python main.py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    patch_overlap = 4, 4, 4
    grid_sampler = GridSampler(img[0,:,:,:], patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(img[0,:,:,:], patch_overlap)

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            input_tensor = patches_batch['image'].to(device)
            locations = patches_batch['location']
            logits = net(input_tensor)  # some unet

            sigmoid_fnc = torch.nn.Sequential(
                torch.nn.Sigmoid())
            logits = sigmoid_fnc(logits)

            #plt.imshow(logits[0, 0, :, 50, :].cpu().detach())
            #plt.show()
            aggregator.add_batch(logits, locations)

    output_array = aggregator.output_array
    print(output_array.shape)
    plt.imshow(np.max(img[0, :, :, :], axis=2), cmap='gray')
    plt.imshow(np.mean(output_array, axis=2), alpha = 0.6)
    plt.show()


def _extract_tensors(batch):

    img_names = ['mri']
    label_names = ['label', 'label2']

    targets = torch.cat([batch[key]['data'] for key in label_names], dim=1)
    inputs = torch.cat([batch[key]['data'] for key in img_names], dim=1)

    return inputs, targets


def create_datasets():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ/')

    path_labels = list(work_dir.glob('**/*liver_3mm*'))
    path_labels.sort()
    path_labels2 = list(work_dir.glob('**/*spleen_3mm*'))
    path_labels2.sort()
    path_images = list(work_dir.glob('**/*ct_3mm*'))
    path_images.sort()

    paths = zip(path_images, path_labels, path_labels2)

    subjects_list = []

    for path in paths:
        subjects_list.append([
            Image('mri', str(path[0]), INTENSITY),
            Image('label', str(path[1]), LABEL),
            Image('label2', str(path[2]), LABEL)
        ])

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
    label_distribution = {'label': 0.3, 'label2': 0.5}


def train():
    
    print_interval = 10

    log_dir = Path('runs/')

    import datetime
    ts = datetime.datetime.now().timestamp()
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    log_dir = str(log_dir.joinpath('log_' + readable))
    writer = SummaryWriter(log_dir)

    model_path = Path('/mnt/share/raheppt1/pytorch_models')

    # Create training and evaluation datasets.
    train_dataset, eval_dataset = create_datasets()

    # Define the dataset as a queue of patches.
    workers = range(mp.cpu_count() + 1)
    print(f'#{workers} workers available')
    patch_size = (96, 96, 32)

    train_queue = Queue(
        train_dataset,
        max_length=300,
        samples_per_volume=50, # 30
        patch_size=patch_size,
        sampler_class=PatchSampler,
        num_workers=0,
        shuffle_subjects=False,
        shuffle_patches=True
    )
    train_loader = DataLoader(train_queue, batch_size=4)

    eval_queue = Queue(
        eval_dataset,
        max_length=300,
        samples_per_volume=50, # 30
        patch_size=patch_size,
        sampler_class=PatchSampler,
        num_workers=0,
        shuffle_subjects=False,
        shuffle_patches=False
    )
    eval_loader = DataLoader(eval_queue, batch_size=4)

    # Check cuda device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Restore
    b_restore = False
    if b_restore:
        print('Loading checkpoint ...')

    # Create model and send it to GPU.
    net = unet.model.ResidualUNet3D(in_channels=1, out_channels=2,
                                    final_sigmoid=False,
                                    f_maps=64)
    net.to(device)

    # Initialize optimizer and loss function.
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    criterion = unet.loss.DiceLossB(sigmoid_normalization=True)

    num_epochs = 10
    finished_epochs = 0
    num_epochs = num_epochs - finished_epochs

    print(f'Training started with epoch {finished_epochs+1}...')

    start = time.time()
    for epoch in range(finished_epochs, num_epochs):
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
    print()


def main():
    train()



if __name__ == "__main__":
    main()