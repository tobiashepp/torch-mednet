import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import DataLoader
from torchio import Image, ImagesDataset, transforms, INTENSITY, LABEL, Queue
from torchio.inference import GridSampler, GridAggregator
import torch
import torchio
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchio.sampler import ImageSampler
import time
import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from tqdm import trange
from utils.labelsampler import RandomLabelSampler

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




def matplotlib_imshow(img, label, pred):

    img = img.cpu().detach()
    img = img.mean(dim=0)
    slice = img.size()[2]//2
    npimg = img[:, slice, :].numpy()

    label = label.cpu().detach()
    label = label.mean(dim=0)
    slice = label.size()[2] // 2
    nplbl = label[:, slice, :].numpy()

    pred = pred.cpu().detach()
    pred = pred.mean(dim=0)
    slice = pred.size()[2] // 2
    nppred = pred[:, slice, :].numpy()

    fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(npimg, cmap="Greys")
    ax1.axis('off')
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(npimg, cmap="Greys")
    ax2.imshow(nplbl, alpha=0.7)
    ax2.axis('off')
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(nppred)
    ax3.axis('off')
    plt.tight_layout()

    return fig

def test():
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
    writer.add_figure('new', matplotlib_imshow(patch['mri']['data'],
                                               patch['label']['data'],
                                               patch['label']['data']))
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

    net = unet.model.UNet3D(in_channels=1, out_channels=1,
                            final_sigmoid=True,
                            f_maps=64)


    checkpoint = torch.load('/mnt/share/raheppt1/pytorch_models/model.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    loss = checkpoint['loss']
    net.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    grid_sampler = GridSampler(img[0,:,:,:], patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(img[0,:,:,:], patch_overlap)

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            input_tensor = patches_batch['image'].to(device)
            locations = patches_batch['location']
            logits = net(input_tensor)  # some unet
            outputs = logits
            aggregator.add_batch(logits, locations)

    output_array = aggregator.output_array
    print(output_array.shape)
    plt.imshow(output_array[:, :, 140])
    plt.show()



def get_subjects():
    work_dir = Path('/mnt/share/raheppt1/NAKO/Tho_COR')

    path_labels = list(work_dir.glob('**/*Tho_Aorta_Segm_rs.nii'))
    path_labels.sort()
    path_images = list(work_dir.glob('**/*MRA_Tho_COR_rs.nii'))
    path_images.sort()

    paths = zip(path_images, path_labels)

    subjects_list = []

    for path in paths:
        subjects_list.append([
            Image('mri', str(path[0]), INTENSITY),
            Image('label', str(path[1]), LABEL)
        ])

    return subjects_list


def train():
    writer = SummaryWriter('runs/test')

    model_path = Path('/mnt/share/raheppt1/pytorch_models')
    subjects_list = get_subjects()

    # Define transforms for data normalization and augmentation.
    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomFlip(axes=(0,)))
    #RandomAffine(scales=(0.9, 1.1), degrees=10),
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    # Random patch sampling (define the frequency for each label class).
    class PatchSampler(RandomLabelSampler):
        label_distribution = {'label': 0.5}

    # Define the dataset as a queue of patches.
    workers = range(mp.cpu_count() + 1)
    print(f'#{workers} workers available')
    queue_dataset = Queue(
        subjects_dataset,
        max_length=300,
        samples_per_volume=30,
        patch_size=(96, 96, 96),
        sampler_class=PatchSampler,
        num_workers=0,
    )
    batch_loader = DataLoader(queue_dataset, batch_size=4)

    # Check cuda device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Create model and send it to GPU.
    net = unet.model.ResidualUNet3D(in_channels=1, out_channels=1,
                                    final_sigmoid=False,
                                    f_maps=64)
    net.to(device)

    # Initialize optimizer and loss function.
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    criterion = unet.loss.DiceLossB(sigmoid_normalization=True)

    print('Training started ...')
    num_epochs = 10
    start = time.time()
    for epoch in trange(num_epochs, leave=False):
        start_time = time.time()
        running_loss = 0
        step = 0
        if epoch == 0:
        for batch in batch_loader:

            targets = batch['label']['data']
            inputs  = batch['mri']['data']

            # concat multiple labels
            #targets = torch.cat([batch['label']['data'],
            #                     batch['label']['data']],
            #                      dim=1)
            #inputs = torch.cat([batch['mri']['data'],
            #                    batch['mri']['data']],
            #                      dim=1)

            # Send data to GPU.
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = net(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step = step + 1

            interval = 10
            if step % interval == (interval-1):
                print('[%d, %4d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / interval))

                global_step = epoch * len(batch_loader) + (step + 1)

                writer.add_scalar('training loss',
                                  running_loss / interval,
                                  global_step=global_step)

                writer.add_figure('new2', matplotlib_imshow(inputs[0, ...],
                                                            targets[0, ...],
                                                            logits[0, ...]),
                                  global_step=global_step)
                running_loss = 0.0

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / step}, str(model_path.joinpath('model2.pt')))
        end_time = time.time()
        print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, running_loss / step,
                                                                   end_time - start_time))
    print('Time:', int(time.time() - start), 'seconds')
    print()


def main():
    train()

if __name__ == "__main__":
    main()