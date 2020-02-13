import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import argparse

import torch
import torch.nn as nn
import torchio
import torchvision
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader

from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomFlip,
    RandomAffine,
)

import torchio.utils
from torchio.data.images import Image, ImagesDataset
from torchio import INTENSITY, LABEL
from torchio.data.inference import GridAggregator, GridSampler

import unet.model
import unet.loss
from config import Config


class GridDataset(Dataset):

    def __init__(self,
                 subject,
                 patch_size,
                 patch_overlap):
        self.samplers = []
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        for key, val in subject.items():
            if torchio.utils.is_image_dict(val):
                # Create grid sampler for each input image.
                if val['type'] == INTENSITY:
                    self.samplers.append(GridSampler(val['data'][0, :, :, :],
                                                     patch_size, patch_overlap))

    def __len__(self):
        return len(self.samplers[0])

    def __getitem__(self, idx):
        image = torch.cat([self.samplers[c][idx]['image']
                           for c in range(len(self.samplers))], dim=0)
        location = self.samplers[0][idx]['location']
        return {'image': image, 'location': location}


def predict(config,
            selected_subjects,
            model_path,
            save_dir,
            batch_size=None,
            patch_overlap=(40, 40, 40)):

    # Set parameters.
    save_dir = Path(str(save_dir))
    patch_size = config.patch_size
    out_channels = config.model['out_channels']
    in_channels = config.model['in_channels']

    if not batch_size:
        batch_size = config.eval_batchsize

    # Check cuda device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    # Create model and send it to GPU.
    print(f'unet: inputs {in_channels} outpus {out_channels}')
    print(out_channels)
    net = unet.model.ResidualUNet3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    final_sigmoid=False,
                                    f_maps=config.model['f_maps'])

    # Restore checkpoint from file.
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    finished_epochs = checkpoint['epoch']
    print(f'Restoring model, epoch: {finished_epochs}')
    net.to(device)

    dice_results = {'file': [], 'dice': []}
    dataset = ImagesDataset(selected_subjects, transform=ZNormalization())
    for k, subject_data in enumerate(dataset):

        labels = []
        imgs = []
        # Save input and label files as numpy
        for key, val in subject_data.items():
            if torchio.utils.is_image_dict(val):
                if val['type'] == INTENSITY:
                    print(val['stem'], val['data'].size())
                    save_path = str(save_dir.joinpath(f'{k}_input_{key}.npy'))
                    np.save(save_path, val['data'][0, :, :, :])
                    imgs.append(val['data'][0, :, :, :].numpy())
                else:
                    save_path = str(save_dir.joinpath(f'{k}_label_{key}.npy'))
                    label = val['data'][0, :, :, :]
                    labels.append(label.numpy())
                    np.save(save_path, label)

        # Get one image as template for the output aggregator.
        for key, val in subject_data.items():
            if torchio.utils.is_image_dict(val):
                output_template = np.zeros_like(val['data'][0, :, :, :])
                out_filename = val['stem'] + '.nii.gz'
                out_affine = val['affine']
                break

        # Create aggregators for the network outputs.
        aggregators = [GridAggregator(output_template, patch_overlap)
                       for _ in range(out_channels)]

        # Inference.
        ds = GridDataset(subject_data, patch_size, patch_overlap)
        patch_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        net.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(patch_loader)):

                # Concat channels to input tensor.
                inputs = batch['image']

                # Forward propagation only.
                inputs = inputs.to(device)
                logits = net(inputs)
                outputs = nn.Softmax(dim=1)(logits)

                # Aggregate patches.
                for c in range(out_channels):
                    aggregators[c].add_batch(outputs[:, [c], :, :, :],
                                             batch['location'])

        # Aggregate output patches and store predicitions.
        dice_scores = []
        for c, agg in enumerate(aggregators):
            output_array = aggregators[c].output_array

            # Compute dice score.
            if c > 0:
                class_prediction = torch.tensor(labels[c-1].astype(np.float)[np.newaxis, np.newaxis, ...], dtype=torch.float32)
                class_target = torch.tensor(output_array.astype(np.float32)[np.newaxis, np.newaxis, ...], dtype=torch.float32)
                dice = unet.loss.compute_per_channel_dice(class_prediction, class_target)
                dice_scores.append(dice.numpy())

            # Save predicted mask to nii file.
            out_nii = nib.Nifti1Image(output_array.astype(np.int16), out_affine)
            save_path = str(save_dir.joinpath(out_filename)).replace('.nii.gz', f'{c}.nii.gz')
            nib.save(out_nii, save_path)
        print(dice_scores)
        dice_results['file'].append(out_filename)
        dice_results['dice'].append(dice_scores)

        plt.imshow(np.max(imgs[0], axis=1))
        plt.imshow(np.max(labels[0], axis=1), alpha=0.5)
        plt.imshow(np.max(output_array, axis=1), alpha=0.5)
        plt.show()

    # Save dice scores to csv file.
    df = pd.DataFrame(data=dice_results)
    df.to_csv(save_dir.joinpath('dice_results.csv'))


    return


def _get_subjects_from_config(config, selection,
                              start_subject=0, num_subjects=1):
    if selection == 'test':
        parser_results = config.parse_subjects(config.test_dir)
    elif selection in ['train', 'validate']:
        parser_results = config.parse_subjects(config.train_dir)

    subjects_list = parser_results['subjects_list']

    # Select subjects:
    if selection == 'train':
        subj_min = 0
        if num_subjects > 0:
            subj_max = min(num_subjects, config.validation_split)
        else:
            subj_max = config.validation_split
    elif selection == 'validate':
        subj_min = config.validation_split
        if num_subjects > 0:
            subj_max = min(config.validation_split + num_subjects, len(subjects_list))
        else:
            subj_max = len(subjects_list)
    elif selection == 'test':
        subj_min = 0
        if num_subjects > 0:
            subj_max = min(num_subjects, len(subjects_list))
        else:
            subj_max = len(subjects_list)

    subj_min = min(subj_max-1, subj_min+start_subject)
    print(subj_min, subj_max)
    selected_subjects = subjects_list[subj_min:subj_max]
    return selected_subjects


def main():
    # paths
    config_path = './config/ctorgan_config.yaml'
    model_path = '/mnt/share/raheppt1/pytorch_models/seg/ctorgan_model.pt'

    parser = argparse.ArgumentParser(description='Process some integers.')
    # -I pet:**/*img1.nii ct:**/*img2.nii
    parser.add_argument('-c', '--config')
    parser.add_argument('-m', '--model')
    parser.add_argument('-o', '--outdir')
    parser.add_argument('-s', '--selection', choices=['validate', 'train', 'test'])
    parser.add_argument('-I', '--Images', nargs='+')
    parser.add_argument('-L', '--Labels', nargs='+')
    parser.add_argument('-D', '--Directory')
    args = parser.parse_args()

    # Standard settings.
    start_subject = 0
    num_subjects = 0
    out_dir = './tmp'
    selection = 'validate'

    if args.outdir:
        out_dir = args.out_dir

    if args.selection:
        selection = args.selection

    if args.model:
        model_path = args.model
    if args.config:
        config_path = args.config

    # Read config file.
    config = Config(conf=config_path)

    # Overwrite parameters.
    if args.Directory:
        config.test_dir = str(args.Directory)

    if args.Images:
        config.images = []
        for tmp in args.Images:
            tmp = str(tmp).split(':')
            config.images.append({'name': tmp[0],
                                  'pattern': tmp[1]})

    if args.Labels:
        config.labels = []
        for tmp in args.Labels:
            tmp = str(tmp).split(':')
            config.labels.append({'name': tmp[0],
                                  'pattern': tmp[1]})

    print(config.labels)
    print(config.images)

    # Get filenames.
    selected_subjects = _get_subjects_from_config(config, selection,
                                                  num_subjects=num_subjects,
                                                  start_subject=start_subject)
    # Inference
    predict(config,
            selected_subjects,
            model_path,
            save_dir=out_dir)


def visualize():
    work_dir = Path('./tmp')
    pets = work_dir.glob('*pet*')
    outs = work_dir.glob('*output_1*')
    labels = work_dir.glob('*tumor*')

    #
    k = 0
    for pet, out, label in zip(pets, outs, labels):
        if k>=3:
            print(pet, out, label)
            out = np.load(out).astype(np.float32)
            pet = np.load(pet).astype(np.float32)
            label = np.load(label).astype(np.float32)
            import utils.plots
            #utils.plots.create_mipGIF_from_3D(pet, out.astype(np.float), f'{k}output', './results', cmap='Blues')
            #utils.plots.create_mipGIF_from_3D(pet, label.astype(np.float), f'{k}label', './results', cmap='Reds')
            utils.plots.create_slice_gif(pet, out.astype(np.float), f'{k}output', './results2')
            utils.plots.create_slice_gif(pet, label.astype(np.float), f'{k}label', './results2')
        k=k+1


    from array2gif import write_gif
    # ctinput.npy  out.npy  petinput.npy  tumorlabel.npy
    #plt.imshow(np.max(pet, axis=1), cmap='gray', vmin=0, vmax=10, aspect=0.66)
    a = np.max(out, axis=1).astype(np.float32)
    pet = np.max(pet, axis=1).astype(np.float32)

    print(np.sum(a<0.5))
    import numpy.ma as ma
    a_masked = ma.masked_array(a, mask=a<0.5)

    plt.imshow(pet, cmap='Greys', aspect=0.66, vmin=0, vmax=10.0)
    plt.imshow(a_masked, cmap='Reds', aspect=0.66, vmin=0, vmax=1, alpha=0.5)
    plt.show()


def test3():
    work_dir = Path('/mnt/share/raheppt1/MelanomCT_Organ')

    cts = work_dir.glob('**/*ct_3mm_low*')
    labels = work_dir.glob('**/*spleen_3mm_low*')

    for ct, label in zip(cts, labels):
        print(ct, label)
        #ct = np.load(ct).astype(np.float32)
        #label = np.load(label).astype(np.float32)

        #plt.imshow(np.max(ct, axis=1))
        #plt.imshow(np.max(label, axis=1), alpha=0.5)
        #plt.show()

        if '54' in str(ct):
            nii = nib.load(str(ct))
            ctdata = nii.get_fdata(dtype=np.float32)
            nii = nib.load(str(label))
            labeldata = nii.get_fdata(dtype=np.float32)

            plt.imshow(np.max(ctdata, axis=1))
            plt.imshow(np.max(labeldata, axis=1), alpha=0.5)
            plt.show()


def test2():
    work_dir = Path('./tmp2')
    cts = work_dir.glob('*ct*')
    outs = work_dir.glob('*output_1*')
    labels = work_dir.glob('*liver*')

    #
    k = 0
    p_list = []
    p_list_abs = []
    vol_list = []
    for ct, out, label in zip(cts, outs, labels):
        print(ct)
        print(label)
        out = np.load(out).astype(np.float32)
        ct = np.load(ct).astype(np.float32)
        label = np.load(label).astype(np.float32)

        plt.imshow(np.max(ct, axis=1))
        plt.imshow(np.max(label, axis=1), alpha=0.5)
        plt.show()


def test():
    work_dir = Path('./tmp')
    pets = work_dir.glob('*pet*')
    outs = work_dir.glob('*output_1*')
    labels = work_dir.glob('*tumor*')

    #
    k = 0
    p_list = []
    p_list_abs = []
    vol_list = []
    for pet, out, label in zip(pets, outs, labels):
        out = np.load(out).astype(np.float32)
        pet = np.load(pet).astype(np.float32)
        label = np.load(label).astype(np.float32)
        int_out = np.sum(out)
        int_ref = np.sum(label)
        vol_list.append(int_ref)
        p = (int_out-int_ref)/int_ref
        p_abs = abs(p)
        print(f'{k} out {int_out} ref {int_ref} relative{p}')
        p_list.append(p*100)
        p_list_abs.append(p_abs)


    print(np.median(p_list))
    print(np.median(p_list_abs))
    plt.scatter(vol_list, p_list)
    axes = plt.gca()
    axes.set_ylim([-150, 150])
    plt.xlabel('Vol[Voxels]')
    plt.ylabel('Diff[percentage]')
    plt.show()

if __name__ == '__main__':
    main()