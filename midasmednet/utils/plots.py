import os
import nibabel as nib
import numpy as np
import shutil
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import datetime
import numpy as np
import scipy
from scipy import ndimage
import numpy.ma as ma
from pathlib import Path
import torch
from torchvision.utils import make_grid


# TODO subplots for channels
def vis_logimages(inputs,
                  steps=5):
    """Generate grid of sample slices for input images.

    Args:
        inputs (np.array): CxHxWxD array
        steps (int): Number of plotted slices. Defaults to 5.
    Returns:
        fig, ax: matplolib plot
    """
    # images
    channels = inputs.shape[0]
    num_slices = inputs.shape[2]
    grid = np.concatenate([np.stack([inputs[c, :, idx, :] 
                        for idx in range(0, num_slices, num_slices//steps)],axis=0)
                        for c in range(channels)], axis=0)
    grid = np.expand_dims(grid, axis=1)
    grid_img = make_grid(torch.tensor(grid), nrow=steps)
    fig, ax = plt.subplots()
    ax.imshow(grid_img[0, ...], cmap='gray')# vmin=0.0, vmax=1.0)
    ax.axis('off')
    return fig, ax


def vis_loglabels(labels,
                  pred_class,
                  mip_axis=1,
                  inputs=None,
                  alpha=0.3,
                  projection_type='mean'):
    """Generate grid of sample slices for labels.

    Args:
        labels (np.array): HxWxD array, ground truth class labels.
        pred_class (np.array): HxWxD array, predicted class.
        mip_axis (int): maximum projection axis. Defaults to 1.
        inputs (np.array): HxWxD array (input image). Defaults to None.
        alpha (float): Overlay transparency. Defaults to 0.3.
        projection_type (str): 'mean' or 'max' projection.
    Returns:
        fig, ax: matplolib plot
    """
    # labels
    fig, ax = plt.subplots()
    grid = np.stack([np.max(pred_class, axis=mip_axis),
                     np.max(labels,    axis=mip_axis)], axis=0)
    grid = grid[:, np.newaxis, ...]
    grid_mask = make_grid(torch.tensor(grid))

    if inputs is not None:
        assert projection_type in ['mean', 'max']
        if projection_type == 'mean':
            mip = inputs.mean(axis=mip_axis)
        elif projection_type == 'max':
            mip = inputs.max(axis=mip_axis)
        grid_bg = np.stack(2*[mip], axis=0)
        grid_bg = grid_bg[:, np.newaxis, ...]
        grid_bg = make_grid(torch.tensor(grid_bg))
        ax.imshow(grid_bg[0, ...], cmap='gray')# vmin=0.0, vmax=1.0)
        ax.imshow(np.ma.array(grid_mask[0, ...], 
                  mask=(grid_mask[0, ...]==0)),
                  cmap='tab10', vmin=-0.1, vmax=9.9,
                  alpha=alpha)
    else:
        ax.imshow(grid_mask[0, ...], cmap='tab10', vmin=-0.1, vmax=9.9)

    ax.axis('off')
    #plt.tight_layout()
    return fig, ax
    

def vis_logheatmaps(inputs, output_heatmaps, heatmaps, mip_axis=1,
                    alpha=0.6,
                    projection_type='mean'):
    """Generate grid of MIP sample slices for labels.

    Args:
        inputs (np.array): HxWxD array
        output_heatmaps (np.array): CxHxWxD array
        heatmaps (np.array): CxHxWxD array
        alpha (float): Overlay transparency. Defaults to 0.4.
        projection_type (str): 'mean' or 'max' projection.
    Returns:
        fig, ax: matplolib plot
    """

    num_heatmaps = heatmaps.shape[0] 
    assert projection_type in ['mean', 'max']
    if projection_type == 'mean':
        mip = inputs.mean(axis=mip_axis)
    elif projection_type == 'max':
        mip = inputs.max(axis=mip_axis)

    grid_bg = np.stack(2*num_heatmaps*[mip], axis=0)
    grid_bg = np.expand_dims(grid_bg, axis=1)
    grid_bg = make_grid(torch.tensor(grid_bg), nrow=num_heatmaps)
    grid_fg = heatmaps.max(axis=mip_axis+1)
    grid_fg = np.expand_dims(grid_fg, axis=1)
    grid_fg = np.concatenate([grid_fg,
                output_heatmaps.max(axis=mip_axis+1)[:,np.newaxis,...]], axis=0)
    grid_fg = make_grid(torch.tensor(grid_fg), nrow=num_heatmaps)
    fig, ax = plt.subplots()
    ax.imshow(grid_bg[0,...], cmap='bone', vmin=0.0, vmax=1.0)
    ax.imshow(grid_fg[0,...], cmap='inferno', vmin=0.0, vmax=255.0, alpha=alpha)
    ax.axis('off')
    plt.tight_layout()
    return fig, ax

