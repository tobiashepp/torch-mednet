import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def get_img_names(sample):
    """
    Extracts all the keynames of images from
    an ImageDataset sample (dict).
    :param sample: dict
    :return:
    """

    img_names = []
    for key, item in sample.items():
        if type(item) is dict:
            if 'type' in item:
                if item['type'] == 'intensity':
                    img_names.append(key)
    return img_names

def heatmap_plot(inputs, logits, heatmaps):
    inputs = inputs.cpu().detach().numpy()
    logits = logits.cpu().detach().numpy()
    heatmaps = heatmaps.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.max(inputs[0], axis=2), cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(np.max(np.sum(logits, axis=0), axis=2), cmap='coolwarm', vmin=0, vmax=255)
    ax[1].axis('off')
    ax[2].imshow(np.max(np.sum(heatmaps, axis=0), axis=2), cmap='coolwarm', vmin=0, vmax=255)
    ax[2].axis('off')
    plt.tight_layout()
    return fig


def class_plot(inputs, logits, class_target):
    normalization = nn.Softmax(dim=1)
    outputs = normalization(logits)
    out_classes = torch.argmax(outputs, dim=0)
    input = input.cpu().detach().numpy()
    out_classes = out_classes.cpu().detach().numpy()
    class_target = class_target.cpu().detach().numpy()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(np.max(inputs[0,...], axis=2), cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(np.max(out_classes, axis=2), cmap='coolwarm', vmin=0, vmax=1)
    ax[1].axis('off')
    ax[2].imshow(np.max(class_target, axis=2), cmap='coolwarm', vmin=0, vmax=1)
    ax[2].axis('off')
    plt.tight_layout()

    return fig


def matplotlib_imshow(inputs, outputs, labels):

    # todo move to utils.misc
    # Tensor format: BCHWD
    num_plots = inputs.size()[1] + outputs.size()[1]-1 + labels.size()[1]-1
    fig, ax = plt.subplots(1, num_plots)

    def _subplot_slice(n, img, title='', cmap='grey', style='mean', use_min_max=False,
                       vmin=0.0, vmax=1.0):
        img = img.cpu().detach()
        # Select the slice in the middle ox the patch.
        #slice = img.size()[2] // 2
        #npimg = img[:, :, slice].numpy()
        npimg = img[:, :, :].numpy()
        if style == 'mean':
            npimg = np.mean(npimg, axis=2)
        else:
            npimg = np.max(npimg, axis=2)
        if use_min_max:
            ax[n].imshow(npimg, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax[n].imshow(npimg, cmap=cmap)
        ax[n].axis('off')
        ax[n].set_title(title)

    # Apply softmax activation.
    normalization = nn.Softmax(dim=1)
    outputs = normalization(outputs)

    i = 0
    for k in range(inputs.size()[1]):
        style = 'max'
        _subplot_slice(i, inputs[0, k, ...], cmap='gray', title=f'input{k}', style=style)
        i = i + 1

    for k in range(1, labels.size()[1]):
        style = 'mean'
        _subplot_slice(i, labels[0, k, ...], cmap='inferno', title=f'label{k}', style=style, use_min_max=True, vmin=0, vmax=0.1)
        i = i + 1

    for k in range(1, outputs.size()[1]):
        style = 'mean'
        _subplot_slice(i, outputs[0, k, ...], cmap='inferno', title=f'output{k}', style=style, use_min_max=True, vmin=0, vmax=0.1)
        i = i + 1

    plt.tight_layout()

    return fig
