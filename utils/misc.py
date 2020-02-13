import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

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


def matplotlib_imshow(inputs, outputs, labels):

    # todo move to utils.misc
    # Tensor format: BCHWD
    num_plots = inputs.size()[1] + outputs.size()[1]-1 + labels.size()[1]-1
    fig, ax = plt.subplots(1, num_plots)

    def _subplot_slice(n, img, title='', cmap='gray', style='mean', use_min_max=False,
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
