import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
import glob
import matplotlib.pylab as plt
import imageio
import datetime
import numpy as np
import scipy
from scipy import ndimage
import numpy.ma as ma
from pathlib import Path
#https://github.com/paul-bd/MIP-PET/blob/master/GIF_mip.py
# todo move to toolbox
#### Create MIP GIF
import matplotlib.pyplot as plt

def create_gif(filenames, duration, name, dir):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    dir = Path(str(dir))
    #output_file = name+'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    output_file = name + '.gif'
    output_file = dir.joinpath(output_file)
    imageio.mimsave(str(output_file), images, duration=duration)


def get_mip(angle, img_data, mask_data):
    ls_slice = []
    print(angle)

    vol_angle = scipy.ndimage.interpolation.rotate(img_data, angle, order=3)
    #todo nearest neighbor?
    mask_angle = scipy.ndimage.interpolation.rotate(mask_data, angle, order=3)

    MIP = np.amax(vol_angle, axis=1)
    MIP -= 1e-5
    MIP[MIP < 1e-5] = 0
    MIP = np.flipud(MIP.T)

    MIP_mask = np.amax(mask_angle, axis=1)
    MIP_mask -= 1e-5
    MIP_mask[MIP_mask < 1e-5] = 0
    MIP_mask = np.flipud(MIP_mask.T)

    return MIP, MIP_mask


def create_slice_gif(img_data, mask_data, name, dir, duration=0.05, cmap='Reds'):
    try:
        shutil.rmtree('test_gif/')
    except:
        pass
    os.mkdir('test_gif/')

    # todo add parameters
    # todo add axis

    assert img_data.shape == mask_data.shape
    print(img_data.shape)
    for i in range(img_data.shape[2]):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        img_slice = img_data[:, :, i]
        mask_slice = mask_data[:, :, i]

        mask_slice = ma.masked_array(mask_slice, mask=mask_slice < 0.5)
        plt.imshow(img_slice, cmap='Greys', vmin=0, vmax=5)
        plt.imshow(mask_slice, cmap=cmap, vmin=0, vmax=1.0, alpha=0.6)
        fig.savefig('test_gif/MIP' + '%04d' % (i) + '.png')
        plt.close(fig)

    filenames = glob.glob('test_gif/*.png')

    create_gif(filenames, duration, name, dir)
    try:
        shutil.rmtree('test_gif/')
    except:
        pass

def create_mipGIF_from_3D(img_data, mask_data, name, dir, nb_image=18, duration=0.15, is_mask=False, borne_max=None,
                          cmap='Reds'):
    ls_mip = []
    ls_mip_mask = []
    #img_data = img.get_data()
    img_data += 1e-5
    mask_data += 1e-5
    print('starting')
    #for angle in np.linspace(0, 360, nb_image):

    from joblib import Parallel, delayed
    import multiprocessing
    #
    # # what are your inputs, and what operation do you want to
    # # perform on each input. For example...
    inputs = np.linspace(0, 360, nb_image, endpoint=False)
    #
    # def processInput(i):
    #     return i * i
    #
    num_cores = multiprocessing.cpu_count()
    #
    #process_angle = lambda x: get_mip(x, img_data, mask_data)
    results = Parallel(n_jobs=num_cores)(delayed(get_mip)(angle, img_data, mask_data) for angle in inputs)

    try:
        shutil.rmtree('test_gif/')
    except:
        pass
    os.mkdir('test_gif/')

    ls_image = []
    for i, res in enumerate(results):
        mip = res[0]
        mip_mask = res[1]
        fig, ax = plt.subplots()
        ax.set_axis_off()
        if borne_max is None:
            if is_mask == True:
                borne_max = 1
            else:
                borne_max = 10


        import numpy.ma as ma
        mip_masked = ma.masked_array(mip_mask, mask=mip_mask < 0.5)

        plt.imshow(mip, cmap='Greys',vmin=0, vmax=borne_max)
        plt.imshow(mip_masked, cmap=cmap,vmin=0, vmax=1.0, alpha=0.6)
        fig.savefig('test_gif/MIP' + '%04d' % (i) + '.png')
        plt.close(fig)

    filenames = glob.glob('test_gif/*.png')

    create_gif(filenames, duration, name, dir)
    try:
        shutil.rmtree('test_gif/')
    except:
        pass