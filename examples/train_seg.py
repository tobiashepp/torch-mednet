from pytorch_lightning import Trainer
from midasmednet.dataset import MedDataset, read_h5
from midasmednet.unet.loss import DiceLoss
from midasmednet.segmentation_lightning import SegmentationTrainer
from torch.utils.data import DataLoader

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform, ContrastAugmentationTransform

transform = Compose([BrightnessTransform(mu=0.0, sigma=0.3, data_key='data'),
                     GammaTransform(gamma_range=(0.7, 1.3), data_key='data'),
                     ContrastAugmentationTransform(contrast_range=(0.3, 1.7), data_key='data')])

train_ds = MedDataset('/mnt/qdata/raheppt1/data/vessel/interim/mra_train.h5',
                 ['100000'],
                 1,
                 [96, 96, 96],
                 image_group ='images',
                 label_group ='labels',
                 heatmap_group='heatmaps',
                 data_reader=read_h5)

val_ds = MedDataset('/mnt/qdata/raheppt1/data/vessel/interim/mra_train.h5',
                 ['100000'],
                 1,
                 [96, 96, 96],
                 image_group ='images',
                 label_group ='labels',
                 heatmap_group='heatmaps',
                 data_reader=read_h5)

model = SegmentationTrainer(train_ds,
                            val_ds,
                            batch_size=1,
                            num_workers=0,
                            in_channels=1,
                            out_channels=2,
                            loss_criterion=DiceLoss())

trainer = Trainer(gpus=1,
                  max_epochs=2)
trainer.fit(model)