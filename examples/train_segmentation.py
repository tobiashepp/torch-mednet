from midasmednet.segmentation import SegmentationTrainer
from sacred import Experiment
from sacred.observers import MongoObserver
import midasmednet.dataset

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform, ContrastAugmentationTransform

ex = Experiment('train_segmentation')
ex.observers.append(MongoObserver(db_name='mednet'))
ex.add_config(
    '/home/raheppt1/projects/mednet/config/aortath_segmentation.yaml')

@ex.config
def segmentation_config():
    data_reader = midasmednet.dataset.read_zarr
    transform = Compose([BrightnessTransform(mu=0.0, sigma=0.3, data_key='data'),
                         GammaTransform(gamma_range=(0.7, 1.3), data_key='data'),
                         ContrastAugmentationTransform(contrast_range=(0.3, 1.7), data_key='data')])

@ex.capture
def start_segmentation(run_name,
                        log_dir,
                        model_path,
                        print_interval,
                        max_epochs,
                        learning_rate,
                        data_path,
                        training_subject_keys,
                        validation_subject_keys,
                        image_group,
                        label_group,
                        samples_per_subject,
                        class_probabilities,
                        patch_size, batch_size,
                        num_workers,
                        in_channels,
                        out_channels,
                        f_maps,
                        data_reader,
                       transform,
                        _run):

    trainer = SegmentationTrainer(run_name,
                                log_dir,
                                model_path,
                                print_interval,
                                max_epochs,
                                learning_rate,
                                data_path,
                                training_subject_keys,
                                validation_subject_keys,
                                image_group,
                                label_group,
                                samples_per_subject,
                                class_probabilities,
                                patch_size, batch_size,
                                num_workers,
                                in_channels,
                                out_channels,
                                f_maps,
                                data_reader,
                                  transform=transform,
                                _run=_run)
    trainer.run()

@ex.automain
def main():
    start_segmentation()
    
