from midasmednet.landmarks import LandmarkTrainer
from sacred import Experiment
import midasmednet.dataset
ex = Experiment('landmark_detection')
ex.add_config('/home/raheppt1/projects/mednet/config/aortath_landmarks.yaml')


@ex.config
def landmark_config():
    run_name = 'test_run'
    data_reader = midasmednet.dataset.read_h5


@ex.automain
def main(run_name,
         log_dir,
         model_path,
         print_interval,
         max_epochs,
         learning_rate,
         data_path,
         training_subject_keys,
         validation_subject_keys,
         image_group,
         heatmap_group,
         samples_per_subject,
         class_probabilities,
         patch_size, batch_size,
         num_workers,
         in_channels,
         out_channels,
         f_maps,
         heatmap_treshold,
         heatmap_num_workers,
         data_reader):

    trainer = LandmarkTrainer(run_name,
                              log_dir,
                              model_path,
                              print_interval,
                              max_epochs,
                              learning_rate,
                              data_path,
                              training_subject_keys,
                              validation_subject_keys,
                              image_group,
                              heatmap_group,
                              samples_per_subject,
                              class_probabilities,
                              patch_size, batch_size,
                              num_workers,
                              in_channels,
                              out_channels,
                              f_maps,
                              heatmap_treshold,
                              heatmap_num_workers,
                              data_reader)

    trainer.run()
