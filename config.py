import yaml
from pathlib import Path
from torchio.data.images import Image, ImagesDataset
from torchio import INTENSITY, LABEL

# todo add verbose, define yaml file externally
class Config:
    def __init__(self, conf='./config.yaml'):

        # Data.
        self.train_dir = ''
        self.test_dir = ''
        self.images = []
        self.labels = []
        self.validation_split = 1

        # Training parameters.
        self.learning_rate = 0.0001
        self.max_epochs = 300
        self.loss = 'DICE'

        # Model configuration and path to save checkpoints.
        self.model_path = '/mnt/share/raheppt1/pytorch_models/seg'
        self.model = {'in_channels': 2,
                      'out_channels': 2,
                      'f_maps': 64}

        # Tensorboard logging.
        self.run = ''
        self.log_dir = './runs'
        self.print_interval = 10

        # Dataloading settings.
        self.patch_size = (96, 96, 96)
        self.train_batchsize = 4
        self.eval_batchsize = 4
        self.queue = {'max_length': 0,
                      'samples_per_volume': 0,
                      'num_workers': 0,
                      'shuffle_subjects': False,
                      'shuffle_patches': True}

        self._parse_from_yaml(conf)

    def parse_subjects(self, work_dir):

        # Parse images and labels.
        work_dir = Path(str(work_dir))
        paths = []
        num_subjects = set()
        names = {'images': [], 'labels': []}
        label_distribution = {}
        for img in self.images:
            # Collect image names.
            names['images'].append(img['name'])
            # Get sorted file list and append to paths.
            files = [str(file) for file in work_dir.glob(img['pattern'])]
            files.sort()
            num_subjects.add(len(files))
            paths.append([{'name': img['name'], 'type': INTENSITY, 'path': file}
                          for file in files])

        # Convert label probablities (e.g. label0 0.5 label 1 0.3) to
        # cumulative probabilites (e.g. 0.5 and 0.8). So uniform random
        # samples from [0,1) can be used to selected the label name.
        prob = 0
        for lbl in self.labels:
            # Collect label names and probabilites to sample from this label.
            names['labels'].append(lbl['name'])
            if 'prob' in lbl.keys():
                prob += lbl['prob']
            label_distribution[lbl['name']] = prob
            # Get sorted file list and append to paths.
            files = [str(file) for file in work_dir.glob(lbl['pattern'])]
            files.sort()
            num_subjects.add(len(files))
            paths.append([{'name': lbl['name'], 'type': LABEL, 'path': file}
                          for file in files])

        # All image, label lists must have the same size.
        assert len(num_subjects) == 1
        num_subjects = next(iter(num_subjects))

        # Create subjects_list of torchio.Image objects.
        subjects_list = []
        for idx in range(num_subjects):
            subject_files = []
            for k in range(len(paths)):
                subject_files.append(Image(paths[k][idx]['name'],
                                           paths[k][idx]['path'],
                                           paths[k][idx]['type']))
            subjects_list.append(subject_files)

        return {'subjects_list': subjects_list,
                'names': names,
                'label_distribution': label_distribution}

    def _parse_from_yaml(self, config_path):
        '''
        Read yaml config file to extract configuration parameters.
        :param config_path:
        :return:
        '''
        with open(config_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

            # Data directories.
            if 'train_dir' in data.keys():
                self.train_dir = data['train_dir']
            if 'test_dir' in data.keys():
                self.test_dir = data['test_dir']

            # Validation split.
            if 'validation_split' in data.keys():
                self.validation_split = data['validation_split']

            # Images.
            if 'images' in data.keys():
                self.images = data['images']

            # Labels.
            if 'labels' in data.keys():
                self.labels = data['labels']

            # Training parameters.
            if 'learning_rate' in data.keys():
                self.learning_rate = data['learning_rate']
            if 'max_epochs' in data.keys():
                self.max_epochs = data['max_epochs']
            if 'loss' in data.keys():
                self.loss = data['loss']

            # Model configuration and path to save checkpoints.
            if 'model_path' in data.keys():
                self.model_path = data['model_path']

            if 'model' in data.keys():
                for k, val in data['model'].items():
                    if k in self.model.keys():
                        self.model[k] = val

            # Tensorboard logging.
            if 'run' in data.keys():
                self.run = data['run']

            if 'log_dir' in data.keys():
                self.log_dir = data['log_dir']

            if 'print_interval' in data.keys():
                self.print_interval = data['print_interval']

            # Dataloading settings.
            if 'patch_size' in data.keys():
                self.patch_size = tuple(data['patch_size'])

            if 'train_batchsize' in data.keys():
                self.train_batchsize = data['train_batchsize']

            if 'eval_batchsize' in data.keys():
                self.eval_batchsize = data['eval_batchsize']

            if 'queue' in data.keys():
                for k, val in data['queue'].items():
                    if k in self.queue.keys():
                        self.queue[k] = val


def main():
    config = Config(conf='./config/ctorgan_config.yaml')
    res = config.parse_subjects(config.train_dir)
    subjects_list = res['subjects_list']
    print(res['label_distribution'])
    print(res['names'])
    dataset = ImagesDataset(subjects_list)
    for sample in dataset:
        for key, val in sample.items():
            import torchio.utils
            if torchio.utils.is_image_dict(val):
                print(val['stem'], val['data'].size())
        print()
        # todo consistency check


if __name__ == '__main__':
    main()
