


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