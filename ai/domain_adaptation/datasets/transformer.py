from torchvision import transforms
from torch import tensor


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


class PlaceCrop(object):

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


def get_img_transformer(image_file_path=None, is_train=None, resize_size=None, crop_size=None, args=None):
    if args.digits is True:
        return get_digits_transformer(image_file_path)
    else:
        return get_office_transformer(is_train, resize_size=resize_size, crop_size=crop_size)


def get_office_transformer(is_train, resize_size=None, crop_size=None):
    if resize_size is None:
        raise ValueError('must provide `resize_size` in get_office_transformer()')
    if crop_size is None:
        raise ValueError('must provide `crop_size` in get_office_transformer()')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train is not True:  # eval mode
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
    else:  # training mode
        transformer = transforms.Compose([ResizeImage(resize_size),
                                          # transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
    return transformer


def get_digits_transformer(dataset_name):
    norm_stats = {
        # 'mnist': (tensor([0.1309, 0.1309, 0.1309]),
        #           tensor([0.2890, 0.2890, 0.2890])),
        'mnist': (tensor([0.5]),
                  tensor([0.5])),
        'usps': (tensor([0.1576, 0.1576, 0.1576]),
                 tensor([0.2327, 0.2327, 0.2327])),
        'synth': (tensor([0.4717, 0.4729, 0.4749]),
                  tensor([0.3002, 0.2990, 0.3008])),
        'synth-small': (tensor([0.4717, 0.4729, 0.4749]),
                        tensor([0.3002, 0.2990, 0.3008])),
        # 'svhn': (tensor([0.4377, 0.4438, 0.4728]),
        #          tensor([0.1923, 0.1953, 0.1904])),
        'svhn': (tensor([0.5]),
                 tensor([0.5])),
        'basic': (tensor([0, 0, 0]),
                 tensor([1, 1, 1])),
    }
    d = {
        'mnist': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.expand(3, -1, -1).clone()),
            transforms.Normalize(
                mean=norm_stats['mnist'][0],
                std=norm_stats['mnist'][1],
            )
        ]),

        'svhn': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_stats['svhn'][0],
                std=norm_stats['svhn'][1],
            )
        ])
    }
    return d[dataset_name.lower()]
