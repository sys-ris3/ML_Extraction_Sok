import os.path as osp
from torchvision.datasets.folder import ImageFolder
import knockoff.config as cfg

class Flowers102(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'flowers')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://www.kaggle.com/c/oxford-102-flower-pytorch/data'
            ))

        # Initialize ImageFolder
        split = 'train' if train else 'valid'
        super().__init__(root=osp.join(root, split), transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, split,
                                                                len(self.samples)))

class Flowers17(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'flowers17')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz'
            ))

        # Initialize ImageFolder
        split = 'train' if train else 'test'
        super().__init__(root=osp.join(root, split), transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, split,
                                                                len(self.samples)))
