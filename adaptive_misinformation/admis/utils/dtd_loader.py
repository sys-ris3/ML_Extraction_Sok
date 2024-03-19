import os.path as osp
from torchvision.datasets.folder import ImageFolder
import knockoff.config as cfg

class DTD(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'dtd')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'images'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))
