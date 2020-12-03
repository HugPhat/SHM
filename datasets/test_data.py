import os
import math

import cv2
import torch
import torch.utils.data as data

#from utils.data import make_test_dataset
import utils.data as extended_transforms

import warnings
warnings.filterwarnings('ignore')


def make_test_dataset(path, mode):
    """[summary]

    Args:
        path ([str]): [path to root dataset]

    Returns:
        [list]: [image, trimap, alpha]
    """
    if mode == 'test':
        txt = os.path.join(path, 'val.txt')
    else:
        txt = os.path.join(path, 'train.txt')
    with open(txt, 'r') as f:
        data = f.read().split('\n')
    items = []
    for each in data:
        image = os.path.join(path, 'input', each)
        mask = os.path.join(path, 'mask', each)
        trimap = os.path.join(path, 'trimap', each)
        items.append([image, trimap, mask])
    return items

class TestDataset(data.Dataset):
    def __init__(self, root, mode, transforms=None):
        self.items = make_test_dataset(root, mode)
        self.mode = mode
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_name, trimap_name, alpha_name = self.items[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        trimap = cv2.imread(trimap_name, cv2.IMREAD_GRAYSCALE)
        alpha = cv2.imread(alpha_name, cv2.IMREAD_GRAYSCALE)
        if self.transforms is not None:
            for transform in self.transforms:
                image, trimap, alpha = transform(image, trimap, alpha)

        if self.mode == 'test':
            return image_name.split('/')[-1], image, trimap, alpha
        return image, trimap, alpha


class TestDatasetDataLoader(object):
    def __init__(self, root, mode, batch_size):
        assert mode in ['pretrain_tnet', 'pretrain_mnet', 'end_to_end', 'test']

        if mode == 'pretrain_tnet':
            transforms = [
                extended_transforms.RandomPatch(400),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = TestDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        elif mode == 'pretrain_mnet':
            transforms = [
                extended_transforms.RandomPatch(320),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.TrimapToCategorical(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = TestDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        elif mode == 'end_to_end':
            transforms = [
                extended_transforms.RandomPatch(320),
                extended_transforms.RandomFlip(),
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            train_set = TestDataset(root, mode, transforms)
            self.train_loader = data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        elif mode == 'test':
            transforms = [
                extended_transforms.Normalize(),
                extended_transforms.NumpyToTensor()
            ]
            test_set = TestDataset(root, mode, transforms)
            self.test_loader = data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False, num_workers=4)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass


if __name__ == '__main__':
    data_loader = TestDataset(
        '/data/datasets', 'pretrain_tnet', 4).train_loader
    image, trimap, alpha = next(iter(data_loader))
    print(image.size(), trimap.size(), alpha.size())
    print(image)
    print(trimap)
    print(alpha)
