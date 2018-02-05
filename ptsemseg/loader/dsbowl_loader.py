import os
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob


class DSBowlLoader(data.Dataset):
    """MITSceneParsingBenchmarkLoader

    http://sceneparsing.csail.mit.edu/

    Data is derived from ADE20k, and can be downloaded from here:
    http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    NOTE: this loader is not designed to work with the original ADE20k dataset;
    for that you will need the ADE20kLoader

    This class can also be extended to load data for places challenge:
    https://github.com/CSAILVision/placeschallenge/tree/master/sceneparsing

    """
    def __init__(self, root, split="training", is_transform=False, img_size=512, augmentations=None):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2  # 0 is reserved for "other"
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, '*', 'images/')
        self.annotations_base = os.path.join(self.root, self.split, '*', 'masks/')

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.dirname(os.path.abspath(img_path))
        lbl_path = lbl_path.replace('images', 'masks')

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbls = []
        for lbl_mask in recursive_glob(rootdir=lbl_path, suffix='.png'):
            lbl = m.imread(lbl_mask)
            lbl = np.array(lbl, dtype=np.uint8)
            lbls.append(lbl)

        lbl = np.array(lbls)
        lbl = lbl.any(0).astype(int)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform

        :param img:
        :param lbl:
        """
        # todo input img is 4 channel
        img = img[:, :, :3]
        img = img.astype(np.float64)
        # img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCWH
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl) < self.n_classes):
            raise ValueError("Segmentation map contained invalid class values")

        # print(img.shape)
        # print(img)
        # m.imsave('t1.png', img.transpose(1, 2, 0))
        # print(lbl.shape)
        # print(lbl)
        # m.imsave('t2.png', lbl)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
