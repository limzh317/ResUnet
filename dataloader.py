#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/27 下午8:29
# @Author  : chuyu zhang
# @File    : model.py
# @Software: PyCharm

from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset

import os
import sys
import glob

import ipdb
sys.path.append(os.path.abspath('./'))


class ImageLoader(Dataset):
    def __init__(self, mode, data_root_dir=None, transforms=None):
        self.mode = mode
        self.all_label = True
        self.channel3 = True
        self.transforms = transforms
        self.interval = 0

        assert self.mode in ["train", "val", "test"], "Only support train, val and test"

        image_dir = 'images_file/'
        label_dir = 'labels_file/'

        images_list = glob.glob(os.path.join(data_root_dir, image_dir, self.mode, "*.png"))
        images_list.sort()

        labels_list = glob.glob(os.path.join(data_root_dir, label_dir, self.mode, "*.png"))
        labels_list.sort()

        print(len(labels_list), len(images_list))
        assert (len(labels_list) == len(images_list))

        self.images_list = images_list
        self.labels_list = labels_list

        print('Done initializing ' + self.mode + ' Dataset')

    def __getitem__(self, index):
        # Data augment hasn't been applied.

        """
        image = Image.open(os.path.join(self.labels_list[index]))
        image = image.convert("RGB")
        label = Image.open(os.path.join(self.labels_list[index]))
        """

        image = cv2.imread(os.path.join(self.images_list[index]))  # X, Y, 3
        image = cv2.resize(image, (480, 480), cv2.INTER_CUBIC)
        label = cv2.imread(os.path.join(self.labels_list[index]), 0)
        label = cv2.resize(label, (480, 480), cv2.INTER_NEAREST)

        if not self.all_label:
            label[label == 2] = 0
            label[label == 3] = 0
            label[label == 4] = 2
        # ipdb.set_trace()
        # TODO: if apply transform to image, how about the label?
        # The same operation should do both to image and label
        # There is no need to do transpose, just pass it to transform
        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)
        else:
            sample["image"] = np.transpose(sample["image"], axes=[2, 0, 1])
        return sample

    def __len__(self):
        return len(self.images_list)

