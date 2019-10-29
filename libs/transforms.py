import random
import cv2
import numpy as np
from torchvision.transforms import functional as F
import torch

from skimage import exposure


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):
        rot = 0
        sc = 0
        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]

        if isinstance(sample, list):
            sample = [self.apply(elem, rot, sc) for elem in sample]
        elif isinstance(sample, dict):
            sample = self.apply(sample, rot, sc)
        return sample

    @staticmethod
    def apply(sample, rot, sc):
        # I don't know the content of sample
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=cv2.INTER_NEAREST)

            sample[elem] = tmp

        return sample


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """
    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        if isinstance(sample, list):
            sample = [self.apply(elem, sc) for elem in sample]
        elif isinstance(sample, dict):
            sample = self.apply(sample)
        return sample

    def apply(self, sample, sc):
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

            sample[elem] = tmp

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            if isinstance(sample, list):
                sample = [self.apply(elem) for elem in sample]
            elif isinstance(sample, dict):
                sample = self.apply(sample)

        return sample

    def apply(self, sample):
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]
            tmp = cv2.flip(tmp, flipCode=1)
            sample[elem] = tmp
        return sample


class RandomVerticalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            if isinstance(sample, list):
                sample = [self.apply(elem) for elem in sample]
            elif isinstance(sample, dict):
                sample = self.apply(sample)

        return sample

    def apply(self, sample):
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]
            tmp = cv2.flip(tmp, flipCode=0)
            sample[elem] = tmp
        return sample


class Gamma(object):
    """Gamma."""
    def __init__(self, gamma = 1.5):
        self.gamma = 1.5

    def __call__(self, sample, ):

        if random.random() < 0.5:
            if isinstance(sample, list):
                sample = [self.apply(elem) for elem in sample]
            elif isinstance(sample, dict):
                sample = self.apply(sample)

        return sample

    def apply(self, sample):
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]
            tmp = exposure.adjust_gamma(tmp, self.gamma)
            sample[elem] = tmp
        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, list):
            sample = [self.apply(elem) for elem in sample]
        elif isinstance(sample, dict):
            sample = self.apply(sample)

        return sample

    def apply(self, sample):
        for elem in sample.keys():
            if 'fname' in elem:
                continue
            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = np.array(tmp, dtype=np.float32)
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        if isinstance(sample, list):
            sample = [self.apply(elem) for elem in sample]
        elif isinstance(sample, dict):
            sample = self.apply(sample)
        return sample

    def apply(self, sample):
        sample['image'] = F.normalize(sample['image'].float().div(255), self.mean, self.std)
        return sample
