import skimage.io
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import class_labels


class ArtifactDataset(Dataset):
    """Artifact dataset class that provides slide and mask patch pairs."""

    def __init__(self, df_path, classes=None, transform=None, normalize=None, ink_filters=None):
        self.df = pd.read_csv(df_path)
        self.num_classes = 1 if len(classes) == 1 else len(classes) + 1
        self.class_values = [class_labels[c] for c in classes]
        self.normalize = normalize
        self.transform = transform
        self.ink_filters = ink_filters

    def __getitem__(self, i):
        img_path = self.df.iloc[i]['path']
        mask_path = self.df.iloc[i]['target']
        image = skimage.io.imread(img_path)
        mask = skimage.io.imread(mask_path)
        
        # set background to black or white
        if self.num_classes > 1:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            image[mask_3d == 0] = np.random.randint(2) * 255  # set invalid pixels in patches to black or white
        mask[mask == 1] = 0

        if self.ink_filters:
            # apply ink threshold on rough annotations
            maps = []

            for f in self.ink_filters:
                m = np.stack((
                    (f[0][0] <= image[:, :, 0]) & (image[:, :, 0] < f[0][1]),
                    (f[1][0] <= image[:, :, 1]) & (image[:, :, 1] < f[1][1]),
                    (f[2][0] <= image[:, :, 2]) & (image[:, :, 2] < f[2][1]),
                    (mask == class_labels['Ink'])), axis=-1)
                m = np.all(m, axis=-1)
                maps.append(m)

            ann_mask = (mask == class_labels['Ink'])
            ink_mask = np.any(np.stack(maps, axis=-1), axis=-1)
            mask[(ann_mask != ink_mask)] = 0  # set non-included ink pixels to zero

        if self.transform:  # image and mask augmentations
            sample = self.transform(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # extract binary mask for each class
        if self.num_classes > 1:
            mask[mask == 0] = 1
            masks = [(mask == v) for v in range(1, self.num_classes + 1)]
            mask = np.stack(masks, axis=-1).astype(np.float32)
        else:
            mask = np.where(mask == self.class_values[0], 1, 0)
            mask = mask[:, :, np.newaxis].astype(np.float32)

        if self.normalize:  # adopted from: https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place
            if self.normalize == 'simple':
                image = image / 255.
            elif self.normalize == 'inception':
                mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                image = image.astype(np.float32)
                image = image / 255.
                image = image - mean
                image = image * np.reciprocal(std, dtype=np.float32)
            elif self.normalize == 'imagenet':
                mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
                image = image.astype(np.float32)
                image = image - mean
                image = image * np.reciprocal(std, dtype=np.float32)
            else:
                image = image / 255.

        # reshape from HWC to CHW format
        image = image.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)

        return image, mask

    def __len__(self):
        return len(self.df)
