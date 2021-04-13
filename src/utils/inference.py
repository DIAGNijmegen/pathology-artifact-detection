from typing import Callable, List, Optional, Sequence, Tuple, Union

import os
import cv2
import yaml
import numpy as np

import torch
import torch.nn.functional as F

from segmentation_models_pytorch import Unet, PSPNet, PAN, DeepLabV3Plus


def _augment_image(self, image):
    """
    Augment the image with the 8 rotation/mirroring configurations.
    """

    batch_image = np.empty(shape=(8,) + image.shape, dtype=image.dtype)
    batch_image[0] = image
    batch_image[1] = np.rot90(m=image, k=1, axes=(0, 1))
    batch_image[2] = np.rot90(m=image, k=2, axes=(0, 1))
    batch_image[3] = np.rot90(m=image, k=3, axes=(0, 1))
    batch_image[4] = np.fliplr(m=image)
    batch_image[5] = np.rot90(m=batch_image[4], k=1, axes=(0, 1))
    batch_image[6] = np.rot90(m=batch_image[4], k=2, axes=(0, 1))
    batch_image[7] = np.rot90(m=batch_image[4], k=3, axes=(0, 1))
    return batch_image


def _sum_augmentations(self, batch_result):
    """
    Sum the result of augmented inference.
    """
    batch_result[1] = np.rot90(m=batch_result[1], k=-1, axes=(0, 1))
    batch_result[2] = np.rot90(m=batch_result[2], k=-2, axes=(0, 1))
    batch_result[3] = np.rot90(m=batch_result[3], k=-3, axes=(0, 1))
    batch_result[4] = np.fliplr(m=batch_result[4])
    batch_result[5] = np.fliplr(m=np.rot90(m=batch_result[5], k=-1, axes=(0, 1)))
    batch_result[6] = np.fliplr(m=np.rot90(m=batch_result[6], k=-2, axes=(0, 1)))
    batch_result[7] = np.fliplr(m=np.rot90(m=batch_result[7], k=-3, axes=(0, 1)))

    return scipy.stats.mstats.gmean(a=batch_result, axis=0)


def load_background_network(config_path, checkpoint_path):
    """Loads the background/tissue segmentation network."""

    # load config file
    with open(config_path, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    num_classes = 1
    activation = 'sigmoid'

    aux_params = dict(
        pooling=config['pooling'],  # one of 'avg', 'max'
        dropout=config['dropout'],  # dropout ratio, default is None
        activation=activation,  # activation function, default is None
        classes=num_classes)  # define number of output labels

    # configure model
    models = {
        'unet': Unet(
            encoder_name=config['encoder_name'],
            encoder_weights=None,
            decoder_use_batchnorm=config['use_batchnorm'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'pspnet': PSPNet(
            encoder_name=config['encoder_name'],
            encoder_weights=None,
            psp_use_batchnorm=config['use_batchnorm'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'pan': PAN(
            encoder_name=config['encoder_name'],
            encoder_weights=None,
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'deeplabv3plus': DeepLabV3Plus(
            encoder_name=config['encoder_name'],
            encoder_weights=None,
            classes=num_classes,
            activation=activation,
            aux_params=aux_params)}

    network = models[config['architecture']]
    network.load_state_dict(torch.load(checkpoint_path))
    network.eval()

    return network


def bounding_box(mask):
    """Finds the smallest bounding box for a binary mask."""

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return y_min, y_max, x_min, x_max


def crop_image(image, mask, background, size=(1024, 1024)):
    """Crops an image using the bounding box of the segmentation mask."""

    # find bounding box coordinates
    y_min, y_max, x_min, x_max = bounding_box(mask)

    # crop the image, mask, and background
    image = image[y_min:y_max+1, x_min:x_max+1]
    mask = mask[y_min:y_max+1, x_min:x_max+1]
    background = background[y_min:y_max+1, x_min:x_max+1]

    # resize the image, mask, and background
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)  # cv2 uses (w,h,c) not (h,w,c)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    background = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    return image, mask, background


def combine_mask_background(mask, background):
    """Combines the artifact segmentation mask and background mask."""

    num_classes = mask.shape[2]
    mask = np.argmax(mask, axis=-1)
    mask = [(mask == v) for v in range(num_classes)]
    mask = [m[:, :, np.newaxis] for m in mask]
    mask = np.concatenate(mask, axis=-1)
    mask = mask.astype(np.uint8)

    mask[background == 1] = 0

    image = np.argmax(image, axis=-1)
    image = [(image == v) for v in range(self.num_classes)]
    image = [m[:, :, np.newaxis] for m in image]
    image = np.concatenate(image, axis=-1)
    image = image.astype(np.uint8)


def stack_channel_features(mask, background):
    label_map = {'tissue': 2, 'ink': 3, 'air': 4, 'dust': 5, 'marker': 6, 'focus': 7}

    values = [v for k, v in label_map]
    features = [(mask == v) for v in values]
    features = features
    mask = np.stack(masks, axis=-1).astype(np.uint8)


def concat_image_mask(image, mask):
    """Concatenates the image to the segmentation mask."""
    pass