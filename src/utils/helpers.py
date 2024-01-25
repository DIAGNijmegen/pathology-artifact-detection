import os
import json
import torch
import random
import numpy as np
import torch.backends.cudnn
import matplotlib.pyplot as plt
import albumentations as albu

from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import wandb
#from omegaconf import DictConfig, OmegaConf
from typing import Optional, Callable, List
from pathlib import Path

from .labels import class_labels, conversion_order


def seed_everything(seed=1234):
    """Set seed for multiple random processes."""

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def class_count(batch, num_classes):
    """Count the unique classes present in a batch."""

    batch = torch.argmax(batch, dim=1)
    device = batch.get_device() if batch.is_cuda else 'cpu'
    targets = torch.zeros(size=(batch.size()[0], num_classes), device=device)

    for idx in range(batch.size()[0]):
        mask = batch[idx]
        unique = torch.unique(mask)

        for class_value in range(num_classes):
            if class_value in unique:
                targets[idx, class_value] = 1.0

    return targets


def count_elements(array, exclude=0):
    count = np.bincount(array[array != exclude])
    return exclude if count.size == 0 else np.argmax(count)


def save_predictions(out_path, index, image, ground_truth_mask, predicted_mask):
    """Plot and save segmentation predictions."""

    titles = ['Image', 'Ground Truth Mask', 'Predicted Mask']
    images = [image, ground_truth_mask, predicted_mask]
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(zip(titles, images)):
        plt.subplot(1, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name)

        plt.imshow(image, vmin=0, vmax=6, cmap='Spectral')

    out_name = os.path.join(out_path, f'predictions_{str(index).zfill(5)}.png')
    plt.savefig(out_name)
    plt.close('all')


def read_json(fname):
    """Read a JSON file."""

    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    """Save a JSON file."""

    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """"Wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32."""
    test_transform = [
        albu.PadIfNeeded(TILE_SIZE, TILE_SIZE)]

    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Constructs preprocessing augmentation.

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)

    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor)]

    return albu.Compose(_transform)


def write_dictconfig(d, f, child: bool = False, ntab=0):
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def initialize_wandb(
    cfg,
    tags: Optional[List] = None,
    key: Optional[str] = "",
    fold = 0
):
    command = f"wandb login {key}"
    if tags == None:
        tags = []

    run = wandb.init(
        settings=wandb.Settings(start_method='fork'),
        project=cfg['project'],
        entity=cfg['username'],
        name=cfg['exp_name'] + '_fold_{}'.format(fold),
        dir=cfg['dir'],
        tags=tags,
        config=cfg
    )

    return run

