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

from digitalpathology.image.processing.conversion import create_annotation_mask
from digitalpathology.errors.imageerrors import AnnotationOpenError

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


def convert_annotations(img_dir: str, xml_dir: str, out_dir: str, suffix: str = '_mask'):
    """Converts all image/annotation pairs in a directory to masks."""

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Find al XML files in the given directory.
    xml_files = []
    for file in os.listdir(xml_dir):
        if file.endswith('.xml'):
            xml_files.append(file)

    # Find al image files in the given directory.
    img_files = []
    for path, sub_dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(('.tif', '.mrxs')):
                img_files.append(os.path.join(path, file))

    # Save the base names of the files.
    xml_names = [file.rstrip('.xml') for file in xml_files]
    img_names = [os.path.basename(file).rstrip('.tif').rstrip('.mrxs') for file in img_files]

    # Get indices of images that have a corresponding annotation.
    matches = [i for i, x in enumerate(img_names) if x in xml_names]

    img_list = [img_files[i] for i in matches]
    xml_list = [os.path.join(xml_dir, img_names[i] + '.xml') for i in matches]
    out_list = [os.path.join(out_dir, img_names[i] + suffix + '.tif') for i in matches]

    # For each image/xml/output triplet, create the mask.
    for e, (img_path, xml_path, out_path) in enumerate(zip(img_list, xml_list, out_list)):
        print(f'Processing image: {img_path}')

        try:
            assert img_path.endswith(('.tif', '.tiff', '.mrxs'))
            assert xml_path.endswith('.xml')
            assert out_path.endswith('.tif')

            create_annotation_mask(
                image=img_path,
                annotation=xml_path,
                label_map=class_labels,
                conversion_order=conversion_order,
                conversion_spacing=None,
                spacing_tolerance=0.25,
                output_path=out_path,
                strict=True,
                accept_all_empty=True,
                work_path=None,
                clear_cache=True,
                overwrite=True)

        except AnnotationOpenError:
            print(f'AnnotationOpenError for annotation {xml_path}')

        print(f'Processed {str(e + 1).zfill(3)}/{str(len(img_list)).zfill(3)} files')


def ensure_dir(dirname):
    """Make sure that a directory exists."""

    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


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
