import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import gc
import os
import cv2
import json
import time
import yaml
import keras
import torch
import argparse
import skimage.io
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import timm
import torch
import torch.utils.data
import torch.nn.functional as F
import pytorch_lightning as pl

import digitalpathology.image.processing.regions as dptregions
import digitalpathology.image.processing.inference as dptinference
import digitalpathology.errors.imageerrors as dptimageerrors
import digitalpathology.utils.loggers as dptloggers

from torch.utils.data import DataLoader
from skimage.util.shape import view_as_windows
from segmentation_models_pytorch import Unet, PSPNet, PAN, DeepLabV3Plus

from digitalpathology.image.io.imagereader import ImageReader
from digitalpathology.image.io.imagewriter import ImageWriter

from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from pytorch_lightning.metrics.functional.classification import auroc
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

gpu_memory = 0.25
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))


def reset_keras():
    """Resets a Keras session and clears memory."""

    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del network  # this is from global space - change this as you need
    except:
        pass

    try:
        del network_model  # this is from global space - change this as you need
    except:
        pass

    print(gc.collect())  # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    config.gpu_options.visible_device_list = '0'
    set_session(tf.Session(config=config))


def augment_image(image):
    """Augment the image with the 8 rotation/mirroring configurations."""

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


def sum_augmentations(batch_result):
    """Sum the result of augmented inference."""

    batch_result[1] = np.rot90(m=batch_result[1], k=-1, axes=(0, 1))
    batch_result[2] = np.rot90(m=batch_result[2], k=-2, axes=(0, 1))
    batch_result[3] = np.rot90(m=batch_result[3], k=-3, axes=(0, 1))
    batch_result[4] = np.fliplr(m=batch_result[4])
    batch_result[5] = np.fliplr(m=np.rot90(m=batch_result[5], k=-1, axes=(0, 1)))
    batch_result[6] = np.fliplr(m=np.rot90(m=batch_result[6], k=-2, axes=(0, 1)))
    batch_result[7] = np.fliplr(m=np.rot90(m=batch_result[7], k=-3, axes=(0, 1)))

    return np.mean(a=batch_result, axis=0)


def predict_array(self, m):
    start = time.time()

    m = m / 255.
    m = m.transpose((0, 3, 1, 2))
    m = torch.Tensor(m)  # transform to torch tensor

    dataset = torch.utils.data.TensorDataset(m)
    loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    pred = []
    for batch_idx, x in enumerate(loader):
        x = x[0].to(self.device)
        with torch.no_grad():
            y_pred, _ = self.model.forward(x)
            y_pred = y_pred.cpu().numpy().astype(np.float16)
            pred.append(y_pred)

    pred = np.concatenate(pred, axis=0)
    pred = pred.transpose((0, 2, 3, 1))
    pred = pred.astype(np.float16)
    print(f'Finished predicting in {round(time.time() - start, 1)}s')

    return pred


def detect_tissue(input_path, output_dir, model_path, config, patch_size=1024):
    file_name = input_path.split('/')[-1].split('.')[0]
    output_name = file_name + '_background.tif'
    background_path = os.path.join(output_dir, output_name)
    mask_path = None

    successful_items, failed_items = dptinference.apply_network_batch(
        job_list=[(input_path, mask_path, background_path)],
        model_path=model_path,
        patch_size=patch_size,
        output_class=1,
        number_of_classes=-1,
        normalizer='rgb_to_0-1',
        normalizer_source_range='[]',
        normalizer_target_range='[]',
        soft_mode=True,
        input_spacing=config['tissue_spacing'],
        output_spacing=config['spacing'],
        spacing_tolerance=0.25,
        input_channels=(0, 1, 2),
        confidence=0.5,
        test_augmentation=True,
        minimum_region_diagonal=config['minimum_region'],
        minimum_hole_diagonal=config['minimum_hole'],
        dilation_distance=0.0,
        full_connectivity=True,
        unrestrict_network=True,
        quantize=True,
        interpolation_order=1,
        copy_path=None,
        work_path=None,
        clear_cache=True,
        keep_intermediates=False,
        single_mode=False,
        overwrite=True,
        touch=True)

    if failed_items:
        print(f'Failed on {len(failed_items)} items:')
        for path in failed_items:
            print(f'{path}')

    # clear gpu memory
    keras.backend.clear_session()
    print(gc.collect())
    reset_keras()


def initialize_artifacts_network(config):
    num_classes = config['num_classes']
    activation = 'sigmoid' if num_classes == 1 else 'softmax2d'

    aux_params = dict(
        pooling=config['pooling'],  # one of 'avg', 'max'
        dropout=config['dropout'],  # dropout ratio, default is None
        activation=activation,  # activation function, default is None
        classes=num_classes)  # define number of output labels

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

    # prepare network pretrained weights
    assert config['architecture'] in models.keys()
    network = models[config['architecture']]
    network = torch.nn.DataParallel(network, device_ids=[0])
    network.load_state_dict(torch.load(config['artifact_network']))
    network.to(torch.device('cuda'))
    network.eval()

    return network


def initialize_quality_network(config):
    # create backbone
    model_arch = 'mobilenetv3_large_100'
    backbone = Backbone(
        model_arch='mobilenetv3_large_100',
        num_classes=1,
        pretrained=False)

    backbone.model.conv_stem = torch.nn.Conv2d(
        in_channels=10, out_channels=16,
        kernel_size=(3, 3), stride=(2, 2),
        padding=(1, 1), bias=False)

    model = QualityControlClassifier(backbone=backbone)
    model = model.load_from_checkpoint(config['quality_network'])
    return model


class SlideClassifier:
    def __init__(self, config):
        self.batch_size = 1
        self.tile_size = config['tile_size']
        self.workers = config['workers']
        self.overlap = config['overlap']
        self.half_overlap = int(round(self.overlap / 2))
        self.step_size = self.tile_size - self.overlap
        self.window_shape = (self.tile_size, self.tile_size, 3)
        self.num_classes = config['num_classes']
        self.source_shape = None
        self.image_shape = None
        self.tiles_h = None
        self.tiles_w = None

    def create_tiles(self, image):
        # add padding for creating tiles
        self.source_shape = image.shape
        tile_pad_h = self.step_size - (image.shape[0] % self.step_size)
        tile_pad_w = self.step_size - (image.shape[1] % self.step_size)
        image = np.pad(image, ((0, tile_pad_h), (0, tile_pad_w), (0, 0)), constant_values=0)

        # add padding as overlap for all tiles
        image = np.pad(image, ((self.half_overlap, self.half_overlap),
                               (self.half_overlap, self.half_overlap),
                               (0, 0)), constant_values=0)

        # create tiles
        self.image_shape = image.shape
        image = view_as_windows(image, self.window_shape, step=self.step_size)
        # image = np.squeeze(image)  # remove dimensions of size one

        # resize tiles
        self.tiles_h = image.shape[0]
        self.tiles_w = image.shape[1]
        image = image.reshape((-1, self.tile_size, self.tile_size, 3))

        return image

    def merge_tiles(self, preds):
        tiles = preds[:, self.half_overlap:-self.half_overlap, self.half_overlap:-self.half_overlap]
        tiles = tiles.reshape((self.tiles_h, self.tiles_w,
                               (self.tile_size - self.overlap),
                               (self.tile_size - self.overlap),
                               self.num_classes))
        image = np.zeros((self.image_shape[0], self.image_shape[1], self.num_classes), dtype=np.float16)

        # merge patches
        for i in range(self.tiles_h):
            for j in range(self.tiles_w):
                patch = tiles[i, j]
                a_0 = i * (self.tile_size - self.overlap)
                a_1 = (i + 1) * (self.tile_size - self.overlap)
                b_0 = j * (self.tile_size - self.overlap)
                b_1 = (j + 1) * (self.tile_size - self.overlap)
                image[a_0:a_1, b_0:b_1, :] = patch

        # cut off image to original shape
        image = image[:self.source_shape[0], :self.source_shape[1], :]
        return image


class Backbone(nn.Module):
    def __init__(self, model_arch, num_classes=1, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class QualityControlClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return torch.sigmoid(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

    def on_epoch_end(self):
        # train auc
        preds = []
        targets = []

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(torch.device('cuda'))
            y_batch = y_batch.detach().cpu().numpy()
            y_pred = torch.sigmoid(self.backbone(x_batch))
            y_pred = y_pred.detach().cpu().numpy()

            for p in y_pred:
                v = np.squeeze(p)
                preds.append(v)
            for t in y_batch:
                v = np.squeeze(np.squeeze(t))
                targets.append(v)

        targets = np.squeeze(targets).astype(np.float32)
        preds = np.squeeze(preds).astype(np.float32)

        auc = roc_auc_score(y_true=targets, y_score=preds)
        self.log('train_auc', auc)

        # valid auc
        preds = []
        targets = []

        for batch_idx, (x_batch, y_batch) in enumerate(valid_loader):
            x_batch = x_batch.to(torch.device('cuda'))
            y_batch = y_batch.detach().cpu().numpy()
            y_pred = torch.sigmoid(self.backbone(x_batch))
            y_pred = y_pred.detach().cpu().numpy()

            for p in y_pred:
                v = np.squeeze(p)
                preds.append(v)
            for t in y_batch:
                v = np.squeeze(np.squeeze(t))
                targets.append(v)

        targets = np.squeeze(targets).astype(np.float32)
        preds = np.squeeze(preds).astype(np.float32)

        auc = roc_auc_score(y_true=targets, y_score=preds)
        self.log('valid_auc', auc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, config['max_epochs'], eta_min=1e-8, last_epoch=-1)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def crop_image(image, mask, size=(1024, 1024)):
    """Crops an image using the bounding box of the segmentation mask."""

    y_min = y_max = x_min = x_max = None
    valid_bbox = False

    try:
        # find bounding box coordinates
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        valid_bbox = True
    except IndexError:
        pass

    if valid_bbox:
        # crop the image, mask, and background
        image = image[y_min:y_max + 1, x_min:x_max + 1]
        mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        # print('Size:', str((y_max-y_min, x_max-x_min)))

    width = image.shape[1]
    height = image.shape[0]
    max_length = max(width, height)
    y_pad = max_length - height
    x_pad = max_length - width

    # add padding as overlap for all tiles
    image = np.pad(image, ((0, y_pad), (0, x_pad), (0, 0)), constant_values=255)  # try out black
    mask = np.pad(mask, ((0, y_pad), (0, x_pad)), constant_values=0)

    # resize the image, mask, and background
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)  # cv2 uses (w,h,c) not (h,w,c)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    return image, mask, valid_bbox


# def combine_mask_background(mask, background):
#     """Combines the artifact segmentation mask and background mask."""
#
#     num_classes = mask.shape[2]
#     mask = np.argmax(mask, axis=-1)
#     mask = [(mask == v) for v in range(num_classes)]
#     mask = [m[:, :, np.newaxis] for m in mask]
#     mask = np.concatenate(mask, axis=-1)
#     mask = mask.astype(np.uint8)
#
#     mask[background == 1] = 0
#
#     image = np.argmax(image, axis=-1)
#     image = [(image == v) for v in range(self.num_classes)]
#     image = [m[:, :, np.newaxis] for m in image]
#     image = np.concatenate(image, axis=-1)
#     image = image.astype(np.uint8)


# def stack_channel_features(mask, background):
#     label_map = {'tissue': 2, 'ink': 3, 'air': 4, 'dust': 5, 'marker': 6, 'focus': 7}
#
#     values = [v for k, v in label_map]
#     features = [(mask == v) for v in values]
#     features = features
#     mask = np.stack(masks, axis=-1).astype(np.uint8)


def prepare_input(image, mask, num_classes=7):
    image = image / 255.
    tissue = np.where(mask > 0, 1, 0)
    masks = [(mask == v) for v in range(2, num_classes + 1)]
    masks = [tissue] + masks
    masks = np.stack(masks, axis=-1)

    # stack image and mask
    x = np.concatenate((image, masks), axis=-1)
    x = x.transpose((2, 0, 1))

    # to tensor
    x = x.astype(np.float32)
    x = torch.from_numpy(x).type(torch.float32)

    return x


def main(config):
    # initialize artifact network
    artifacts_network = initialize_artifacts_network(config=config)
    print('Successfully loaded artifact segmentation network!')

    # initialize and evaluate classification network
    quality_network = initialize_quality_network(config=config)
    quality_network(torch.randn(1, 10, 1024, 1024))
    print('Successfully loaded quality score classification network!')

    # image reader settings
    spacing = config['spacing']
    input_files = config['input_files']
    output_dir = config['output_folder']
    print(f'Processing total of {len(input_files)} slides...')

    for e, file_path in enumerate(input_files):
        print(f'Processing: {file_path}')

        # run tissue segmentation network
        file_name = file_path.split('/')[-1].split('.')[0]
        artifacts_path = os.path.join(output_dir, file_name + '_artifacts.tif')
        background_path = os.path.join(output_dir, file_name + '_background.tif')
        results_path = os.path.join(output_dir, file_name + '_results.json')

        # set patch size and spacing
        image_reader = ImageReader(file_path, spacing_tolerance=0.25)
        image_level = image_reader.level(spacing)
        img_spacing = image_reader.spacings[image_level]
        image = image_reader.content(img_spacing)
        tile_size = config['tile_size']
        min_length = min(image.shape[0], image.shape[1])
        if tile_size > min_length:
            tile_size = min_length
        del image_reader, image_level, image

        # run tissue segmentation network
        detect_tissue(
            input_path=file_path,
            output_dir=output_dir,
            model_path=config['tissue_network'],
            patch_size=tile_size,
            config=config)

        # clear gpu memory
        keras.backend.clear_session()
        print(gc.collect())
        reset_keras()

        # read input image
        image_reader = ImageReader(file_path, spacing_tolerance=0.25)
        image_level = image_reader.level(spacing)
        img_spacing = image_reader.spacings[image_level]
        image = image_reader.content(img_spacing)
        print('Image shape:', image.shape)

        # read background image
        image_reader = ImageReader(background_path, spacing_tolerance=0.25)
        image_level = image_reader.level(spacing)
        img_spacing = image_reader.spacings[image_level]
        background = image_reader.content(img_spacing)
        print('Background shape:', background.shape)

        # initialize slide classifier
        clf = SlideClassifier(config=config)

        if len(background.shape) == 3 and background.shape[2] == 1:
            background_3d = np.concatenate([background] * 3, axis=-1)
        else:
            background_3d = np.copy(background)

        image_tiles = clf.create_tiles(image=image)
        background_tiles = clf.create_tiles(image=background_3d)
        background_tiles = background_tiles[:, :, :, 0].reshape((background_tiles.shape[0], -1))
        tile_indices = np.nonzero(np.max(background_tiles, axis=1))[0]
        background = background.squeeze()  # flatten array

        # keep track of predictions
        preds = []
        start = time.time()
        num_classes = config['num_classes']

        for t, array in enumerate(image_tiles):
            if t in tile_indices:
                batch = augment_image(array)
                batch = batch / 255.
                batch = batch.transpose((0, 3, 1, 2))
                batch = torch.Tensor(batch)  # transform to torch tensor

                with torch.no_grad():
                    batch = batch.to(torch.device('cuda'))
                    y_pred, _ = artifacts_network.forward(batch)
                    y_pred = y_pred.detach().cpu().numpy()

                y_pred = y_pred.transpose((0, 2, 3, 1))
                y_pred = sum_augmentations(y_pred).astype(np.float16)

            else:
                y_pred = np.zeros((array.shape[0], array.shape[1], num_classes), dtype=np.float16)

            preds.append(y_pred)

        preds = np.stack(preds, axis=0)
        duration = round(time.time() - start)
        print(f'Finished predicting in {duration}s')

        # merge individual tiles
        merged = clf.merge_tiles(preds)
        merged = np.argmax(merged, axis=-1)
        merged = merged.astype(np.uint8)

        # process artifacts
        artifacts = np.copy(merged).astype(np.uint8)
        artifacts = artifacts + 1
        artifacts = np.where(artifacts > 1, artifacts, 0).astype(np.uint8)

        # filter thresholds
        t1 = config['region_threshold']
        t2 = config['hole_threshold']

        # create binary artifact mask
        binary = np.where(artifacts > 0, 1, 0)

        filtered, _, _ = dptregions.filter_regions_array(
            input_array=binary,
            diagonal_threshold=t1,
            full_connectivity=True,
            foreground_labels=None,
            background_label=0)

        # create filtered artifact mask
        filtered = np.where(filtered == 1, artifacts, 0)

        for i in range(num_classes):
            idx = i + 1
            array = np.where(filtered == idx, 1, 0)

            filled, _, _ = dptregions.fill_holes_array(
                input_array=array,
                diagonal_threshold=t2,
                full_connectivity=True,
                foreground_labels=None,
                fill_value=1)

            filtered[(filled == 1)] = idx

        binary = np.where(filtered > 0, 1, 0)
        combined = np.where(binary == background, filtered, background).astype(np.uint8)

        # save filtered output
        writer = ImageWriter(
            image_path=artifacts_path,
            shape=(image.shape[0], image.shape[1]),
            spacing=spacing,
            dtype=np.uint8,
            coding='indexed',
            indexed_channels=1,
            compression=None,
            interpolation=None,
            tile_size=512,
            jpeg_quality=None,
            empty_value=0,
            skip_empty=None,
            cache_path=None)

        writer.fill(combined)
        writer.close()

        # crop image and mask
        image_crop, mask_crop, valid_bbox = crop_image(image, combined)
        tensor_input = prepare_input(image_crop, mask_crop, num_classes=7)
        tensor_input = torch.unsqueeze(tensor_input, dim=0)

        # calculate quality score
        output = quality_network(tensor_input)
        output = torch.squeeze(output, dim=0)
        output = output.detach().cpu().numpy()
        output = np.round(float(output), 2)
        output = {'quality_score': output}

        with open(results_path, 'w') as outfile:
            json.dump(output, outfile)

        print('Saved artifacts mask to:', artifacts_path)
        print('Saved quality score to:', results_path)

        if config['remove_mask']:
            os.remove(background_path)


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='Detect artifacts in slide.')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as yaml_file:
        config_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # run training process
    main(config=config_file)
