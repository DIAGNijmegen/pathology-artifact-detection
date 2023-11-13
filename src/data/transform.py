import numpy as np
import albumentations as A


def load_train_transform(transform_type, patch_size=512):
    if transform_type == 'basic':
        transform = A.Compose([
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.PadIfNeeded(patch_size, patch_size)])

    elif transform_type == 'light':
        transform = A.Compose([
            # rotate image
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),

            # apply random resized crop
            A.RandomSizedCrop(min_max_height=(256, 768), height=patch_size, width=patch_size, interpolation=2, p=1.0),
            A.PadIfNeeded(patch_size, patch_size),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=0,
                p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.15, p=1.0),
                A.RandomGamma(p=.0)], p=1.0),

            A.PadIfNeeded(patch_size, patch_size)])

    elif transform_type == 'medium':
        transform = A.Compose([
            # rotate image
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),

            # apply random resized crop
            A.RandomSizedCrop(min_max_height=(256, 768), height=patch_size, width=patch_size, interpolation=2, p=1.0),
            A.PadIfNeeded(patch_size, patch_size),

            A.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
            A.CoarseDropout(max_holes=12, max_height=12, max_width=12, fill_value=0, p=0.25),
            A.RandomResizedCrop(patch_size, patch_size, scale=(0.5, 1.5), ratio=(0.5, 1.5), interpolation=1, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),

            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=20,
                val_shift_limit=5,
                p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.0, p=0.75),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=0.75),
                A.RandomGamma(gamma_limit=(75, 125), p=0.5)], p=1.0),

            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.75),
                A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.75),
                A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=0, p=0.75)], p=1.0),
        ])

    elif transform_type == 'moderate':
        transform = A.Compose([
            # rotate image
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),

            # apply random resized crop
            A.RandomSizedCrop(min_max_height=(256, 768), height=patch_size, width=patch_size, interpolation=2, p=1.0),
            A.PadIfNeeded(patch_size, patch_size),

            A.OneOf([
                A.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=1.0)], p=0.25),

            A.OneOf([
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1.0)], p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
                A.RandomGamma(p=.0)], p=0.5),

            A.OneOf([
                A.ElasticTransform(
                    alpha=200,
                    sigma=20,
                    alpha_affine=20,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    mask_value=0,
                    p=1.0),
                A.GridDistortion(
                    num_steps=10,
                    distort_limit=0.3,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    mask_value=0,
                    p=1.0)], p=0.5),

            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=0, p=1.0)], p=0.5),
        ])

    elif transform_type == 'artifact':
        transform = A.Compose([
            # rotate image
            A.RandomRotate90(p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.25),

            A.OneOf([
                A.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
                A.CoarseDropout(max_holes=8, max_height=20, max_width=20, fill_value=0, p=1.0)], p=0.25),

            A.OneOf([
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=1.0)], p=0.5),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.2, p=1.0),
                A.RandomGamma(p=.0)], p=0.5),

            A.OneOf([
                A.ElasticTransform(
                    alpha=200,
                    sigma=20,
                    alpha_affine=20,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    mask_value=0,
                    p=1.0),
                A.GridDistortion(
                    num_steps=10,
                    distort_limit=0.3,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    mask_value=0,
                    p=1.0)], p=0.5),

            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, compression_type=0, p=1.0)], p=0.5),

            A.PadIfNeeded(patch_size, patch_size)
        ])

    else:
        transform = A.Compose([A.PadIfNeeded(patch_size, patch_size)])

    return transform


def load_valid_transform(patch_size=768):
    return A.Compose([A.PadIfNeeded(patch_size, patch_size)])
