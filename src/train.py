import os
import yaml
import torch
import shutil
import argparse
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, compute_roc_auc
from monai.networks.utils import one_hot

from segmentation_models_pytorch import Unet, PSPNet, PAN, DeepLabV3Plus
from segmentation_models_pytorch.utils.metrics import Fscore, IoU, Accuracy

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.optim import Adam, AdamW, RMSprop
from torch.nn import BCELoss

from sklearn.metrics import f1_score

from data import ArtifactDataset, load_train_transform, load_valid_transform
from utils import seed_everything, save_predictions, initialize_wandb
from trainer import Trainer


def main(cfg):
    """Runs main training procedure."""

    # fix random seeds for reproducibility
    seed_everything(seed=cfg['seed'])

    if cfg['enable_wandb']:
        key = os.environ.get("WANDB_API_KEY")
        tracker = initialize_wandb(cfg, key=key, fold=0)
    else:
        tracker = None

    print('Preparing model and data...')
    print('Using SMP version:', smp.__version__)

    num_classes = 1 if len(cfg['classes']) == 1 else (len(cfg['classes']) + 1)
    activation = 'sigmoid' if num_classes == 1 else 'softmax2d'
    background = False if cfg['ignore_channels'] else True
    binary = True if num_classes == 1 else False
    softmax = False if num_classes == 1 else True
    sigmoid = True if num_classes == 1 else False

    aux_params = dict(
        pooling=cfg['pooling'],  # one of 'avg', 'max'
        dropout=cfg['dropout'],  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=num_classes)  # define number of output labels

    # configure model
    models = {
        'unet': Unet(
            encoder_name=cfg['encoder_name'],
            encoder_weights=cfg['encoder_weights'],
            decoder_use_batchnorm=cfg['use_batchnorm'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'pspnet': PSPNet(
            encoder_name=cfg['encoder_name'],
            encoder_weights=cfg['encoder_weights'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'pan': PAN(
            encoder_name=cfg['encoder_name'],
            encoder_weights=cfg['encoder_weights'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params),
        'deeplabv3plus': DeepLabV3Plus(
            encoder_name=cfg['encoder_name'],
            encoder_weights=cfg['encoder_weights'],
            classes=num_classes,
            activation=activation,
            aux_params=aux_params)}

    assert cfg['architecture'] in models.keys()
    model = models[cfg['architecture']]

    # configure loss
    losses = {
        'dice_loss': DiceLoss(include_background=background, softmax=softmax, sigmoid=sigmoid, batch=cfg['combine']),
        'generalized_dice': GeneralizedDiceLoss(
            include_background=background, softmax=softmax, sigmoid=sigmoid, batch=cfg['combine'])}

    assert cfg['loss'] in losses.keys()
    loss = losses[cfg['loss']]

    # configure optimizer
    optimizers = {
        'adam': Adam([dict(params=model.parameters(), lr=cfg['lr'])]),
        'adamw': AdamW([dict(params=model.parameters(), lr=cfg['lr'])]),
        'rmsprop': RMSprop([dict(params=model.parameters(), lr=cfg['lr'])])}

    assert cfg['optimizer'] in optimizers.keys()
    optimizer = optimizers[cfg['optimizer']]

    # configure metrics
    metrics = {
        'dice_score': DiceMetric(include_background=background, reduction='mean'),
        'dice_smp': Fscore(threshold=cfg['rounding'], ignore_channels=cfg['ignore_channels']),
        'iou_smp': IoU(threshold=cfg['rounding'], ignore_channels=cfg['ignore_channels']),
        'generalized_dice': GeneralizedDiceLoss(
            include_background=background, softmax=softmax, sigmoid=sigmoid, batch=cfg['combine']),
        'dice_loss': DiceLoss(include_background=background, softmax=softmax, sigmoid=sigmoid, batch=cfg['combine']),
        'cross_entropy': BCELoss(reduction='mean'),
        'accuracy': Accuracy(ignore_channels=cfg['ignore_channels'])}

    assert all(m['name'] in metrics.keys() for m in cfg['metrics'])
    metrics = [(metrics[m['name']], m['name'], m['type']) for m in cfg['metrics']]  # tuple of (metric, name, type)
    
    # configure scheduler
    schedulers = {
        'steplr': StepLR(optimizer, step_size=cfg['step_size'], gamma=0.5),
        'cosine': CosineAnnealingLR(optimizer, cfg['epochs'], eta_min=cfg['eta_min'], last_epoch=-1)}

    assert cfg['scheduler'] in schedulers.keys()
    scheduler = schedulers[cfg['scheduler']]

    # configure augmentations
    train_transform = load_train_transform(transform_type=cfg['transform'], patch_size=cfg['patch_size_train'])
    valid_transform = load_valid_transform(patch_size=cfg['patch_size_valid'])  # manually selected patch size

    train_dataset = ArtifactDataset(
        df_path=cfg['train_data'],
        classes=cfg['classes'],
        transform=train_transform,
        normalize=cfg['normalize'],
        ink_filters=cfg['ink_filters'])

    valid_dataset = ArtifactDataset(
        df_path=cfg['valid_data'],
        classes=cfg['classes'],
        transform=valid_transform,
        normalize=cfg['normalize'],
        ink_filters=cfg['ink_filters'])

    test_dataset = ArtifactDataset(
        df_path=cfg['test_data'],
        classes=cfg['classes'],
        transform=valid_transform,
        normalize=cfg['normalize'],
        ink_filters=cfg['ink_filters'])

    # load pre-sampled patch arrays
    train_image, train_mask = train_dataset[0]
    valid_image, valid_mask = valid_dataset[0]
    print('Shape of image patch', train_image.shape)
    print('Shape of mask patch', train_mask.shape)
    print('Train dataset shape:', len(train_dataset))
    print('Valid dataset shape:', len(valid_dataset))
    assert train_image.shape[1] == cfg['patch_size_train'] and train_image.shape[2] == cfg['patch_size_train']
    assert valid_image.shape[1] == cfg['patch_size_valid'] and valid_image.shape[2] == cfg['patch_size_valid']

    # save intermediate augmentations
    if cfg['eval_dir']:
        default_dataset = ArtifactDataset(
            df_path=cfg['train_data'],
            classes=cfg['classes'],
            transform=None,
            normalize=None,
            ink_filters=cfg['ink_filters'])

        transform_dataset = ArtifactDataset(
            df_path=cfg['train_data'],
            classes=cfg['classes'],
            transform=train_transform,
            normalize=None,
            ink_filters=cfg['ink_filters'])

        for idx in range(0, min(500, len(train_dataset)), 10):
            image_input, image_mask = default_dataset[idx]
            image_input = image_input.transpose((1, 2, 0)).astype(np.uint8)

            image_mask = image_mask.transpose(1, 2, 0)
            image_mask = np.argmax(image_mask, axis=2) if not binary else image_mask.squeeze()
            image_mask = image_mask.astype(np.uint8)

            image_transform, _ = transform_dataset[idx]
            image_transform = image_transform.transpose((1, 2, 0)).astype(np.uint8)

            idx_str = str(idx).zfill(3)
            skimage.io.imsave(
                os.path.join(
                    cfg['eval_dir'],
                    f'{idx_str}a_image_input.png'),
                image_input,
                check_contrast=False)
            plt.imsave(
                os.path.join(
                    cfg['eval_dir'],
                    f'{idx_str}b_image_mask.png'),
                image_mask,
                vmin=0,
                vmax=6,
                cmap='Spectral')
            skimage.io.imsave(
                os.path.join(
                    cfg['eval_dir'],
                    f'{idx_str}c_image_transform.png'),
                image_transform, check_contrast=False)

        del transform_dataset

    # update process
    print('Starting training...')
    print('Available GPUs for training:', torch.cuda.device_count())

    # pytorch module wrapper
    class DataParallelModule(torch.nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    # data parallel training
    if torch.cuda.device_count() > 1:
        model = DataParallelModule(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['workers'],
        shuffle=True)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg['batch_size'] / 4),
        num_workers=cfg['workers'],
        shuffle=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg['batch_size'] / 4),
        num_workers=cfg['workers'],
        shuffle=False)

    trainer = Trainer(
        model=model,
        device=cfg['device'],
        save_checkpoints=cfg['save_checkpoints'],
        checkpoint_dir=cfg['checkpoint_dir'],
        checkpoint_name=cfg['checkpoint_name'], tracker=tracker)

    trainer.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        num_classes=num_classes)

    trainer.fit(
        train_loader,
        valid_loader,
        epochs=cfg['epochs'],
        scheduler=scheduler,
        verbose=cfg['verbose'],
        loss_weight=cfg['loss_weight'],
        test_loader=test_loader,
        binary=binary)

    # validation inference
    model.load_state_dict(torch.load(os.path.join(cfg['checkpoint_dir'], cfg['checkpoint_name'])))
    model.to(cfg['device'])
    model.eval()

    # save best checkpoint to 
    #if enable_neptune:
    #    neptune.log_artifact(os.path.join(cfg['checkpoint_dir'], cfg['checkpoint_name']))

    # setup directory to save plots
    if os.path.isdir(cfg['plot_dir_valid']):
        shutil.rmtree(cfg['plot_dir_valid'])
    os.makedirs(cfg['plot_dir_valid'], exist_ok=True)

    # valid dataset without transformations and normalization for image visualization
    valid_dataset_vis = ArtifactDataset(
        df_path=cfg['valid_data'],
        classes=cfg['classes'],
        ink_filters=cfg['ink_filters'])

    # keep track of valid masks
    valid_preds = []
    valid_masks = []

    if cfg['save_checkpoints']:
        print('Predicting valid patches...')
        for n in range(len(valid_dataset)):
            image_vis = valid_dataset_vis[n][0].astype('uint8')
            image_vis = image_vis.transpose(1, 2, 0)
            image, gt_mask = valid_dataset[n]
            gt_mask = gt_mask.transpose(1, 2, 0)
            gt_mask = np.argmax(gt_mask, axis=2) if not binary else gt_mask.squeeze()
            gt_mask = gt_mask.astype(np.uint8)
            valid_masks.append(gt_mask)

            x_tensor = torch.from_numpy(image).to(cfg['device']).unsqueeze(0)
            pr_mask, _ = model.predict(x_tensor)
            pr_mask = pr_mask.squeeze(axis=0).cpu().numpy().round()
            pr_mask = pr_mask.transpose(1, 2, 0)
            pr_mask = np.argmax(pr_mask, axis=2) if not binary else pr_mask.squeeze()
            pr_mask = pr_mask.astype(np.uint8)
            valid_preds.append(pr_mask)

            save_predictions(
                out_path=cfg['plot_dir_valid'],
                index=n + 1,
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask)

    del train_dataset, valid_dataset
    del train_loader, valid_loader

    # calculate dice per class
    valid_masks = np.stack(valid_masks, axis=0)
    valid_masks = valid_masks.flatten()
    valid_preds = np.stack(valid_preds, axis=0)
    valid_preds = valid_preds.flatten()
    dice_score = f1_score(y_true=valid_masks, y_pred=valid_preds, average=None)
    #if enable_neptune:
    #    neptune.log_text('valid_dice_class', str(dice_score))
    print('Valid dice score (class):', str(dice_score))

    if cfg['evaluate_test_set']:
        print('Predicting test patches...')

        # setup directory to save plots
        if os.path.isdir(cfg['plot_dir_test']):
            shutil.rmtree(cfg['plot_dir_test'])
        os.makedirs(cfg['plot_dir_test'], exist_ok=True)

        # test dataset without transformations and normalization for image visualization
        test_dataset_vis = ArtifactDataset(
            df_path=cfg['test_data'],
            classes=cfg['classes'],
            ink_filters=cfg['ink_filters'])

        # keep track of test masks
        test_masks = []
        test_preds = []

        for n in range(len(test_dataset)):
            image_vis = test_dataset_vis[n][0].astype('uint8')
            image_vis = image_vis.transpose(1, 2, 0)
            image, gt_mask = test_dataset[n]
            gt_mask = gt_mask.transpose(1, 2, 0)
            gt_mask = np.argmax(gt_mask, axis=2) if not binary else gt_mask.squeeze()
            gt_mask = gt_mask.astype(np.uint8)
            test_masks.append(gt_mask)

            x_tensor = torch.from_numpy(image).to(cfg['device']).unsqueeze(0)
            pr_mask, _ = model.predict(x_tensor)
            pr_mask = pr_mask.squeeze(axis=0).cpu().numpy().round()
            pr_mask = pr_mask.transpose(1, 2, 0)
            pr_mask = np.argmax(pr_mask, axis=2) if not binary else pr_mask.squeeze()
            pr_mask = pr_mask.astype(np.uint8)
            test_preds.append(pr_mask)

            save_predictions(
                out_path=cfg['plot_dir_test'],
                index=n + 1,
                image=image_vis,
                ground_truth_mask=gt_mask,
                predicted_mask=pr_mask)

        # calculate dice per class
        test_masks = np.stack(test_masks, axis=0)
        test_masks = test_masks.flatten()
        test_preds = np.stack(test_preds, axis=0)
        test_preds = test_preds.flatten()
        dice_score = f1_score(y_true=test_masks, y_pred=test_preds, average=None)
        #if enable_neptune:
        #    neptune.log_text('test_dice_class', str({dice_score}))
        print('Test dice score (class):', str(dice_score))

    if cfg['enable_wandb']:
        tracker.finish()

    # end of training process
    print('Finished training!')


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='Train segmentation model.')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args = parser.parse_args()

    # load config file
    with open(args.config, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # run training process
    main(cfg=config)
    
