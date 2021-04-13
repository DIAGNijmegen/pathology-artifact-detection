import segmentation_models_pytorch as smp


class ModelBuilder:
    def __init__(self):
        self.model_zoo = {
            'Unet': smp.Unet,
            'Linknet': smp.Linknet,
            'FPN': smp.FPN,
            'PSPNet': smp.PSPNet,
            'PAN': smp.PAN,
            'DeepLabV3': smp.DeepLabV3,
            'DeepLabV3Plus': smp.DeepLabV3Plus}

    def load_model(self, architecture, encoder_name, encoder_weights, num_classes, activation, aux_params=None):
        """Loads one of the available segmentation models."""

        assert architecture in self.model_zoo.keys(), f'Architecture {architecture} not available in model zoo.'

        # TODO: Implement auxiliary classification output.

        model = self.model_zoo[architecture](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=activation,
            aux_params=None)

        return model


import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pl_examples import cli_lightning_logo
from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule


class Backbone(torch.nn.Module):
    """
    >>> Backbone()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Backbone(
      (l1): Linear(...)
      (l2): Linear(...)
    )
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class SlideClassifier(pl.LightningModule):

    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
