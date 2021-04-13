import os
import time
import torch
import neptune
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.utils import one_hot
from utils import class_count


class AverageMetricTracker:
    """Tracker that keeps track of average loss/scores over all batches in an epoch."""

    # TODO: Simplify AverageMetricTracker functions and attributes.

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


class Trainer:
    """Basic class to train segmentation models."""

    def __init__(self, model, device, save_checkpoints=False, checkpoint_dir=None, checkpoint_name='checkpoint'):
        # trainer attributes
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_checkpoints = save_checkpoints
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        if not self.checkpoint_path.endswith('.pth'):
            self.checkpoint_path = self.checkpoint_path + '.pth'

        # fit attributes
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.scheduler = None
        self.epochs = None
        self.loss_weight = None
        self.accumulation_steps = None

        # compile attributes
        self.loss = None
        self.optimizer = None
        self.metrics = None
        self.weight = None
        self.num_classes = None

        # other attributes
        self.history = {'train': {}, 'valid': {}, 'test': {}}
        self.verbose = True
        self.valid_metric = None
        self.best_score = None

        # classification loss
        self.criterion = torch.nn.BCELoss(reduction='mean')

        # binary problem
        self.binary = None

    @staticmethod
    def _prepare_device():
        return torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    def _to_device(self):
        self.model.to(self.device)

    @staticmethod
    def _format_logs(logs):
        return ' - '.join([f'{k}: {round(v, 4)}' for k, v in logs.items()])

    def _update_logs(self, loss, y_pred, y_true, logs):
        # TODO: Implement log update function in train and valid epochs.
        pass

    def _show_progress(self, duration, logs, stage='train'):
        steps = len(self.train_loader) if stage.lower() == 'train' else len(self.valid_loader)

        if self.verbose:
            duration = int(round(duration, 0))
            print(f'{stage} {steps}/{steps} [==========] - {duration}s - {self._format_logs(logs)}')

    def _train_epoch(self, epoch):
        start = time.time()
        logs = {}
        loss_meter = AverageMetricTracker()
        metric_trackers = {m[1]: AverageMetricTracker() for m in self.metrics}

        self.model.train()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            # make predictions
            self.optimizer.zero_grad()
            out1, out2 = self.model.forward(x)
            classes = class_count(y, self.num_classes)

            # calculate weighted loss
            clf_loss = self.criterion(out2, classes)
            sgm_loss = self.loss(out1, y)
            loss = ((1.0 - self.loss_weight) * clf_loss) + (self.loss_weight * sgm_loss)

            loss.backward()
            self.optimizer.step()

            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {'loss': loss_meter.mean}
            logs.update(loss_logs)

            # neptune logging (train step)
            neptune.log_metric('train_loss_step', loss_value)

            for m in self.metrics:
                metric = m[0]  # unpack metric class
                metric_name = m[1]  # unpack metric name
                metric_type = m[2]  # unpack metric type

                if metric_type == 'classification':
                    metric_value = metric(out2, classes).cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                elif metric_type == 'segmentation':
                    d = out1.get_device() if out1.is_cuda else 'cpu'
                    if not self.binary:
                        y_pred = torch.argmax(out1, dim=1, keepdim=True).to(d)
                        y_pred = one_hot(y_pred, num_classes=out1.shape[1])
                    else:
                        y_pred = torch.round(out1).to(d)

                    # calculate metric
                    metric_value = metric(y_pred, y)
                    metric_value = metric_value[0][0] if isinstance(metric_value, tuple) else metric_value
                    metric_value = metric_value.cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                else:
                    raise ValueError(f'Type {metric_type} is not a valid metric type.')

                # neptune logging (train step)
                neptune.log_metric('train_' + metric_name + '_step', metric_value)

            metrics_logs = {k: v.mean for k, v in metric_trackers.items()}
            logs.update(metrics_logs)

        # neptune logging (train epoch)
        for k, v in logs.items():
            neptune.log_metric('train_' + k + '_epoch', v)

        duration = time.time() - start

        if self.verbose:
            self._show_progress(duration, logs, stage='Train')

        return logs

    def _valid_epoch(self, epoch):
        start = time.time()
        logs = {}
        loss_meter = AverageMetricTracker()
        metric_trackers = {m[1]: AverageMetricTracker() for m in self.metrics}

        self.model.eval()

        for batch_idx, (x, y) in enumerate(self.valid_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                # make predictions
                out1, out2 = self.model.forward(x)
                classes = class_count(y, self.num_classes)

                # calculate weighted loss
                clf_loss = self.criterion(out2, classes)
                sgm_loss = self.loss(out1, y)
                loss = ((1.0 - self.loss_weight) * clf_loss) + (self.loss_weight * sgm_loss)

            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {'loss': loss_meter.mean}
            logs.update(loss_logs)

            # neptune logging (valid step)
            neptune.log_metric('valid_loss_step', loss_value)

            for m in self.metrics:
                metric = m[0]  # unpack metric class
                metric_name = m[1]  # unpack metric name
                metric_type = m[2]  # unpack metric type

                if metric_type == 'classification':
                    metric_value = metric(out2, classes).cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                elif metric_type == 'segmentation':
                    d = out1.get_device() if out1.is_cuda else 'cpu'
                    if not self.binary:
                        y_pred = torch.argmax(out1, dim=1, keepdim=True).to(d)
                        y_pred = one_hot(y_pred, num_classes=out1.shape[1])
                    else:
                        y_pred = torch.round(out1).to(d)

                    # calculate metric
                    metric_value = metric(y_pred, y)
                    metric_value = metric_value[0][0] if isinstance(metric_value, tuple) else metric_value
                    metric_value = metric_value.cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                else:
                    raise ValueError(f'Type {metric_type} is not a valid metric type.')

                # neptune logging (valid step)
                neptune.log_metric('valid_' + metric_name + '_step', metric_value)

            metrics_logs = {k: v.mean for k, v in metric_trackers.items()}
            logs.update(metrics_logs)

        # neptune logging (valid epoch)
        for k, v in logs.items():
            neptune.log_metric('valid_' + k + '_epoch', v)

        duration = time.time() - start

        if self.verbose:
            self._show_progress(duration, logs, stage='Valid')

        return logs

    def _test_epoch(self, epoch):
        start = time.time()
        logs = {}
        loss_meter = AverageMetricTracker()
        metric_trackers = {m[1]: AverageMetricTracker() for m in self.metrics}

        self.model.eval()

        for batch_idx, (x, y) in enumerate(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                # make predictions
                out1, out2 = self.model.forward(x)
                classes = class_count(y, self.num_classes)

                # calculate weighted loss
                clf_loss = self.criterion(out2, classes)
                sgm_loss = self.loss(out1, y)
                loss = ((1.0 - self.loss_weight) * clf_loss) + (self.loss_weight * sgm_loss)

            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {'loss': loss_meter.mean}
            logs.update(loss_logs)

            # neptune logging (valid step)
            neptune.log_metric('test_loss_step', loss_value)

            for m in self.metrics:
                metric = m[0]  # unpack metric class
                metric_name = m[1]  # unpack metric name
                metric_type = m[2]  # unpack metric type

                if metric_type == 'classification':
                    metric_value = metric(out2, classes).cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                elif metric_type == 'segmentation':
                    d = out1.get_device() if out1.is_cuda else 'cpu'
                    if not self.binary:
                        y_pred = torch.argmax(out1, dim=1, keepdim=True).to(d)
                        y_pred = one_hot(y_pred, num_classes=out1.shape[1])
                    else:
                        y_pred = torch.round(out1).to(d)

                    # calculate metric
                    metric_value = metric(y_pred, y)
                    metric_value = metric_value[0][0] if isinstance(metric_value, tuple) else metric_value
                    metric_value = metric_value.cpu().detach().numpy()
                    metric_trackers[metric_name].add(metric_value)

                else:
                    raise ValueError(f'Type {metric_type} is not a valid metric type.')

                # neptune logging (valid step)
                neptune.log_metric('test_' + metric_name + '_step', metric_value)

            metrics_logs = {k: v.mean for k, v in metric_trackers.items()}
            logs.update(metrics_logs)

        # neptune logging (valid epoch)
        for k, v in logs.items():
            neptune.log_metric('test_' + k + '_epoch', v)

        duration = time.time() - start

        if self.verbose:
            self._show_progress(duration, logs, stage='Test')

        return logs

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)

        if self.verbose:
            print('Checkpoint saved!')

    def compile(self, optimizer, loss, metrics, num_classes):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.num_classes = num_classes
        self._to_device()

    def fit(self, train_loader, valid_loader, epochs, scheduler, binary,
            loss_weight=0.5, verbose=True, test_loader=None):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.scheduler = scheduler
        self.loss_weight = loss_weight
        self.verbose = verbose
        self.binary = binary

        self.valid_metric = self.metrics[0][1]
        self.best_score = 0.0

        for epoch in range(self.epochs):

            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.epochs}')  # starts at zero

            # training and validation epochs
            train_logs = self._train_epoch(epoch=epoch)
            valid_logs = self._valid_epoch(epoch=epoch)
            self.scheduler.step()  # update learning rate scheduler
            self.history['train'][epoch] = train_logs
            self.history['valid'][epoch] = valid_logs

            if self.best_score < valid_logs[self.valid_metric]:
                self.best_score = valid_logs[self.valid_metric]

                if self.save_checkpoints:
                    self._save_checkpoint()

        if test_loader:
            print('Starting testing...')

            # load best checkpoint
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            self.model.to(self.device)
            self.model.eval()

            if self.verbose:
                print(f'Epoch {1}/{1}')  # starts at zero

            epoch = 0

            # testing epoch
            test_logs = self._test_epoch(epoch=epoch)
            self.history['test'][epoch] = test_logs
