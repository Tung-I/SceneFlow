import torch
import random
import numpy as np
import logging
from tqdm import tqdm
from src.runner.trainers.base_trainer import BaseTrainer


class PWCDisparityTrainer(BaseTrainer):
    """The KiTS trainer for segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').

        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:

            batch = self._allocate_data(batch)
            rgb_l, rgb_r, target = self._get_inputs_targets(batch)

            # print(rgb_l.size())
            # print(target.size())

            # rgb_l = rgb_l.permute(0, 3, 1, 2).contiguous()
            # rgb_r = rgb_r.permute(0, 3, 1, 2).contiguous()
            # rgb_next_l = rgb_next_l.permute(0, 3, 1, 2).contiguous()
            # rgb_next_r = rgb_next_r.permute(0, 3, 1, 2).contiguous()
            # disparity = disparity.contiguous()
            # disparity_next = disparity_next.contiguous()
            # target = disparity.permute(0, 3, 1, 2).contiguous()


            if mode == 'training':
                outputs = self.net(rgb_l, rgb_r)
                # print(outputs.size())

                losses = self._compute_losses(outputs, target)

                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(rgb_l, rgb_r)
                    losses = self._compute_losses(outputs, target)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, target)

            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            # print(f'######################{count}')
            self._update_log(log, batch_size, loss, losses, metrics)
            count += batch_size
            trange.set_postfix(**dict((key, f'{value / count: .3f}') for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, outputs

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        return batch['rgb_l'], batch['rgb_r'], batch['disparity']

    def _compute_losses(self, output, target, mask=None):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target, mask) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target, mask=None):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor): The data target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target, mask) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] = 0
                for i in range(self.net.out_channels):
                    log[f'Dice_{i}'] = 0
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (list of torch.Tensor): The computed losses.
            metrics (list of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] += _metric.mean().item() * batch_size
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item() * batch_size
            else:
                log[metric.__class__.__name__] += _metric.item() * batch_size