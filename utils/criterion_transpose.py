import torch
import torch.nn as nn
import numpy as np

class Criterion(nn.Module):
    """
    Various type of criterions.
    """
    def __init__(self, type, **kwargs):
        super(Criterion, self).__init__()
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.type = str.lower(type)
        if 'huber' in self.type:
            self.huber_k = kwargs.get('huber_k', 0.1)
        self.l2_loss = self.mse_loss
        self.masked_l2_loss = self.masked_mse_loss
        self.custom_masked_l2_loss = self.custom_masked_mse_loss
        self.main_loss = getattr(self, type)
        self._mse = self._l2

    def _l1(self, pred, gt):
        gt = gt.squeeze(-1)
        return torch.abs(pred - gt)

    def _l2(self, pred, gt):
        return (pred - gt) ** 2

    def _huber(self, pred, gt):
        delta = torch.abs(pred - gt)
        return torch.where(delta <= self.huber_k, delta ** 2 / 2, self.huber_k * delta - self.huber_k ** 2 / 2)

    def mse_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._l2(pred, gt)[mask].mean()

    def masked_mse_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._l2(pred, gt)[mask].mean()

    def custom_masked_mse_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return self._l2(pred, gt)[mask].mean()

    def l1_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._l1(pred, gt)[mask].mean()

    def masked_l1_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._l1(pred, gt)[mask].mean()

    def custom_masked_l1_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        zero_mask = data_dict['zero_mask']
        loss = self._l1(pred, gt)
        return loss[mask].mean() + 0.01 * loss[zero_mask].mean()

    def huber_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['zero_mask']
        return self._huber(pred, gt)[mask].mean()

    def masked_huber_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['initial_loss_mask']
        return self._huber(pred, gt)[mask].mean()

    def custom_masked_huber_loss(self, data_dict):
        pred = data_dict['pred']
        gt = data_dict['depth_gt']
        mask = data_dict['loss_mask']
        return self._huber(pred, gt)[mask].mean()

    def forward(self, data_dict):
        """
        Calculate criterion given data dict.
        """
        loss_dict = {
            self.type: self.main_loss(data_dict),
            'loss': self.main_loss(data_dict)
        }
        return loss_dict
