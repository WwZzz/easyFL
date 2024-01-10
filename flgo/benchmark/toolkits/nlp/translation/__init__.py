from typing import *
from collections.abc import Callable
from torch.utils.data import random_split, Subset
from torchtext.data.functional import to_map_style_dataset
from flgo.benchmark.base import BasicTaskCalculator, BasicTaskGenerator, BasicTaskPipe
import os
try:
    import ujson as json
except:
    import json
import torch

class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.DataLoader = torch.utils.data.DataLoader

    def criterion(self, outputs, targets, ignore_index=-100):
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_func(outputs[1:].view(-1, outputs.shape[-1]), targets[1:].view(-1))

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        sources, targets = self.to_device(data)
        outputs = model(sources, targets)
        loss = self.criterion(outputs, targets, model.ignore_index if hasattr(model, 'ignore_index') else -100)
        return {'loss': loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]

        Args:
            model:
            dataset:
            batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0], batch_data[1])
            batch_mean_loss = self.criterion(outputs, batch_data[1], model.ignore_index if hasattr(model, 'ignore_index') else -100).item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)
