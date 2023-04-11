import torch.utils.data
from flgo.benchmark.base import BasicTaskCalculator
from flgo.benchmark.toolkits.cv.classification import BuiltinClassPipe as ClsPipe
from flgo.benchmark.toolkits.cv.classification import BuiltinClassGenerator
import numpy as np

class BuiltinClassPipe(ClsPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, perturbation=None, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx: p for idx, p in zip(indices, perturbation)} if perturbation is not None else None
            self.pin_memory = pin_memory
            if not self.pin_memory:
                self.X = None
                self.Y = None
            else:
                self.X = [self.dataset[i][0] for i in self.indices]
                self.Y = [self.dataset[i][1] for i in self.indices]

        def __getitem__(self, idx):
            if self.X is not None:
                if self.perturbation is None:
                    return self.X[idx], self.Y[idx]
                else:
                    return self.X[idx] + self.perturbation[self.indices[idx]], self.Y[idx]
            else:
                if self.perturbation is None:
                    if isinstance(idx, list):
                        return self.dataset[[self.indices[i] for i in idx]]
                    return self.dataset[self.indices[idx]]
                else:
                    return self.dataset[self.indices[idx]][0] + self.perturbation[self.indices[idx]], \
                           self.dataset[self.indices[idx]][1]

    def __init__(self, task_path, buildin_class, transform=None):
        super(BuiltinClassPipe, self).__init__(task_path, buildin_class, transform)

class GeneralCalculator(BasicTaskCalculator):
    r"""
    Calculator for the dataset in torchvision.datasets.

    Args:
        device (torch.device): device
        optimizer_name (str): the name of the optimizer
    """
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        images, targets = data
        images, targets = model.transform(images, targets, device=self.device)
        output = model(images, targets)
        res = model.organize_output(output)
        # print(res['loss'])
        return res

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False, metrics=['loss']):
        dataloader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        metrics = model.evaluate(dataloader)
        return metrics
        #
        # if hasattr(model, 'name') and 'rcnn' in model.name:
        #     if metrics=='all': metrics=['loss', 'mAP']
        #     if type(metrics) is not list: metrics=[metrics]
        #     res = {}
        #     for metric_name in metrics:
        #         if metric_name=='loss':
        #             losses = []
        #             model.train()
        #             dataloader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        #             for images, targets in dataloader:
        #                 images, targets = model.transform(images, targets, device=self.device)
        #                 output = model(images, targets)
        #                 output = model.organize_output(output)
        #                 losses.append(output['loss']*len(dataset))
        #             res['loss'] = sum(losses)/len(dataset)
        #         elif metric_name=='mAP':
        #             APs = []
        #             model.eval()
        #             dataloader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers,
        #                                              pin_memory=pin_memory)
        #             for images, targets in dataloader:
        #                 images, targets = model.transform(images, targets, device=self.device)
        #                 preds = model(images, targets)
        #                 preds = model.organize(preds)
        #                 APs.append(self.compute_ap(targets, preds))
        #             res['mAP'] = np.mean(APs)
        #     return res


    def to_device(self, images, targets):
        images = list(img.to(self.device) for img in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=lambda x:tuple(zip(*x)))

