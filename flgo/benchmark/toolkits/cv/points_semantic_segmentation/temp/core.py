import torch
from tqdm import tqdm
import torch.nn as nn
import os
import numpy as np
from flgo.benchmark.toolkits.cv.points_semantic_segmentation import GeneralCalculator, FromDatasetPipe, FromDatasetGenerator
from .config import train_data
try:
    from .config import test_data
except:
    test_data = None
try:
    from .config import val_data
except:
    val_data = None

def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy #

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies

def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU #

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))
    return ious

def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    accuracies = []
    ious = []
    with torch.no_grad():
        for points, labels in tqdm(loader, desc='Validation', leave=False):
            points = points.to(device)
            labels = labels.to(device)
            scores = model(points)
            loss = criterion(scores, labels)
            losses.append(loss.cpu().item())
            accuracies.append(accuracy(scores, labels))
            ious.append(intersection_over_union(scores, labels))
    return {'loss':np.mean(losses),
            # 'accuracies':np.nanmean(np.array(accuracies), axis=0),
            # 'IoUs':np.nanmean(np.array(ious), axis=0),
            'mIoU': np.nanmean(np.array(ious), axis=0).mean(),
            'accuracy':np.nanmean(np.array(accuracies))
            }


class TaskGenerator(FromDatasetGenerator):
    def __init__(self):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1],
                                            train_data=train_data, val_data=val_data, test_data=test_data)

class TaskPipe(FromDatasetPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, train_data=train_data, val_data=val_data, test_data=test_data)

class TaskCalculator(GeneralCalculator):
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        data_loader = self.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=self.collect_fn)
        return evaluate(model, data_loader, criterion=nn.CrossEntropyLoss(), device=self.device)