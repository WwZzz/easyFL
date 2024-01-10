import os

import numpy as np
import torch.utils.data
from flgo.benchmark.base import BasicTaskCalculator
import flgo.benchmark.toolkits.cv.segmentation
from flgo.benchmark.toolkits.cv.classification import BuiltinClassPipe as ClsPipe
from flgo.benchmark.toolkits.cv.classification import BuiltinClassGenerator
from flgo.benchmark.toolkits.cv.detection.utils import mean_average_precision, precision_recall_ap, accuracy, redundancy, iou, average_precision
from sklearn.metrics import auc
import flgo.benchmark.base
import collections
try:
    import ujson as json
except:
    import json
from tqdm import tqdm

FromDatasetGenerator = flgo.benchmark.base.FromDatasetGenerator

FromDatasetPipe = flgo.benchmark.toolkits.cv.segmentation.FromDatasetPipe

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
        self.collect_fn = lambda x:tuple(zip(*x))

    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        model.train()
        tdata = self.to_device(data)
        output = model(*tdata)
        if hasattr(model, 'compute_loss'):
            loss = model.compute_loss(output, tdata)
        else:
            loss = sum(list(output.values()))
        return {'loss':loss}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        dataloader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        num_classes = dataset.num_classes
        model.train()
        num_samples = 0
        losses = {}
        for batch_data in dataloader:
            batch_data = self.to_device(batch_data)
            output = model(*batch_data)
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(output, batch_data).item()
            else:
                for k in output:
                    losses[k] = losses.get(k, 0.0) + output[k].item()*len(batch_data[0])
                loss = sum([v.item() for v in output.values()])
            losses['all_loss'] = losses.get('all_loss', 0.0) + loss*len(batch_data[0])
            num_samples += len(batch_data[0])
        for k,v in losses.items():
            losses[k]/=num_samples
        # compute AP
        predictions = []
        targets = []
        model.eval()
        for images, labels in tqdm(dataloader, desc='Predicting'):
            images = list(img.to(self.device) for img in images)
            labels = [{k: v.numpy() for k, v in t.items()} for t in labels]
            outputs = model(images)
            for out in outputs:
                for k in out.keys():
                    out[k] = out[k].cpu().numpy()
            predictions.extend(outputs)
            targets.extend(labels)
        # count TP for each class
        dects = {i:[] for i in range(1, num_classes)}
        gts = {i: {} for i in range(1, num_classes)}
        for image_id, pred in enumerate(predictions):
            for det_id in range(len(pred['boxes'])):
                class_id = int(pred['labels'][det_id])
                dects[class_id].append([image_id, class_id, pred['scores'][det_id], pred['boxes'][det_id]])
        for image_id, target in enumerate(targets):
            for gt_id in range(len(target['boxes'])):
                class_id = int(target['labels'][gt_id])
                gts[class_id][image_id] = gts[class_id].get(image_id, []) + [[image_id, class_id, [], target['boxes'][gt_id]]]
        res = []
        ious = np.arange(0.5, 1.0, 0.05)
        tf_dicts = {class_id:{iou_th:{'tp':None, 'fp':None} for iou_th in ious} for class_id in range(1, num_classes)}
        for class_id in range(1, num_classes):
            c_dects = sorted(dects[class_id], key=lambda d:d[2], reverse=True)
            c_gts = gts[class_id]
            c_tf_dict = tf_dicts[class_id]
            for iou_th in c_tf_dict:
                c_tf_dict[iou_th]['tp'] = np.zeros(len(c_dects))
                c_tf_dict[iou_th]['fp'] = np.zeros(len(c_dects))
            # c_tp = np.zeros(len(c_dects))
            # c_fp = np.zeros(len(c_dects))
            c_npos = sum(list(len(v) for v in c_gts.values()))
            for det_id in range(len(c_dects)):
                image_id = c_dects[det_id][0]
                gt = c_gts[image_id] if image_id in c_gts else []
                max_iou = -0.1
                for j in range(len(gt)):
                    d_iou = iou(gt[j][3], c_dects[det_id][3])
                    if d_iou> max_iou:
                        max_iou = d_iou
                        jmax = j
                for iou_th in ious:
                    if max_iou>iou_th:
                        if iou_th not in c_gts[c_dects[det_id][0]][jmax][2]:
                            c_gts[c_dects[det_id][0]][jmax][2].append(iou_th)
                            c_tf_dict[iou_th]['tp'][det_id] = 1
                            # c_tp[det_id] = 1
                        else:
                            c_tf_dict[iou_th]['fp'][det_id] = 1
                            # c_fp[det_id] = 1
                    else:
                        c_tf_dict[iou_th]['fp'][det_id] = 1
            res_ious = {}
            for iou_th in ious:
                c_acc_fp_i = np.cumsum(c_tf_dict[iou_th]['fp'])
                c_acc_tp_i = np.cumsum(c_tf_dict[iou_th]['tp'])
                c_recall_i = c_acc_tp_i/(c_npos+1e-8)
                c_precision_i = np.divide(c_acc_tp_i, (c_acc_tp_i + c_acc_fp_i))
                c_ap_i, c_mpre_i, c_mrec_i, c_ii_i = average_precision(c_recall_i, c_precision_i)
                res_ious[iou_th] = c_ap_i
            res.append(res_ious)
        # mAP@0.5
        tmp = [np.array([c_res[iou_th] for c_res in res]).mean() for iou_th in ious]
        mAP_05 = tmp[0]
        mAP_075 = tmp[5]
        mAP_05_095 = np.array(tmp).mean()
        ret = {}
        ret.update(losses)
        ret.update({'mAP@.5':float(mAP_05), 'mAP@.75':float(mAP_075), 'mAP@.5:.95':mAP_05_095})
        return ret

    def to_device(self, data):
        images, targets = data
        images = list(img.to(self.device) for img in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collect_fn)

