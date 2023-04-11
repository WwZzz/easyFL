import collections

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import flgo.utils.fmodule as fmodule
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.metrics import auc
import torch
from torch import nn, Tensor

class Model(fmodule.FModule):
    CLASSES = (
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    def __init__(self):
        super(Model, self).__init__()
        self.name = 'faster-rcnn'
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True, num_classes=len(self.CLASSES))

    def convert_target(self, target):
        anno = target['annotation']
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            iscrowd.append(int(obj['difficult']))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno['filename'][5:-4]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd
        return target

    def transform(self, images, targets, device=None):
        if device is None:
            images = list(images)
            targets = [self.convert_target(t) for t in targets]
            targets = [{k: v for k, v in t.items() if not isinstance(v, dict)} for t in targets]
        else:
            images = list(img.to(device) for img in images)
            targets = [self.convert_target(t) for t in targets]
            targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, dict)} for t in targets]
        return images, targets

    def forward(self, images, targets):
        output = self.model(images, targets)
        return output

    def organize_output(self, output):
        if self.model.training:
            loss_total = sum(loss for loss in output.values())
            return {'loss':loss_total}
        else:
            return output

    def evaluate(self, dataloader=None):
        if dataloader is None or len(dataloader)==0: return {}
        losses = []
        with torch.no_grad():
            self.model.train()
            for batch_id, (images, targets) in enumerate(dataloader):
                images, targets = self.transform(images, targets, device=self.get_device())
                output = self.model(images, targets)
                output = self.organize_output(output)
                losses.append(output['loss']*len(images))

        self.model.eval()
        metric_dict_by_class = [collections.defaultdict(list) for _ in self.CLASSES]
        for batch_id, (images, targets) in enumerate(dataloader):
            images, _ = self.transform(images, targets, device=self.get_device())
            output = self.model(images)
            predictions = self.organize_output(output)
            targets = [
                {'boxes': [torch.LongTensor([int(ob['bndbox']['xmin']), int(ob['bndbox']['ymin']), int(ob['bndbox']['xmax']), int(ob['bndbox']['ymax'])])for ob in t['annotation']['object']],
                'labels': [self.CLASSES.index(ob['name']) for ob in t['annotation']['object']]
                 } for t in targets]
            for label, class_name in enumerate(self.CLASSES):
                precision, recall, ap, num_class_obj  = precision_recall_ap(targets, predictions, iou_thresh=0.5, class_label=label)
                if num_class_obj>0:
                    mean_ap = mean_average_precision(targets, predictions, class_label=label)
                    acc = accuracy(targets, predictions, class_label=label)
                    red = redundancy(targets, predictions, class_label=label)
                    metric_dict_by_class[label]['num_objs'].append(num_class_obj)
                    metric_dict_by_class[label]['precision'].append(precision)
                    metric_dict_by_class[label]['recall'].append(recall)
                    metric_dict_by_class[label]['ap'].append(ap)
                    metric_dict_by_class[label]['mAP'].append(mean_ap)
                    metric_dict_by_class[label]['acc'].append(acc)
                    metric_dict_by_class[label]['red'].append(red)
        mkeys = ['precision', 'recall', 'mAP', 'AP', 'acc', 'red']
        res = {mk:[] for mk in mkeys}
        res['ap_class'] = []
        res['loss'] = sum(losses)/len(dataloader)
        for label, class_name in enumerate(self.CLASSES):
            if len(metric_dict_by_class[label]['num_objs'])==0:
                continue
            num_class_objs = sum(metric_dict_by_class[label]['num_objs'])
            for mk in mkeys:
                res[mk].append(sum(ni*pi for ni,pi in zip(metric_dict_by_class[label]['num_objs'], metric_dict_by_class[label][mk]))/num_class_objs)
            res['ap_class'].append(label)
        return res

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)

def iou(real_box, pred_box):
    """Calculates Intersection over Union for real and predicted bounding box."""

    if len(real_box) == 0:
        raise "Length of target box is 0."
    if len(pred_box) == 0:
        raise "Length of predicted box is 0."
    x1 = max(real_box[0], pred_box[0])
    y1 = max(real_box[1], pred_box[1])
    x2 = min(real_box[2], pred_box[2])
    y2 = min(real_box[3], pred_box[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    real_box_area = (real_box[2] - real_box[0] + 1) * (real_box[3] - real_box[1] + 1)
    pred_box_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    return intersection_area / (real_box_area + pred_box_area - intersection_area)


def precision_recall_ap(targets, predictions, iou_thresh=0.5, class_label=None):
    """Calculates Precision, Recall and Average Precision for given threshold."""

    tp_list = []
    num_gt_boxes = 0
    for target in targets:
        for idx, box in enumerate(target['boxes']):
            if target['labels'][idx] == class_label or class_label is None:
                num_gt_boxes += 1
    for target, pred in zip(targets, predictions):
        if len(pred['boxes']) == 0:
            continue
        if len(target['boxes']) == 0:
            for i in range(len(pred['boxes'])):
                tp_list.append(0)
        for target_idx, target_box in enumerate(target['boxes']):
            if target['labels'][target_idx] != class_label and class_label is not None:
                continue
            is_append = False
            for pred_box in pred['boxes']:
                if iou(target_box, pred_box) > iou_thresh:
                    tp_list.append(1)
                    is_append = True
                    break
            tp_list.append(0) if not is_append else None
    if len(tp_list) == 0:
        if num_gt_boxes == 0:
            return 1, 1, 1, num_gt_boxes
        return 0, 0, 0, num_gt_boxes
    if num_gt_boxes == 0:
        return 0, 0, 0, num_gt_boxes
    tp_list.sort(reverse=True)
    precision_values = [sum(tp_list[:i+1])/len(tp_list[:i+1]) for i in range(len(tp_list))]
    recall_values = [sum(tp_list[:i+1])/num_gt_boxes for i in range(len(tp_list))]
    precision, recall = precision_values[-1], recall_values[-1]
    average_precision = precision if len(precision_values) == 1 else auc(recall_values, precision_values)
    return precision, recall, average_precision, num_gt_boxes

def f1_score(precision, recall):
    """Calculates F1-score."""

    return 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

def mean_average_precision(targets, predictions, class_label=None, thresh_values=np.arange(0.5, 1, 0.05)):
    """Calculates mean Average Precision."""

    ap_list = []
    for thresh in thresh_values:
        ap_list.append(precision_recall_ap(targets, predictions, thresh, class_label)[2])
    return np.mean(ap_list)


def accuracy(targets, predictions, iou_thresh=0.5, class_label=None):
    """Calculates batch accuracy."""

    num_correct_predictions = 0
    num_predictions = 0
    for pred in predictions:
        for idx, box in enumerate(pred['boxes']):
            if pred['labels'][idx] == class_label or class_label is None:
                num_predictions += 1
    for target, pred in zip(targets, predictions):
        if len(target['boxes']) == 0 or len(pred['boxes']) == 0:
            continue
        for target_idx, target_box in enumerate(target['boxes']):
            if target['labels'][target_idx] != class_label and class_label is not None:
                continue
            for pred_idx, pred_box in enumerate(pred['boxes']):
                if iou(target_box, pred_box) > iou_thresh \
                        and target['labels'][target_idx] == pred['labels'][pred_idx]:
                    num_correct_predictions += 1
                    break
    return 1 if num_predictions == 0 else num_correct_predictions / num_predictions


def redundancy(targets, predictions, class_label=None):
    """Calculates redundancy of predictions."""

    num_predictions, num_gt_boxes = 0, 0
    for pred in predictions:
        for idx, box in enumerate(pred['boxes']):
            if pred['labels'][idx] == class_label or class_label is None:
                num_predictions += 1
    for target in targets:
        for idx, box in enumerate(target['boxes']):
            if target['labels'][idx] == class_label or class_label is None:
                num_gt_boxes += 1
    return num_predictions if num_gt_boxes == 0 else num_predictions / num_gt_boxes