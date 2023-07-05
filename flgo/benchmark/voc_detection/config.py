import torch
import torchvision.models
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flgo.benchmark.toolkits.cv.detection import *
import torchvision.datasets
import os

CLASSES = (
        '__background__',
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
CLASSES_MAP = {name:idx for idx, name in enumerate(CLASSES)}
def voc_target_transform(y):
    objects = y['annotation']['object']
    boxes = [torch.FloatTensor([int(v) for v in obj['bndbox'].values()]) for obj in objects]
    labels = [torch.LongTensor(torch.LongTensor([CLASSES_MAP[obj['name'].lower()]])) for obj in objects]
    return {'boxes': torch.stack(boxes), 'labels':torch.cat(labels)}

transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'PASCAL_VOC')
train_data = torchvision.datasets.VOCDetection(root=path, download=True, image_set='trainval', year='2007', transform=transform, target_transform=voc_target_transform)
test_data = torchvision.datasets.VOCDetection(root=path, download=True, image_set='test', year='2007', transform=transform, target_transform=voc_target_transform)
train_data.num_classes = len(CLASSES)
test_data.num_classes = len(CLASSES)

def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
    return model

