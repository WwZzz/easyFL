import os
import torchvision
import torch
import torchvision.transforms as T
import flgo.benchmark
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
target_transform = T.Compose([
    T.PILToTensor(),
    T.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor))
])
path = os.path.join(flgo.benchmark.path, 'RAW_DATA', 'OXFORD-IIIT-PET')
# 定义训练集实例，并命名为train_data
train_data = torchvision.datasets.OxfordIIITPet(root=path, split='trainval', download=True, target_types='segmentation', transform=transform, target_transform=target_transform)
# 定义测试集实例，并命名为test_data
test_data = torchvision.datasets.OxfordIIITPet(root=path, split='test', download=True, target_types='segmentation', transform=transform, target_transform=target_transform)
train_data.num_classes = 3
test_data.num_classes = 3

def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(num_classes=3)
    return model