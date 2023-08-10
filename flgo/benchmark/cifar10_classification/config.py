import torchvision
import os
import torch.nn as nn
import flgo.benchmark

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))]
)

path = os.path.join(flgo.benchmark.data_root,  'CIFAR10')
train_data = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(1600, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        return self.fc(x)

    def get_embedding(self, x):
        return self.embedder(x)

def get_model():
    return Model()
