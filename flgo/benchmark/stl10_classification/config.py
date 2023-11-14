import torchvision
import os
import torch.nn as nn
import flgo.benchmark

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((96, 96)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2584, 0.2712))
])

path = os.path.join(flgo.benchmark.data_root,  'STL10')
train_data = torchvision.datasets.STL10(root=path, split='train', download=True, transform=transform)
test_data = torchvision.datasets.STL10(root=path, split='test', download=True, transform=transform)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedder = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(12800, 512),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        return self.fc(x)

    def get_embedding(self, x):
        return self.embedder(x)

def get_model():
    return Model()
