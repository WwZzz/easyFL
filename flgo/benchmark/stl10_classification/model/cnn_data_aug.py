import torchvision.transforms
from torch import nn
from flgo.utils.fmodule import FModule
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
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
        self.head = nn.Linear(512, 10)

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

class AugmentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = torchvision.transforms.Compose([RandomCrop(size=(96, 96), padding=12), RandomHorizontalFlip(0.5)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return self.transform(img), label

def init_dataset(object):
    if 'Client' in object.get_classname():
        object.train_data = AugmentDataset(object.train_data)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)