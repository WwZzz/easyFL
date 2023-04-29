import torchvision
import os

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.2859,), (0.3530,))])
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'FASHION')
train_data = torchvision.datasets.FashionMNIST(root=root, download=True, train=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root=root, download=True, train=False, transform=transform)