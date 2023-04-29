import torchvision
import os

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'CIFAR100')
train_data = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform)