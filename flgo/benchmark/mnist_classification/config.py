import torchvision
import os
import flgo.benchmark

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
path = os.path.join(flgo.benchmark.data_root, 'MNIST')
train_data = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
