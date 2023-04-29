import torchvision
import os

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))]
)
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'MNIST')
train_data = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=transform)
