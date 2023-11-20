import torchvision
import os

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2584, 0.2712))]
)

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAW_DATA', 'STL10')
train_data = torchvision.datasets.STL10(root=path, train=True, download=True, transform=transform)
test_data = torchvision.datasets.STL10(root=path, train=False, download=True, transform=transform)