from flgo.utils import fmodule
import torchvision.models
import torch.nn as nn

class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        resnet34 = torchvision.models.resnet34()
        resnet34.fc = nn.Linear(512, 10)
        resnet34.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet34.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet34.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet34.layer1[0].bn3 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet34.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet34.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet34.layer1[1].bn3 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet34.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[0].bn3 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[0].bn4 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[1].bn3 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet34.layer2[1].bn4 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet34.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].bn3 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].bn4 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].bn5 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].bn6 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn3 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn4 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn5 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet34.layer3[1].bn6 = nn.GroupNorm(num_groups=2, num_channels=256)


        resnet34.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[0].bn3 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet34.layer4[1].bn3 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet34

    def forward(self, x):
        return self.model(x)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)