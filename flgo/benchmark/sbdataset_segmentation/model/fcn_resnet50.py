import torchvision.models
import flgo.utils.fmodule as fmodule

class Model(fmodule.FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=21, aux_loss=True)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)