from ..config import get_model
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        if hasattr(self.model, 'compute_loss'):
            self.compute_loss = self.model.compute_loss
        self.encode = self.model.encode
        self.decode = self.model.decode
        self.decode_all = self.model.decode_all

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)