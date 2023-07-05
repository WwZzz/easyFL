from ..config import get_model
from flgo.utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    if 'Client' in object.__class__.__name__:
        object.model = Model().to(object.device)

def init_global_module(object):
    pass