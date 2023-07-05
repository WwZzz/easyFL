from flgo.utils.fmodule import FModule
import torch

class Model(FModule):
    def __init__(self, dim_in = 2, dim_out = 1):
        super(Model, self).__init__()
        self.x = torch.nn.Parameter(torch.zeros(dim_in).unsqueeze(0))

    def forward(self, e):
        output = e-self.x.repeat((len(e), 1))
        output = torch.diag(torch.mm(output, output.T))
        return output

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)