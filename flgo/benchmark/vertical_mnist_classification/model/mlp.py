import torch
import torch.nn

EMBEDDING_SIZE = 64
NUM_CLASS = 10

class PartialMLPLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        return self.mlp(x)

class Classifier(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, fusion):
        x = self.relu(fusion)
        x = self.mlp(x)
        return x

def init_local_module(partial_sample):
    feature = partial_sample[0]
    if feature is None: return None
    else:
        input_size = len(feature.view(-1))
        return PartialMLPLayer(input_size, EMBEDDING_SIZE)

def init_global_module(partial_sample):
    feature, label, id = partial_sample
    if label is None:
        return None
    else:
        return Classifier(EMBEDDING_SIZE, NUM_CLASS)

