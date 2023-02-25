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

def init_local_module(object):
    partial_sample = object.train_data[0]
    feature = partial_sample[0]
    object.local_module = None if feature is None else PartialMLPLayer(len(feature.view(-1)), EMBEDDING_SIZE).to(object.device)

def init_global_module(object):
    partial_sample = object.train_data[0]
    feature, label, id = partial_sample
    object.global_module = None if label is None else Classifier(EMBEDDING_SIZE, NUM_CLASS).to(object.device)