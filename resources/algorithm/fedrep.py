"""
For the Module used by this algorithm, it should be like
```python
class Model:
   def init_global_module(self, object):
      if 'Server' in object.get_classname():
          object.set_model( Encoder())

   def init_local_module(self, object):
      if 'Client' in object.get_classname():
          object.set_model( Head(), 'head')
```
The Encoder and Head should repesctively be the representation and the local classifier.

The Encoder should be like
```python
class Encoder(flgo.utils.fmodule.FModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = None

    def set_head(self, head=None):
        self.head = head

    def forward(self, *args, **kwargs):
        return self.head(self.encoder(*args, **kwargs))

```
"""
import warnings
import torch
import torch.utils.data.dataset
import torch.nn as nn
import flgo.algorithm.fedbase
from flgo.utils.fmodule import FModule

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.aggregation_option = 'uniform'

class Client(flgo.algorithm.fedbase.BasicClient):
    def train(self, model):
        # update local head for several iterations
        self.model.to(self.device)
        model.set_head(self.model)
        head_optim = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for _ in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_grad)
            head_optim.step()
        # update global representation for one step
        model.train()
        model.zero_grad()
        encoder_optim = self.calculator.get_optimizer(model.encoder, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        batch_data = self.get_batch_data()
        loss = self.calculator.compute_loss(model, batch_data)['loss']
        loss.backward()
        if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.encoder.parameters(),max_norm=self.clip_grad)
        encoder_optim.step()

    def test(self, model, flag='val'):
        self.server.model.set_head(self.model)
        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        return self.calculator.test(self.server.model, dataset, min(self.test_batch_size, len(dataset)), self.option['num_workers'])

dataset_list = ['mnist', 'cifar10', 'cifar100', ]

def init_global_module(object):
    if 'Server' in object.get_classname():
        d = object.test_data
        while hasattr(d, 'dataset') and isinstance(d, torch.utils.data.dataset.Subset):
            d = d.dataset
        dataset = object.test_data.__class__.__name__.lower()
        for dname in dataset_list:
            if dname in dataset:
                dataset = dname
                break
        object.set_model(Encoder(dataset).to(object.device))

def init_local_module(object):
    if 'Client' in object.get_classname():
        d = object.train_data
        while hasattr(d, 'dataset') and isinstance(d, torch.utils.data.dataset.Subset):
            d = d.dataset
        dataset = d.__class__.__name__.lower()
        for dname in dataset_list:
            if dname in dataset:
                dataset = dname
                break
        object.set_model(Head(dataset).to(object.device))

def init_dataset(object):
    if 'Client' in object.get_classname():
        dataset = object.train_data.__class__.__name__.lower()
        if 'cifar10' in dataset:
            from flgo.benchmark.cifar10_classification.model.cnn_data_aug import AugmentDataset
            object.train_data = AugmentDataset(object.train_data)

class Encoder(FModule):
    def __init__(self, name='mnist'):
        super().__init__()
        self.name = name
        if name=='mnist':
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(3136, 512),
                nn.ReLU(),
            )
        elif 'cifar' in name:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(1),
                nn.Linear(1600, 384),
                nn.ReLU(),
                nn.Linear(384, 192),
                nn.ReLU(),
            )
        else:
            self.encoder = None
            warnings.WarningMessage('Model has not been implemented')

    def set_head(self, head=None):
        if head is not None:
            self.head = head

    def forward(self, *args, **kwargs):
        if self.head is not None:
            return self.head(self.encoder(*args, **kwargs))
        else:
            return self.encoder(*args, **kwargs)

class Head(FModule):
    def __init__(self, name='mnist'):
        super().__init__()
        self.name = name
        if name=='mnist':
            self.head = nn.Linear(512, 10)
        elif 'cifar' in name:
            self.head = nn.Linear(192, 10)
        else:
            self.head = None
            raise NotImplementedError('Model for {} has not been implemented.'.format(self.name))

    def forward(self, *args, **kwargs):
        return self.head(*args, **kwargs)
