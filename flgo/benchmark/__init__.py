r"""
This module is designed for fast creating federated tasks. For example,
in FL, a commonly used benchmark is federated MNIST that splits MNIST
into 100 shards and each shard contains data of two types of labels.

In FLGo, three basic components are created to describe a general
procedure that can easily convert various ML tasks into federated ones.

Components:
    * `TaskGenerator`
        - load the original dataset
        - partition the original dataset into local_movielens_recommendation data

    * `TaskPipe`
        - store the partition information of TaskGenerator into the disk
            when generating federated tasks
        - load the original dataset and the partition information to
            create the federated scenario when optimizing models

    * `TaskCalculator`
        - support task-specific computation when optimizing models, such
            as putting data into device, computing loss, evaluating models,
            and creating the data loader

The architecture of a complete federate benchmark is shown as follows:

```
benchmark_name                  # benchmark folder
├─ core.py                      # core file
│   ├─ TaskGenerator            # class TaskGenerator(...)
│   ├─ TaskPipe                 # class TaskPipe(...)
│   └─ TaskCalculator           # class TaskCalculator(...)
│
├─  model                       # model folder (i.e. contains various types of models)
│   ├─ model1_name.py           # model 1 (e.g. CNN)
│   ├─ ...
│   └─ modelN_name.py           # model N (e.g. ResNet)
│       ├─ init_local_module    # the function initializes personal models for parties
│       └─ init_global_module   # the function initializes the global models for parties
│
└─ __init__.py                  # containing the variable default_model
```

**Example**:
The architecture of MNIST is
```
├─ core.py
│   ├─ TaskGenerator
│   ├─ TaskPipe
│   └─ TaskCalculator
├─  model
│   ├─ cnn.py
│   └─ mlp.py
│       ├─ init_local_module
│       └─ init_global_module
└─ __init__.py
```

The details of implementing a customized benchmark are in Tutorial.3
"""
import os

path = os.path.abspath(os.path.dirname(__file__))
data_root = os.path.join(path, 'RAW_DATA')