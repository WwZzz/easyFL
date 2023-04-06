r"""
This module is designed for fast creating federated tasks. For example,
in FL, a commonly used benchmark is federated MNIST that splits MNIST
into 100 shards and each shard contains data of two types of labels.

In FLGo, three basic components are created to describe a general
procedure that can easily convert various ML tasks into federated ones:

    1. TaskGenerator
        1.1 load the original dataset
        1.2 partition the original dataset into local data

    2. TaskPipe
        2.1 store the partition information of TaskGenerator into the disk
            when generating federated tasks
        2.2 load the original dataset and the partition information to
            create the federated scenario when optimizing models

    3. TaskCalculator
        3.1 support task-specific computation when optimizing models, such
            as putting data into device, computing loss, evaluating models,
            and creating the data loader

The architecture of a complete federate benchmark is shown as follows:

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
"""

path = '/'.join(__file__.split('/')[:-1])