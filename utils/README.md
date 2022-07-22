# Utils
Here we provide five necessary modules for users to better use easyFL. Firstly, we shortly introduce the major role of each module.

* `fflow` controls the whole flow of running easyFL system to train a fedtask, including reading external options, initializing the system and outputting records. 

* `fmodule` mainly implements some commonly used operations for instances in `torch.nn.Module` (e.g. add, scale, average). All the other things correlating to training models are also placed here (e.g. device, TaskCalculator).

* `result_analysis` provides functions to open and filter the experimental results (i.e. records in `fedtask/task_name/record/`). Some common visualizing methods are also provided (e.g. drawing loss-round curve, bar, scatter).

* `systemic_simulator` simulates the system heterogeneity in the network states (inactivity, communication latency, dropout) and computing power (e.g. imcomplement updates) across clients, which is independent to `benchmark` and `algorithm`.

* `logger` is a package that contains several general loggers' implementions which decide things to be recorded during running-time. The new logger should be added as a new file `xxxlogger.py` here.

# fmodule
This module is designed for performing operations on `torch.nn.Module` in a user-friendly and easy manner, since the models will be frequently aggregated, plused, scaled during federated optimization. To achieve this, we create a new class `FModule` inheritting from `torch.nn.Module` and realize several common operations on this class. 

`fmodule` means federated module, which provides a new class `FModule` that inheritting from `torch.nn.Module`.
# result_analysis

# systemic_simulator

# logger

# fflow
