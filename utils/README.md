# Utils
Here we provide four necessary modules for users to better use easyFL. Firstly, we shortly introduce the major role of each module.

* `fflow` controls the whole flow of running easyFL system to train a fedtask, including reading external options, initializing the system, running time logger and outputting records. 

* `fmodule` mainly implements some commonly used operations for instances in `torch.nn.Module` (e.g. add, scale, average). All the other things correlating to training models are also placed here (e.g. device, TaskCalculator).

* `result_analysis` provides functions to open and filter the experimental results (i.e. records in `fedtask/task_name/record/`). Some common visualizing methods are also provided (e.g. drawing loss-round curve).

* `systemic_simulator` simulates the system heterogeneity in the network states (inactivity, communication latency, dropout) and computing power (e.g. imcomplement updates) across clients, which is independent to `benchmark` and `algorithm`.

# fflow

# fmodule

# result_analysis

# systemic_simulator
