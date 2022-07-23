# Utils
Here we provide five necessary modules for users to better use easyFL. Firstly, we shortly introduce the major role of each module.

* `fflow` controls the whole flow of running easyFL system to train a fedtask, including reading external options, initializing the system and outputting records. 

* `fmodule` mainly implements some commonly used operations for instances in `torch.nn.Module` (e.g. add, scale, average). All the other things correlating to training models are also placed here (e.g. device, TaskCalculator).

* `result_analysis` provides functions to open and filter the experimental results (i.e. records in `fedtask/task_name/record/`). Some common visualizing methods are also provided (e.g. drawing loss-round curve, bar, scatter).

* `systemic_simulator` simulates the system heterogeneity in the network states (inactivity, communication latency, dropout) and computing power (e.g. imcomplement updates) across clients, which is independent to `benchmark` and `algorithm`.

* `logger` is a package that contains several general loggers' implementions which decide things to be recorded during running-time. The new logger should be added as a new file `xxxlogger.py` here.

## fmodule
This module is designed for performing operations on `torch.nn.Module` in a user-friendly and easy manner, since the models will be frequently aggregated, plused, scaled during federated optimization. To achieve this, we create a new class `FModule` inheritting from `torch.nn.Module` and realize several common operations on this class. Here we provide a few examples to show how to use this module. The following code should be run in the python console under the working dictionary of this project.

Example:

```python
>>> from utils.fmodule import FModule
>>> import torch.nn as nn

# create the class of model inheritting from utils.fmodule.FModule
>>> class Model(FModule):
>>>   def __init__(self):
>>>       super().__init__()
>>>       self.w = nn.Linear(2,2)
# create instances of Model
>>> m1, m2 = Model(), Model()
>>> m1.w.weight, m1.w.bias, m2.w.weight, m2.w.bias
```
Then the parameters of model `m1` and `m2` will be printed on the screen like this

```
(Parameter containing:
tensor([[-0.4528,  0.0502],
        [ 0.3343,  0.6270]], requires_grad=True), Parameter containing:
tensor([ 0.2749, -0.1444], requires_grad=True), Parameter containing:
tensor([[ 0.1559,  0.5991],
        [ 0.0758, -0.0572]], requires_grad=True), Parameter containing:
tensor([-0.0408,  0.6844], requires_grad=True))
```
Now we try to add the two model, 

```python
>>> m3 = m1 + m2
>>> m3.w.weight, m3.w.bias
>>> m3.w.weight==m1.w.weight+m2.w.weight, m3.w.bias==m1.w.bias+m2.w.bias
```

and we will obtain:
```
(Parameter containing:
tensor([[-0.2969,  0.6493],
        [ 0.4101,  0.5698]], requires_grad=True), Parameter containing:
tensor([0.2341, 0.5400], requires_grad=True))

(tensor([[True, True],
        [True, True]]), tensor([True, True]))
```
The result is correct as we add the parameters of the two model manually.

Apart from `add`, we also provide operations including: `dot`, `sub`, `normalize`, `scale`, `cos_sim`, `average`...
Please read `utils.fmodule` for more details.

## result_analysis
The result-analysis is designed for analyzing the experimental records in a customized manner. To achieve this goal, we developed two useful processes: 1) records_filter, 2) records_analyser. And both the two processes are totally controlled by a simple `.yml` file. Now we firstly show how to customize the `yml` to decide the behavior of result_analysis.py.

Example:

```yaml
#filter
task:
  mnist_classification_cnum100_dist0_skew0_seed0
header:
  - fedavg
  - fedprox
flt:
  B: <512
  LR: '0.1'
legend_flt:
  - B

#analyser
ploter:
  plot:
    - x: communication_round
      y: test_loss
  bar:
    - x: client_id
      y: valid_loss_dist
info:
  final_value:
    - valid_loss
```

### records_

## systemic_simulator

## logger

## fflow
