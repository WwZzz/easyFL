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

Example: `utils/example.yml`

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
      y: valid_loss
  bar:
    - x: client_id
      y: valid_loss_dist
info:
  final_value:
    - valid_loss
```

For this `.yml` file, the filter will firstly scan all the experimental records in `fedtask/mnist_classification_cnum100_dist0_skew0_seed0/record` according the value of `task`. Then, the `header` will only preserve the records whose name starts with `fedavg` or `fedprox`. Thirdly, the `flt` will filter the records according to the value of the specified hyperparameters. In this case, only records whose `batch_size < 512` and `learning_rate==0.1` will be preserved. For the survived records, the `legend_flt` will decide their legend when plotting (e.g. `fedavg B 64` will be shown as legend for records that running with fedavg and batchsize is 64). 

After the records are selected, the `analyser` will make analysis on them. We use `ploter` to visualize the results based on `matplotlib`, and we use `info` to output the information as a table to the screen. In this case, `plot` will call the method with the same name of `Drawer` that is defined in `result_analysis.py`, and each element of the list of `plot` will be regarded as a `plot_object`, which is the input of `Drawer.plot`. Here we define `plot` to draw data with curves. The `info` is similar. All the key of the dicts of `ploter` and `info` will be handled respectively by `Drawer` and `Former` with the same-name methods.

Now we run the commands below:
```sh
# fedavg
python main.py --task mnist_classification_cnum100_dist0_skew0_seed0 --model cnn --algorithm fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.1 --proportion 0.1 --batch_size 10 --eval_interval 1 --gpu 0 --logger simple_logger
# fedprox mu=0.1
python main.py --task mnist_classification_cnum100_dist0_skew0_seed0 --model cnn --algorithm fedprox --algo_para 0.1 --num_rounds 20 --num_epochs 5 --learning_rate 0.1 --proportion 0.1 --batch_size 10 --eval_interval 1 --gpu 0 --logger simple_logger
# result_analyze
cd utils
python result_analysis.py --config example.yml --save_figure
```
<p float="left">
   <img src="https://github.com/WwZzz/myfigs/blob/master/result_analysis_example_1.png" width="230" />
   <img src="https://github.com/WwZzz/myfigs/blob/master/result_analysis_example_2.png" width="460" />
</p>

## systemic_simulator

## logger
The logger is used to log any variables of interest during the training period and directly inherits from the python's native module `logging`. To customize the logger, we suggest you to create a new logger in `utils.logger` by creating a new file `xxx_logger.py` and designing a new class `Logger` inheritting from `utils.logger.basic_logger.Logger`. When loading the logger, the system will firstly search for the particular logger firstly in `algorithm/fedxxx.py` according to the external inputted options. If not found, the system will load `logger` from `utils.logging.logger_name.Logger`. After loading the particular logger, the system initialization procedure will automatically register the instance server, clients, options as attributes of the logger, which is necessary for the logger to access to the running-time variables during the training process. To save the training time variables and finally make analysis on them, the logger provide a buffer `output` of type `collections.defaultdict(list)` to store everything of interest. To fill the output, the logger will initialize itself by calling its method `initialize`, and log varibales of interest at the beginning of each communication round by calling `log_per_round`. Finally, the logger will call `organize_output` before saving the buffer `output` as `.json` file. Therefore, the major behaviors of a logger totally depend on three parts: `initialize`, `log_per_round` and `organize_output`.

Now we take an example to show how to customize your logger. 

Example: `utils.logger.simple_logger.py`

```python
import utils.logger.basic_logger as bl
import numpy as np

class Logger(bl.Logger):
    """This logger only records metrics on validation dataset at each round"""

    def initialize(self, *args, **kwargs):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.clients:
            self.output['client_datavol'].append(len(c.train_data))
             
    def log_per_round(self, *args, **kwargs):
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()
        
    def organize_output(self, *args, **kwargs):
        """This method will be called before saving self.output"""
        self.output['meta'] = self.meta
        for key in self.output.keys():
            if '_dist' in key:
                self.output[key] = self.output[key][-1]
        return
```

At the initialize phase, the simple logger records the clients' local data sizes into `output`. And at each round, it tests the global model on each client's validation dataset, and records the metrics. After training is finished, this logger will organize the recorded round-wise results into expected ones, which can be directly used by `result_analysis.py`. For example, self.output['valid_loss'] is a round-wise validation loss value list. The analyser can write `.yml` file as 
```yaml
ploter:
  plot:
    - x: communication_round
      y: valid_loss
```
 Then, running the `result_analysis.py` with this yml file will plot the curves of selected records.
## fflow
