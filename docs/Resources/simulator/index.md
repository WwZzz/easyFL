
# Usage
Each simulator is a new defined class inheriting from ```flgo.simulator.base.BasicSimulator```. To use simulators here, please follow the two steps below

1. Copy the source code of simulator

2. Write codes as follows to use it

```python
import flgo
import flgo.simulator.base
import mnist_classification

#### 1. Paste Souce code here ##########################
class MySimulator(flgo.simulator.base.BasicSimulator):
    ...
########################################################
task = './test_mnist_download'
flgo.gen_task({'benchmark': mnist_classification}, task)
import flgo.algorithm.fedavg as fedavg

##### 2. Specify simulator here by para Simulator in flgo.init######
flgo.init(task, fedavg, {'gpu':0}, Simulator=MySimulator).run()
```
