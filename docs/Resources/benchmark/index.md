# Usage
Each benchmark is compressed as a .zip file. To use benchmarks here, please follow the three steps below

1. Download the benchmark .zip file

2. Decompress the .zip file into the currently working project directory

3. Use the decompressed directory as a python module, and generate federated task from it

## Example on MNIST
1. download mnist_classification.zip from [here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/image/classification/mnist_classification.zip) and decompress it into the currently working project directory.

2. Write codes as follows to use it

```python
import flgo
import mnist_classification

task = './test_mnist_download'
flgo.gen_task({'benchmark': mnist_classification}, task)

import flgo.algorithm.fedavg as fedavg

flgo.init(task, fedavg, {'gpu':0}).run()

```
