# easyFL: A Lightning Framework for Federated Learning

This repository is PyTorch implementation for paper [Federated Learning with Fair Averaging](https://arxiv.org/abs/2104.14937) which is accepted by IJCAI-21 Conference.

Our easyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-beginner to quickly realize and compare popular centralized federated learning algorithms. 

## Table of Contents
- [Requirements](#Requirements)
- [QuickStart](#QuickStart)
- [Architecture](#Architecture)
- [Citation](#Citation)

## Requirements

The model is implemented using Python3 with dependencies below:

```
numpy>=1.17.2
pytorch>=1.3.1
torchvision>=0.4.2
cvxopt>=1.2.0
scipy>=1.3.1
matplotlib>=3.1.1
prettytable>=2.1.0
ujson>=4.0.2
```

## QuickStart

**First**, run the command below to get the splited dataset MNIST:

```sh
# generate the splited dataset
python generate_fedtask.py
```

**Second**, run the command below to quickly get a result of the basic algorithm FedAvg on MNIST with a simple CNN:

```sh
python main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedavg --num_rounds 20 --num_epochs 5 --proportion 0.2 --batch_size -1 --train_rate 1 --eval_interval 1
```

The result will be stored in ` ./fedtask/mnist_client100_dist0_beta0_noise0/record/`.

**Third**, run the command below to get a visualization of the result.

```sh
# change to the ./utils folder
cd ../utils
# visualize the results
python result_analysis.py
```
### Performance


<table>
   <tr>
      <td colspan="5">The rounds necessary for FedAVG to achieve 99% accuracy on MNIST using CNN with E=5 (reported in [<sup>1</sup>](#refer-anchor-1)  /  ours)</td>
   </tr>
   <tr>
      <td rowspan="2">Proportion</td>
      <td colspan="2">iid</td>
      <td colspan="2">non-iid</td>
   </tr>
   <tr>
      <td>B=FULL</td>
      <td>B=10</td>
      <td>B=FULL</td>
      <td>B=10</td>
   </tr>
   <tr>
      <td>0.0</td>
      <td>387  /  </td>
      <td>50  /  91</td>
      <td>1181  /  </td>
      <td>956  /  </td>
   </tr>
   <tr>
      <td>0.1</td>
      <td>339  / </td>
      <td>18  /  18 </td>
      <td>1100  / </td>
      <td>206  / </td>
   </tr>
   <tr>
      <td>0.2</td>
      <td>337  / </td>
      <td>18  /  19 </td>
      <td>978  / </td>
      <td>200  / </td>
   </tr>
   <tr>
      <td>0.5</td>
      <td>164  / </td>
      <td>18  /  18 </td>
      <td>1067  / </td>
      <td>261  / </td>
   </tr> 
   <tr>
      <td>1.0</td>
      <td>246  / </td>
      <td>16  / </td>
      <td>--  /</td>
      <td>97  / </td>
   </tr>
</table>


### Options

Basic options:

* `task` is to choose the task of splited dataset. Options: name of fedtask (e.g. `mnist_client100_dist0_beta0_noise0`).

* `method ` is to choose the FL algorithm. Options: `fedfv`, `fedavg`, `fedprox`, …

Server-side options:

* `model` should be the corresponding model of the dataset. Options: `mlp`, `cnn`, `resnet18.`

* `sample` decides the way to sample clients in each round. Options: `uniform` means uniformly, `md` means choosing with probility.

* `aggregate` decides the way to aggregate clients' model. Options: `uniform`, `weighted_scale`, `weighted_com`

* `num_rounds` is the number of communication rounds.

* `proportion` is the proportion of clients to be selected in each round. 

* `lr_scheduler` is the global learning rate scheduler.

* `learning_rate_decay` is the decay rate of the global learning rate.

Client-side options:

* `num_epochs` is the number of local training epochs.

* `learning_rate ` is the step size when locally training.

* `batch_size ` is the size of one batch data during local training.

* `optimizer` is to choose the optimizer. Options: `SGD`, `Adam`.

* `momentum` is the ratio of the momentum item when the optimizer SGD taking each step. 

Other options:

* `seed ` is the initial random seed.

* `gpu ` is the id of the GPU device, `-1` for CPU.

* `eval_interval ` controls the interval between every two evaluations. 

* `train_rate` is the proportion of the train set and validation set in local dataset.

* `drop` controls the dropout of clients after being selected in each communication round according to distribution Beta(drop,1). The larger this term is, the more possible for clients to drop.

Additional hyper-parameters for particular federated algorithms:
* `mu` is the parameter for FedProx.
* `alpha` is the parameter for FedFV.
* `tau` is the parameter for FedFV.
* ...

Each additional parameter can be defined in `./utils/fflow.read_option`

## Architecture

We seperate the FL system into four parts: `benchmark`, `fedtask`, `method` and `utils`.

![](Architecture.bmp)

### Benchmark

This module is to generate `fedtask` by partitioning the particular distribution data through `generate_fedtask.py`. To generate different `fedtask`, there are three parameters: `dist`, `num_clients `, `beta`. `dist` denotes the distribution type (e.g. `0` denotes iid and balanced distribution, `1` denotes niid-label-quantity and balanced distribution). `num_clients` is the number of clients participate in FL system, and `beta` controls the degree of non-iid for different  `dist`. Each dataset can correspond to differrent models (mlp, cnn, resnet18, …). Further details are described in `benchmark/README.md`.

### Fedtask
We define each task as a combination of the federated dataset of a particular distribution and the experimental results on it. The raw dataset is processed into .json file, following LEAF (https://github.com/TalwalkarLab/leaf). The architecture of the .json file is described as below:  

```python
"""
{
    'meta':{
        'benchmark': 'mnist',						//dataset name
        'num_clients': 100,							//the number of clients
        'dist': 0,									//the distribution of data
        'beta': 0,									//the parameter of distribution
    },
    'clients': {
        'user0': {
            'dtrain': {'x': [...], 'y': [...]},
            'dvalid': {'x': [...], 'y': [...]},
            'dvol': 600,
        },...,
        'user99': {
            'dtrain': {'x': [...], 'y': [...]},
            'dvalid': {'x': [...], 'y': [...]},
            'dvol': 600,
        },
    },
    'dtest': {'x':[...], 'y':[...]}
}
"""
```

Run the file `./generate_fedtask.py` to get the splited dataset (.json file).

Since the task-specified models are usually orthogonal to the FL algorithms, we don't consider it an important part in this system. And the model and the basic loss function are defined in `./task/dataset_name/model_name.py`. Further details are described in `fedtask/README.md`.

### Method

This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 


#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate()` with them, who train the model with their local dataset by `train()`. After training the model, the clients send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     

Further details of this module are described in `method/README.md`.

### Utils
Utils is composed of commonly used operations: model-level operation (we convert model layers and parameters to dictionary type and apply it in the whole FL system), the flow controlling of the framework in and the supporting visualization templates to the result. To visualize the results, please run `./utils/result_analysis.py`. Further details are described in `utils/README.md`.

## Citation

Please cite our paper in your publications if this code helps your research.

```
@article{wang2021federated,
  title={Federated Learning with Fair Averaging},
  author={Wang, Zheng and Fan, Xiaoliang and Qi, Jianzhong and Wen, Chenglu and Wang, Cheng and Yu, Rongshan},
  journal={arXiv preprint arXiv:2104.14937},
  year={2021}
}te
```
## Remark
Since we've made great changes on the latest version, to fully reproduce the reported results in our paper [Federated Learning with Fair Averaging](https://arxiv.org/abs/2104.14937), please use another branch `easyFL v1.0` of this project.

## References
<div id="refer-anchor-1"></div>
[1] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)
