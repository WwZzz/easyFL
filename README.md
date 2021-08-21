# Federated Learning  with Fair Averaging

This repository is PyTorch implementation for paper [Federated Learning  with Fair Averaging](https://arxiv.org/abs/2104.14937) which is accepted by IJCAI-21 Conference.

Our EasyFL is a strong reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-beginner to quickly realize and compare popular centralized federated learning algorithms.

## Table of Contents

- [Abstract](#Abstract)
- [Requirements](#Requirements)
- [Quick Start](#Quick Start)
- [Architecture](#Architecture)
- [Citation](#Citation)

## Abstract

Fairness has emerged as a critical problem in federated learning (FL). In this work, we identify a cause of unfairness in FL -- *conflicting* gradients with large differences in the magnitudes. To address this issue, we propose the federated fair averaging (FedFV) algorithm to mitigate potential conflicts among clients before averaging their gradients. We first use the cosine similarity to detect gradient conflicts, and then iteratively eliminate such conflicts by modifying both the direction and the magnitude of the gradients. We further show the theoretical foundation of FedFV to mitigate the issue conflicting gradients and converge to Pareto stationary solutions. Extensive  experiments on a suite of federated datasets confirm that FedFV compares favorably against state-of-the-art methods in terms of fairness, accuracy and efficiency.

For more details, please read our full paper: https://arxiv.org/abs/2104.14937

## Requirements

The model is implemented using Python3 with dependencies below:

```
pytorch=1.3.1
torchvision=0.4.2
cvxopt=1.2.0 (This is required for fedmgda+)
```

## Quick Start

**First**, run the command below to get the splited dataset MNIST:

```sh
# change to the ./benchmark folder
cd ./benchmark
# generate the splited dataset
python generate_fedtask.py
```

**Second**, run the command below to quickly get a result of the basic algorithm FedAvg on MNIST with a simple CNN:

```sh
python main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedavg --num_rounds 20 --num_epochs 5 --proportion 0.2 --batch_size -1 --train_rate 1 --eval_interval 1
```

Or use the command below to get a result of our algorithm FedFV on MNIST with CNN:

```sh
python main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedfv --alpha 0.2 --tau 3 --num_rounds 20 --num_epochs 5 --proportion 0.2 --batch_size -1 --train_rate 1 --eval_interval 1
```

The result will be stored in ` ./fedtask/mnist_client100_dist0_beta0_noise0/record/`.

**Third**, run the command below to get a visualization of the result.

```sh
# change to the ./utils folder
cd ../utils
# visualize the results
python result_analysis.py
```

### Options

Basic options:

* `task` is to choose the task of splited dataset. Options: name of fedtask (e.g. `mnist_client100_dist0_beta0_noise0`).

* `method ` is to choose the FL algorithm. Options: `fedfv`, `fedavg`, `fedprox`, …

Server-side options:

* `model` should be the corresponding model of the dataset. Options: `mlp`, `cnn`, `resnet18.`

* `sample` decides the way to sample clients in each round. Options: `uniform` means uniformly, `prob` means choosing with probility.

* `aggregate` decides the way to aggregate clients' model. Options: `uniform`, `weighted_scale`, `weighted_com`

* `num_rounds` is the number of communication rounds.

* `proportion ` is the proportion of clients to be selected in each round. 

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

Additional hyper-parameters

* `alpha` is the parameter for FedFV.
* `tau` is the parameter for FedFV.

Each additional parameter can be defined in `./utils/tools.read_option`

## Architecture

We seperate the FL system into four parts: `benchmark`, `fedtask`, `method` and `utils`.

![](Architecture.bmp)

### Benchmark

This module is to generate `fedtask` by partitioning the particular distribution data through `generate_fedtask.py`. To generate different `fedtask`, there are three parameters: `dist`, `num_clients `, `beta`. `dist` denotes the distribution type, `0` denotes iid, `1` denotes`num_clients` is the number of clients participate in FL system, and `beta` is the number of the shards (splitted by the sorted labels) owned by each client. Each dataset can correspond to differrent model (mlp, cnn, resnet18, …)

### Fedtask
We define each task as a combination of the `dataset`, the corresponding model, and the basic loss function. The raw dataset is processed into .json file, following LEAF (https://github.com/TalwalkarLab/leaf). The architecture of the .json file is described as below:  

```json
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
```

The raw dataset should be download into ./task/dataset_name/data/raw_data, and then run the file `./benchmark/generate_fedtask.py` to get the splited dataset (.json file).

Since the task-specified models are usually orthogonal to the FL algorithms, we don't consider it an important part in this system. And the model and the basic loss function are defined in `./task/dataset_name/model_name.py`.

### Method

This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 

#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate()` with them, who train the model with their local dataset by `train()`. After training the model, the clients send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     

### Utils
Utils is composed of commonly used operations: model-level operation (we convert model layers and parameters to dictionary type and apply it in the whole FL system), the initialization of the framework in and the supporting visualization templates to the result. To visualize the results, please run `./utils/result_analysis.py`.

## Citation

Please cite our paper in your publications if this code helps your research.

```
@article{wang2021federated,
  title={Federated Learning with Fair Averaging},
  author={Wang, Zheng and Fan, Xiaoliang and Qi, Jianzhong and Wen, Chenglu and Wang, Cheng and Yu, Rongshan},
  journal={arXiv preprint arXiv:2104.14937},
  year={2021}
}
```

