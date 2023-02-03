# FLGo: A Lightning Framework for Federated Learning

This repository is Paddle implementation for FLGo.

Our FLGo is a strong and reusable experimental platform for research on federated learning (FL) algorithm, which has provided a few easy-to-use modules to hold out for those who want to do various federated learning experiments. 

## Table of Contents
- [QuickStart](#QuickStart)
- [Architecture](#Architecture)
- [Citation](#Citation)
- [Contacts](#Contacts)
- [References](#References)

## QuickStart

**First**, follow these steps to generate a fedtask:
- Enter the parameters needed to divide the dataset in the gen_config.yml file, such as the benchmark name and the selected parttioner name, etc. In the example, we use the Diliclet method to divide the MNIST dataset.
- Run the command below to get the splited dataset MNIST:
```sh
# generate the splited dataset
python generate_fedtask.py
```

**Second**, run the command below to quickly get a result of the basic algorithm FedAvg on MNIST with a simple CNN:

```sh
python main.py --task B-mnist_classification_P-dir0.10_N-100_S-0 --model cnn --algorithm fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.1 --proportion 0.1 --batch_size 10 --eval_interval 1
# if using gpu, add the id of the gpu device as '--gpu id' to the end of the command like this
python main.py --task B-mnist_classification_P-dir0.10_N-100_S-0 --model cnn --algorithm fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.1 --proportion 0.1 --batch_size 10 --eval_interval 1 --gpu 0
```

The result will be stored in ` ./fedtask/B-mnist_classification_P-dir0.10_N-100_S-0/record/`.

**Third**, run the command below to get a visualization of the result.

You can modify the parameters of the result visualization in res_config.yml to visualize the results you need.
```sh
# change to the ./experiment folder
cd ../experiment
# visualize the results
python analyzer.py
```



### Reproduced FL Algorithms
| Method   |Reference|Publication|
|----------|---|---|
| FedAvg   |<a href='#refer-anchor-1'>[McMahan et al., 2017]</a>|AISTATS' 2017|
| FedAsync |<a href='#refer-anchor-2'></a>||
| FedBuff  |<a href='#refer-anchor-2'></a>||
| TiFL     |<a href='#refer-anchor-2'></a>||

### Dataset Partition
To divide the dataset using different partitions, make the following changes to the contents of the gen_config.yml file:
```
# I.I.D.
partitioner:
  name: IIDPartitioner
  para:
    num_clients: 100
    
# Imbalace & dirichlet
partitioner:
  name: DirichletPartitioner
  para:
    num_clients: 100
    imbalance: 0.1
    alpha: 0.1
```

### Options

Basic options:

* `task` is to choose the task of splited dataset. Options: name of fedtask (e.g. `mnist_classification_client100_dist0_beta0_noise0`).

* `algorithm` is to choose the FL algorithm. Options: `fedfv`, `fedavg`, `fedprox`, …

* `model` should be the corresponding model of the dataset. Options: `mlp`, `cnn`, `resnet18.`

Server-side options:

* `sample` decides the way to sample clients in each round. Options: `uniform` means uniformly, `md` means choosing with probability.

* `aggregate` decides the way to aggregate clients' model. Options: `uniform`, `weighted_scale`, `weighted_com`

* `num_rounds` is the number of communication rounds.

* `proportion` is the proportion of clients to be selected in each round. 

* `lr_scheduler` is the global learning rate scheduler.

* `learning_rate_decay` is the decay rate of the learning rate.

Client-side options:

* `num_epochs` is the number of local training epochs.

* `num_steps` is the number of local updating steps and the default value is -1. If this term is set larger than 0, `num_epochs` is not valid.

* `learning_rate ` is the step size when locally training.

* `batch_size ` is the size of one batch data during local training. `batch_size = full_batch` if `batch_size==-1` and `batch_size=|Di|*batch_size` if `1>batch_size>0`.

* `optimizer` is to choose the optimizer. Options: `SGD`, `Adam`.

* `weight_decay` is to set ratio for weight decay during the local training process.

* `momentum` is the ratio of the momentum item when the optimizer SGD taking each step. 

Real Machine-Dependent options:

* `seed ` is the initial random seed.

* `gpu ` is the id of the GPU device. (e.g. CPU is used without specifying this term. `--gpu 0` will use device GPU 0, and `--gpu 0 1 2 3` will use the specified 4 GPUs when `num_threads`>0. 

* `server_with_cpu ` is set False as default value,..

* `test_batch_size ` is the batch_size used when evaluating models on validation datasets, which is limited by the free space of the used device.

* `eval_interval ` controls the interval between every two evaluations. 

* `num_threads` is the number of threads in the clients computing session that aims to accelarate the training process.

* `num_workers` is the number of workers of the torch.utils.data.Dataloader

Additional hyper-parameters for particular federated algorithms:

* `algo_para` is used to receive the algorithm-dependent hyper-parameters from command lines. Usage: 1) The hyper-parameter will be set as the default value defined in Server.__init__() if not specifying this term, 2) For algorithms with one or more parameters, use `--algo_para v1 v2 ...` to specify the values for the parameters. The input order depends on the dict `Server.algo_para` defined in `Server.__init__()`.

Logger's setting

* `logger` is used to selected the logger that has the same name with this term.

* `log_level` shares the same meaning with the LEVEL in the python's native module logging.

* `log_file` controls whether to store the running-time information into `.log` in `fedtask/taskname/log/`, default value is false.

* `no_log_console` controls whether to show the running time information on the console, and default value is false.

## Architecture

We seperate the FL system into four parts:`algorithm`, `benchmark`, `experiment`, `fedtask`, `system_simulator` and `utils`.
```
├─ algorithm
│  ├─ fedavg.py                   //fedavg algorithm
│  ├─ ...
│  ├─ fedasync.py                 //the base class for asynchronous federated algorithms
│  └─ fedbase.py                  //the base class for federated algorithms
├─ benchmark
│  ├─ mnist_classification			//classification on mnist dataset
│  │  ├─ model                   //the corresponding model
│  |  └─ core.py                 //the core supporting for the dataset, and each contains three necessary classes(e.g. TaskGen, TaskReader, TaskCalculator)							
│  ├─ ...
│  ├─ RAW_DATA                   // storing the downloaded raw dataset
│  └─ toolkits						//the basic tools for generating federated dataset
│     ├─ cv                      // common federal division on cv
│     │  ├─ horizontal           // horizontal fedtask
│     │  │  └─ image_classification.py   // the base class for image classification
│     │  └─ ...
│     ├─ ...
│     ├─ base.py                 // the base class for all fedtask
│     ├─ partition.py            // the parttion class for federal division
│     └─ visualization.py        // visualization after the data set is divided
├─ experiment
│  ├─ logger                            //the class that records the experimental process
│  │  ├─ basic_logger.py		    	//the base logger class
│  |  └─ simple_logger.py				//a simple logger class
│  ├─ analyzer.py                  //the class for analyzing and printing experimental results
│  ├─ res_config.yml                  //hyperparameter file of analyzer.py
│  ├─ run_config.yml                  //hyperparameter file of runner.py
|  └─ runner.py                    //the class for generating experimental commands based on hyperparameter combinations and processor scheduling for all experimental commands
├─ fedtask
│  ├─ B-mnist_classification_P-dir0.10_N-100_S-0 //a fedtask
│  │  ├─ record							//the directionary of the running result
│  |  └─ data.json						//the splitted federated dataset (fedtask)
|  └─ ...
├─ system_simulator                     //system heterogeneity simulation module
│  ├─ base.py							//the base class for simulate system heterogeneity
│  ├─ default_simulator.py				//the default class for simulate system heterogeneity
|  └─ ...
├─ utils
│  ├─ fflow.py							//option to read, initialize,...
│  └─ fmodule.py						//model-level operators
├─ generate_fedtask.py					//generate fedtask
├─ gen_config.yml                       //hyperparameter file of generate_fedtask.py
├─ requirements.txt
└─ main.py                       //run this file to start easyFL system
```

### Benchmark

We have added many benchmarks covering several different areas such as CV, NLP, etc

### Fedtask

We define each task as a combination of the federated dataset of a particular distribution and the experimental results on it. The raw dataset is processed into .json file, following LEAF (https://github.com/TalwalkarLab/leaf). The architecture of the data.json file is described as below:  

```python
"""
# store the raw data
{
    'store': 'XY'
    'client_names': ['user0', ..., 'user99']
    'user0': {
       'dtrain': {'x': [...], 'y': [...]},
       'dvalid': {'x': [...], 'y': [...]},
     },...,
    'user99': {
       'dtrain': {'x': [...], 'y': [...]},
       'dvalid': {'x': [...], 'y': [...]},
     },
    'dtest': {'x':[...], 'y':[...]}
}
# store the index of data in the original dataset
{
    'store': 'IDX'
    'datasrc':{
        'class_path': 'torchvision.datasets',
        'class_name': dataset_class_name,
        'train_args': {
             'root': "str(raw_data_path)",
             ...
        },
        'test_args': {
             'root': "str(raw_data_path)",
             ...
         }
    }
    'client_names': ['user0', ..., 'user99']
    'user0': {
       'dtrain': [...],
       'dvalid': [...],
     },...,
    'dtest': [...]
}
"""
```

Run the file `./generate_fedtask.py` to get the splited dataset (.json file).

Since the task-specified models are usually orthogonal to the FL algorithms, we don't consider it an important part in this system. And the model and the basic loss function are defined in `./task/dataset_name/model_name.py`. Further details are described in `fedtask/README.md`.

### Algorithm
![image](https://github.com/WwZzz/myfigs/blob/master/fig0.png)
This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 


#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate_with()` them, who first `unpack()` the received package and then train the model with their local dataset by `train()`. After training the model, the clients `pack()` send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     

Further details of this module are described in `algorithm/README.md`.

### Experiment

The experiment module contains experiment command generation and scheduling operation, which can help FL researchers more conveniently conduct experiments in the field of federated learning.

### System_simulator

The system_simulator module is used to realize the simulation of heterogeneous systems, and we set multiple states such as network speed and availability to better simulate the system heterogeneity of federated learning parties.

### Utils

Utils is composed of commonly used operations: model-level operation (we convert model layers and parameters to dictionary type and apply it in the whole FL system). Further details are described in `utils/README.md`.

## Citation

Please cite our paper in your publications if this code helps your research.

```

```

## Contacts
Zheng Wang, zwang@stu.xmu.edu.cn

Xiaoliang Fan, fanxiaoliang@xmu.edu.cn, https://fanxlxmu.github.io

## References
<div id='refer-anchor-1'></div>

\[McMahan. et al., 2017\] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)

