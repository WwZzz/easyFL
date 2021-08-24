# easyFL: A Lightning Framework for Federated Learning

This repository is PyTorch implementation for paper [Federated Learning with Fair Averaging](https://fanxlxmu.github.io/publication/ijcai2021/) which is accepted by IJCAI-21 Conference.

Our easyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms. 

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
python main.py --task mnist_client100_dist0_beta0_noise0 --model cnn --method fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size 10 --train_rate 1 --eval_interval 1
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
      <td colspan="5">The rounds necessary for FedAVG to achieve 99% test accuracy on MNIST using CNN with E=5 (reported in <a href='#refer-anchor-1'>[McMahan. et al. 2017]</a>  /  ours)</td>
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
      <td>387  /  325</td>
      <td>50  /  91</td>
      <td>1181  /  1021</td>
      <td>956  /  771</td>
   </tr>
   <tr>
      <td>0.1</td>
      <td>339  /  203</td>
      <td>18  /  18 </td>
      <td>1100  /  453</td>
      <td>206  /  107</td>
   </tr>
   <tr>
      <td>0.2</td>
      <td>337  /  207</td>
      <td>18  /  19 </td>
      <td>978  /  525</td>
      <td>200  /  95</td>
   </tr>
   <tr>
      <td>0.5</td>
      <td>164  /  214</td>
      <td>18  /  18 </td>
      <td>1067  /  606</td>
      <td>261  /  105</td>
   </tr> 
   <tr>
      <td>1.0</td>
      <td>246  /  267</td>
      <td>16  /  18</td>
      <td>--  /  737</td>
      <td>97  /  90</td>
   </tr>
</table>
<table>
   <tr>
      <td colspan="7"> Accelarating FL Process by Increasing Parallelism For FedAVG on MNIST using CNN (20/100 clients per round)</td>
   </tr>
   <tr>
      <td>Num_threads</td>
      <td>1 </td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>15</td>
      <td>20</td>
   </tr>
   <tr>
      <td>Mean of time cost per round(s/r)</td>
      <td>19.5434 </td>
      <td>13.5733</td>
      <td>9.9935</td>
      <td>9.3092</td>
      <td>9.2885</td>
      <td><b>8.3867</b></td>
   </tr>
</table>

### Reproduced FL Algorithms
|Method|Reference|Publication|
|---|---|---|
|FedAvg|<a href='#refer-anchor-1'>[McMahan et al., 2017]</a>|AISTATS' 2017|
|FedProx|<a href='#refer-anchor-2'>[Li et al., 2020]</a>|MySys' 2020|
|FedFV|<a href='#refer-anchor-3'>[Wang et al., 2021]</a>|IJCAI' 2021|
|qFFL|<a href='#refer-anchor-4'>[Li et al., 2019]</a>|ICLR' 2020|
|AFL|<a href='#refer-anchor-5'>[Mohri et al., 2019]</a>|ICML' 2019|
|FedMGDA+|<a href='#refer-anchor-6'>[Hu et al., 2020]</a>|pre-print|
|FedFA|<a href='#refer-anchor-7'>[Huang et al., 2020]</a>|pre-print|
|...||
### Options

Basic options:

* `task` is to choose the task of splited dataset. Options: name of fedtask (e.g. `mnist_client100_dist0_beta0_noise0`).

* `method ` is to choose the FL algorithm. Options: `fedfv`, `fedavg`, `fedprox`, …

* `model` should be the corresponding model of the dataset. Options: `mlp`, `cnn`, `resnet18.`

Server-side options:

* `sample` decides the way to sample clients in each round. Options: `uniform` means uniformly, `md` means choosing with probability.

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

* `num_threads` is the number of threads in the clients computing session that aims to accelarate the training process.

Additional hyper-parameters for particular federated algorithms:
* `mu` is the parameter for FedProx.
* `alpha` is the parameter for FedFV.
* `tau` is the parameter for FedFV.
* ...

Each additional parameter can be defined in `./utils/fflow.read_option`

## Architecture

We seperate the FL system into four parts: `benchmark`, `fedtask`, `method` and `utils`.

### Benchmark

This module is to generate `fedtask` by partitioning the particular distribution data through `generate_fedtask.py`. To generate different `fedtask`, there are three parameters: `dist`, `num_clients `, `beta`. `dist` denotes the distribution type (e.g. `0` denotes iid and balanced distribution, `1` denotes niid-label-quantity and balanced distribution). `num_clients` is the number of clients participate in FL system, and `beta` controls the degree of non-iid for different  `dist`. Each dataset can correspond to differrent models (mlp, cnn, resnet18, …). We refer to <a href='#refer-anchor-1'>[McMahan et al., 2017]</a>, <a href='#refer-anchor-2'>[Li et al., 2020]</a>, <a href='#refer-anchor-8'>[Li et al., 2021]</a>, <a href='#refer-anchor-4'>[Li et al., 2019]</a>, <a href='#refer-anchor-9'>[Caldas et al., 2018]</a> when realizing this module. Further details are described in `benchmark/README.md`.

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
![image](https://github.com/WwZzz/myfigs/blob/master/fig0.png)
This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 


#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate_with()` them, who first `unpack()` the received package and then train the model with their local dataset by `train()`. After training the model, the clients `pack()` send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     

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
Since we've made great changes on the latest version, to fully reproduce the reported results in our paper [Federated Learning with Fair Averaging](https://fanxlxmu.github.io/publication/ijcai2021/), please use another branch `easyFL v1.0` of this project.

## References
<div id='refer-anchor-1'></div>

\[McMahan. et al., 2017\] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)

<div id='refer-anchor-2'></div>

\[Li et al., 2020\] [Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. arXiv e-prints, page arXiv:1812.06127, 2020.](https://arxiv.org/abs/1812.06127)

<div id='refer-anchor-3'></div>

\[Wang et al., 2021\] [Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang and Rongshan Yu. Federated Learning with Fair Averaging. arXiv e-prints, page arXiv:2104.14937, 2021.](https://arxiv.org/abs/2104.14937)


<div id='refer-anchor-4'></div>

\[Li et al., 2019\] [ Tian Li, Maziar Sanjabi, and Virginia Smith. Fair resource allocation in federated learning. CoRR, abs/1905.10497, 2019.](https://arxiv.org/abs/1905.10497)

<div id='refer-anchor-5'></div>

\[Mohri et al., 2019\] [Mehryar Mohri, Gary Sivek, and Ananda Theertha Suresh. Agnostic federated learning. CoRR, abs/1902.00146, 2019.](https://arxiv.org/abs/1902.00146)

<div id='refer-anchor-6'></div>

\[Hu et al., 2020\] [Zeou Hu, Kiarash Shaloudegi, Guojun Zhang, and Yaoliang Yu. Fedmgda+: Federated learning meets multi-objective optimization. arXiv e-prints, page arXiv:2006.11489, 2020.](https://arxiv.org/abs/2006.11489)

<div id='refer-anchor-7'></div>

\[Huang et al., 2020\] [Wei Huang, Tianrui Li, Dexian Wang, Shengdong Du, and Junbo Zhang. Fairness and accuracy in federated learning. arXiv e-prints, page arXiv:2012.10069, 2020.](https://arxiv.org/abs/2012.10069) 

<div id='refer-anchor-8'></div>

\[Li et al., 2021\][Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng. Federated Learning on Non-IID Data Silos: An Experimental Study. arXiv preprint arXiv:2102.02079, 2021.](https://arxiv.org/abs/2102.02079)

<div id='refer-anchor-9'></div>

\[Caldas et al., 2018\] [Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, Ameet Talwalkar. LEAF: A Benchmark for Federated Settings. arXiv preprint arXiv:1812.01097, 2018.](https://arxiv.org/abs/1812.01097)
