<div align="center">
  <img src='docs/img/flgo_icon.png'  width="200"/>
<h1>FLGo: A Lightning Framework for Federated Learning</h1>

<!-- [![article-red](https://img.shields.io/badge/pypi-v0.0.11-red)](https://pypi.org/project/flgo/)
[![arxiv-orange](https://img.shields.io/badge/pypi-v0.0.11-red)](https://pypi.org/project/flgo/) -->
[![PyPI](https://img.shields.io/badge/pypi-v0.4.4-yellow)](https://pypi.org/project/flgo/)
[![docs](https://img.shields.io/badge/docs-maintaining-green)](https://flgo-xmu.github.io/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/WwZzz/easyFL/blob/FLGo/LICENSE)


</div>


<!-- ## Major Feature -->

[//]: # (## Table of Contents)

[//]: # (- [Introduction]&#40;#Introduction&#41;)

[//]: # (- [QuickStart]&#40;#Quick Start with 3 lines&#41;)

[//]: # (- [Architecture]&#40;#Architecture&#41;)

[//]: # (- [Citation]&#40;#Citation&#41;)

[//]: # (- [Contacts]&#40;#Contacts&#41;)

[//]: # (- [References]&#40;#References&#41;)

[//]: # (- )
# Introduction
FLGo is a library to conduct experiments about Federated Learning (FL). It is strong and reusable for research on FL, providing comprehensive easy-to-use modules to hold out for those who want to do various federated learning experiments. 

## Installation 
* Install FLGo through pip. It's recommended to install pytorch by yourself before installing this library.  
```sh
pip install flgo --upgrade
```
* Install FLGo through git
```sh
git clone https://github.com/WwZzz/easyFL.git
```
## Join Us :smiley:
Welcome to our FLGo's WeChat group/QQ Group for more technical discussion.

<center>
<!-- <img src="https://github.com/user-attachments/assets/230247bc-8fce-4821-901b-d0e22ca360fd" width=180/> -->

  <img src="https://github.com/user-attachments/assets/39070ba7-4752-46ec-b8b4-3d5591992595" width=180/>
</center>

Group Number: 838298386

Tutorials in Chinese can be found [here](https://www.zhihu.com/column/c_1618319253936984064)
# News
**[2024.9.20]** We present a comprehensive benchmark gallery [here](https://github.com/WwZzz/FLGo-Bench)

**[2024.8.01]** Improving efficiency by sharing datasets across multiple processes within each task in the shared memory 

# Quick Start with 3 lines :zap:
```python
import flgo
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import flgo.algorithm.fedavg as fedavg

# Line 1: Create a typical federated learning task
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=100), './my_task')

# Line 2: Running FedAvg on this task
fedavg_runner = flgo.init('./my_task', fedavg, {'gpu': [0,], 'num_rounds':20, 'num_epochs': 1})

# Line 3: Start Training
fedavg_runner.run()
```
We take a classical federated dataset, Federated MNIST, as the example. The MNIST dataset is splitted into 100 parts identically and independently.

Line 1 creates the federated dataset as `./my_task` and visualizes it in `./my_task/res.png`
![result](docs/img/getstart_fig1.png)

Lines 2 and 3 start the training procedure and outputs information to the console
```
2024-04-15 02:30:43,763 fflow.py init [line:642] INFO PROCESS ID:	552206
2024-04-15 02:30:43,763 fflow.py init [line:643] INFO Initializing devices: cuda:0 will be used for this running.
2024-04-15 02:30:43,763 fflow.py init [line:646] INFO BENCHMARK:	flgo.benchmark.mnist_classification
2024-04-15 02:30:43,763 fflow.py init [line:647] INFO TASK:			./my_task
2024-04-15 02:30:43,763 fflow.py init [line:648] INFO MODEL:		flgo.benchmark.mnist_classification.model.cnn
2024-04-15 02:30:43,763 fflow.py init [line:649] INFO ALGORITHM:	fedavg
2024-04-15 02:30:43,774 fflow.py init [line:688] INFO SCENE:		horizontal FL with 1 <class 'flgo.algorithm.fedbase.BasicServer'>, 100 <class 'flgo.algorithm.fedbase.BasicClient'>
2024-04-15 02:30:47,851 fflow.py init [line:705] INFO SIMULATOR:	<class 'flgo.simulator.default_simulator.Simulator'>
2024-04-15 02:30:47,853 fflow.py init [line:718] INFO Ready to start.
...
2024-04-15 02:30:52,466 fedbase.py run [line:253] INFO --------------Round 1--------------
2024-04-15 02:30:52,466 simple_logger.py log_once [line:14] INFO Current_time:1
2024-04-15 02:30:54,402 simple_logger.py log_once [line:28] INFO test_accuracy                 0.6534
2024-04-15 02:30:54,402 simple_logger.py log_once [line:28] INFO test_loss                     1.5835
...
```
* **Show Training Result (optional)**
```python
import flgo.experiment.analyzer as fea
# Create the analysis plan
analysis_plan = {
    'Selector':{'task': './my_task', 'header':['fedavg',], },
    'Painter':{'Curve':[{'args':{'x':'communication_round', 'y':'val_loss'}}]},
}

fea.show(analysis_plan)
```
Each training result will be saved as a record under `./my_task/record`. We can use the built-in analyzer to read and show it. 

![result](docs/img/getstart_fig2.png)

# Why Using FLGo? :hammer_and_wrench:
## Simulate Real-World System Heterogeneity :iphone:
![flgo_simulator](https://raw.githubusercontent.com/WwZzz/myfigs/master/overview_flgo_async.png)
Our FLGo supports running different algorithms in virtual environments like real-world. For example, clients in practice may 
* *be sometime inavailable*, 
* *response to the server very slow*, 
* *accidiently lose connection*, 
* *upload incomplete model updates*,
* ...

All of these behavior can be easily realized by integrating a simple `Simulator` to the runner like
```python
import flgo
from flgo.simulator import ExampleSimulator
import flgo.algorithm.fedavg as fedavg

fedavg_runner = flgo.init('./my_task', fedavg, {'gpu': [0,]}, simulator=ExampleSimulator)
fedavg_runner.run()
```

`Simulator` is fully customizable and can fairly reflect the impact of system heterogeneity on different algorithms. Please refer to [Paper](https://arxiv.org/abs/2306.12079) or [Tutorial](https://flgo-xmu.github.io/Tutorials/4_Simulator_Customization/) for more details.
## Comprehensive Benchmarks :family_woman_woman_boy_boy:
FLGo provides more than 50 benchmarks across different data types, different communication topology,...
<table>
    <tr>
        <td>
        <td>Task
        <td>Scenario
        <td>Datasets
        <td>
    </tr>
    <tr>
        <td rowspan=3>CV
        <td>Classification
        <td>Horizontal & Vertical
        <td>CIFAR10\100, MNIST, FashionMNIST,FEMNIST, EMNIST, SVHN
        <td>
    </tr>
    <tr>
        <td>Detection
        <td>Horizontal
        <td>Coco, VOC
        <td>
    </tr>
    <tr>
        <td>Segmentation
        <td>Horizontal
        <td>Coco, SBDataset
        <td>
    </tr>
    <tr>
        <td rowspan=3>NLP
        <td>Classification
        <td>Horizontal
        <td>Sentiment140, AG_NEWS, sst2
        <td>
    </tr>
    <tr>
        <td>Text Prediction
        <td>Horizontal
        <td>Shakespeare, Reddit
        <td>
    </tr>
    <tr>
        <td>Translation
        <td>Horizontal
        <td>Multi30k
        <td>
    </tr>
    <tr>
        <td rowspan=3>Graph
        <td>Node Classification
        <td>Horizontal
        <td>Cora, Citeseer, Pubmed
        <td>
    </tr>
    <tr>
        <td>Link Prediction
        <td>Horizontal
        <td>Cora, Citeseer, Pubmed
        <td>
    </tr>
    <tr>
        <td>Graph Classification
        <td>Horizontal
        <td>Enzymes, Mutag
        <td>
    </tr>
    <tr>
        <td>Recommendation
        <td>Rating Prediction
        <td>Horizontal & Vertical
        <td>Ciao, Movielens, Epinions, Filmtrust, Douban
        <td>
    </tr>
    <tr>
        <td>Series
        <td>Time series forecasting
        <td>Horizontal
        <td>Electricity, Exchange Rate
        <td>
    </tr>
    <tr>
        <td>Tabular
        <td>Classification
        <td>Horizontal
        <td>Adult, Bank Marketing
        <td>
    </tr>
    <tr>
        <td>Synthetic
        <td>Regression
        <td>Horizontal
        <td>Synthetic, DistributedQP, CUBE
        <td>
    </tr>
</table>

### Usage
Each benchmark can be used to generate federated tasks that denote distributed scenes with specific data distributions like

```python
import flgo
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
import flgo.algorithm.fedavg as fedavg

task = './my_first_cifar' # task name
flgo.gen_task_by_(cifar10, fbp.IIDPartitioner(num_clients=10), task) # generate task from benchmark with partitioner
flgo.init(task, fedavg, {'gpu':0}).run()
```


## Visualized Data Heterogeneity :eyes:
We realize data heterogeneity by flexible partitioners. These partitioners can be easily combined with `benchmark` to generate federated tasks with different data distributions.
```python
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp
```
#### Dirichlet(α) of labels
```python
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=0.1), 'dir0.1_cifar')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0), 'dir1.0_cifar')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=5.0), 'dir5.0_cifar')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=10.0), 'dir10.0_cifar')
```
![cifar10_dir](https://github.com/WwZzz/myfigs/blob/master/readme_flgo_dir_cifar10.png?raw=true)
#### Controllable Data Imbalance
```python
# set imbalance=0.1, 0.3, 0.6 or 1.0
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0, imbalance=0.1), 'dir1.0_cifar_imb0.1')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0, imbalance=0.3), 'dir1.0_cifar_imb0.3')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0, imbalance=0.6), 'dir1.0_cifar_imb0.6')
flgo.gen_task_by_(cifar10, fbp.DirichletPartitioner(num_clients=100, alpha=1.0, imbalance=1.0), 'dir1.0_cifar_imb1.0')
```
![cifar10_imb](https://github.com/WwZzz/myfigs/blob/master/readme_flgo_imb_cifar10.png?raw=true)
#### Heterogeneous Label Diversity
```python
flgo.gen_task_by_(cifar10, fbp.DiversityPartitioner(num_clients=100, diversity=0.1), 'div0.1_cifar')
flgo.gen_task_by_(cifar10, fbp.DiversityPartitioner(num_clients=100, diversity=0.3), 'div0.3_cifar')
flgo.gen_task_by_(cifar10, fbp.DiversityPartitioner(num_clients=100, diversity=0.6), 'div0.6_cifar')
flgo.gen_task_by_(cifar10, fbp.DiversityPartitioner(num_clients=100, diversity=1.0), 'div1.0_cifar')
```
![cifar10_div](https://github.com/WwZzz/myfigs/blob/master/readme_flgo_div_cifar10.png?raw=true)

`Partitioner` is also customizable in flgo. We have provided a detailed example in this [Tutorial](https://flgo-xmu.github.io/Tutorials/3_Benchmark_Customization/3.7_Data_Heterogeneity/).

## Reproduction of Algorithms from TOP-tiers and Journals :1st_place_medal:
We have realized more than 50 algorithms from TOP-tiers and Journals. The algorithms are listed as below

#### Classical FL & Data Heterogeneity

| Method   | Reference | Publication |
|----------|-----------|-------------|
| FedAvg   | [link](http://arxiv.org/abs/1602.05629)  | AISTAS2017  |
| FedProx  | [link](http://arxiv.org/abs/1812.06127)  | MLSys 2020  |
| Scaffold | [link](http://arxiv.org/abs/1910.06378)  | ICML 2020   |
| FedDyn   | [link](http://arxiv.org/abs/2111.04263)  | ICLR 2021   |
| MOON     | [link](http://arxiv.org/abs/2103.16257)  | CVPR 2021   |
| FedNova  | [link](http://arxiv.org/abs/2007.07481)  | NIPS 2021   |
| FedAvgM  | [link](https://arxiv.org/abs/1909.06335)  | arxiv       |
| GradMA   |[link](http://arxiv.org/abs/2302.14307)  | CVPR 2023   |


#### Personalized FL
| Method          | Reference                                                       | Publication        |
|-----------------|-----------------------------------------------------------------|--------------------|
| Standalone      | [link](http://arxiv.org/abs/1602.05629)                         | -                  |
| FedAvg+FineTune | -                                                               | -                  |
| Ditto           | [link](http://arxiv.org/abs/2007.14390)                         | ICML 2021          |
| FedALA          | [link](http://arxiv.org/abs/2212.01197)                         | AAAI 2023          |
| FedRep          | [link](http://arxiv.org/abs/2102.07078)                         | ICML 2021          | 
| pFedMe          | [link](http://arxiv.org/abs/2006.08848)                         | NIPS 2020          | 
| Per-FedAvg      | [link](http://arxiv.org/abs/2002.07948)                         | NIPS 2020          | 
| FedAMP          | [link](http://arxiv.org/abs/2007.03797)                         | AAAI 2021          | 
| FedFomo         | [link](http://arxiv.org/abs/2012.08565)                         | ICLR 2021          | 
| LG-FedAvg       | [link](http://arxiv.org/abs/2001.01523)                         | NIPS 2019 workshop | 
| pFedHN          | [link](https://proceedings.mlr.press/v139/shamsian21a.html)     | ICML 2021          | 
| Fed-ROD         | [link](https://openreview.net/forum?id=I1hQbx10Kxn)             | ICLR 2023          | 
| FedPAC          | [link](http://arxiv.org/abs/2306.11867)                         | ICLR 2023          | 
| FedPer          | [link](http://arxiv.org/abs/1912.00818)                         | AISTATS 2020       | 
| APPLE           | [link](https://www.ijcai.org/proceedings/2022/301)              | IJCAI 2022         | 
| FedBABU         | [link](http://arxiv.org/abs/2106.06042)                         | ICLR 2022          | 
| FedBN           | [link](https://openreview.net/pdf?id=6YEQUn0QICG)               | 	ICLR 2021         |
| FedPHP          | [link](https://dl.acm.org/doi/abs/10.1007/978-3-030-86486-6_36) | ECML/PKDD 2021     |
| APFL            | [link](http://arxiv.org/abs/2003.13461)                         | arxiv              |
| FedProto        | [link](https://ojs.aaai.org/index.php/AAAI/article/view/20819)  | AAAI 2022          |
| FedCP           | [link](http://arxiv.org/abs/2307.01217)                         | KDD 2023           |
| GPFL            | [link](http://arxiv.org/abs/2308.10279)                         | ICCV 2023          |
| pFedPara        | [link](http://arxiv.org/abs/2108.06098)                         | ICLR 2022          |
| FedFA           | [link](https://arxiv.org/abs/2301.12995)                        | ICLR 2023          |

#### Fairness-Aware FL
| Method   |Reference| Publication               |
|----------|---|---------------------------|
| AFL      |[link](http://arxiv.org/abs/1902.00146)   | ICML 2019                 |
| FedFv    |[link](http://arxiv.org/abs/2104.14937)   | IJCAI 2021                |
| FedFa    |[link](http://arxiv.org/abs/2012.10069)   | Information Sciences 2022 |
| FedMgda+ |[link](http://arxiv.org/abs/2006.11489)   | IEEE TNSE 2022            |
| QFedAvg  |[link](http://arxiv.org/abs/1905.10497)   | ICLR 2020                 |

#### Asynchronous FL
| Method   |Reference| Publication  |
|----------|---|--------------|
| FedAsync |[link](http://arxiv.org/abs/1903.03934)   | arxiv        |
| FedBuff  |[link](http://arxiv.org/abs/2106.06639)   | AISTATS 2022 |
| CA2FL    |[link](https://openreview.net/forum?id=4aywmeb97I)   | ICLR2024     |
#### Client Sampling & Heterogeneous Availability
| Method            |Reference| Publication  |
|-------------------|---|--------------|
| MIFA              |[link](http://arxiv.org/abs/2106.04159)   | NeurIPS 2021 |
| PowerofChoice     |[link](http://arxiv.org/abs/2010.13723)   | arxiv        |
| FedGS             |[link](https://arxiv.org/abs/2211.13975)   | AAAI 2023    |
| ClusteredSampling |[link](http://arxiv.org/abs/2105.05883)   | ICML 2021    |


#### Capacity Heterogeneity
| Method           | Reference                                                                                                                    | Publication  |
|------------------|------------------------------------------------------------------------------------------------------------------------------|--------------|
| FederatedDropout | [link](http://arxiv.org/abs/1812.07210)                                                                                      | arxiv        |
| FedRolex         | [link](https://openreview.net/forum?id=OtxyysUdBE)                                                                           | NIPS 2022    |
| Fjord            | [link](https://proceedings.neurips.cc/paper/2021/hash/6aed000af86a084f9cb0264161e29dd3-Abstract.html)                        | NIPS 2021    |
| FLANC            | [link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1b61ad02f2da8450e08bb015638a9007-Abstract-Conference.html) | NIPS 2022    |
| Hermes           | [link](https://dl.acm.org/doi/10.1145/3447993.3483278)                                                                       | MobiCom 2021 |
| FedMask          | [link](https://dl.acm.org/doi/10.1145/3485730.3485929)                                                                       | SenSys 2021  |
| LotteryFL        | [link](http://arxiv.org/abs/2008.03371)                                                                                      | arxiv        |
| HeteroFL         | [link](http://arxiv.org/abs/2010.01264)                                                                                      | ICLR 2021    |
| TailorFL         | [link](https://dl.acm.org/doi/10.1145/3560905.3568503)                                                                       | SenSys 2022  |
| pFedGate         | [link](http://arxiv.org/abs/2305.02776)                                                                                      | ICML 2023    |

## Combine All The Things Together! :bricks:
<img src="https://github.com/WwZzz/myfigs/blob/master/readme_flgo_com.png?raw=true" width=500/>

FLGo supports flexible combinations of benchmarks, partitioners, algorithms and simulators , which are independent to each other and thus can be used like plugins. We have provided these plugins [here](https://github.com/WwZzz/easyFL/tree/FLGo/resources) , where each can be immediately downloaded and used by API

```python
import flgo
import flgo.benchmark.partition as fbp

fedavg = flgo.download_resource(root='.', name='fedavg', type='algorithm')
mnist = flgo.download_resource(root='.', name='mnist_classification', type='benchmark')
task = 'test_down_mnist'
flgo.gen_task_by_(mnist,fbp.IIDPartitioner(num_clients=10,), task_path=task)
flgo.init(task, fedavg, {'gpu':0}).run()
```

[//]: # (## Multiple Communication Topology Support)
## Easy-to-use Experimental Tools :toolbox:
### Load Results
Each runned result will be automatically saved in `task_path/record/`. We provide an API to easily load and filter records.
```python
import flgo
import flgo.experiment.analyzer as fea
import matplotlib.pyplot as plt
res = fea.Selector({'task': './my_task', 'header':['fedavg',], },)
log_data = res.records['./my_task'][0].data
val_loss = log_data['val_loss']
plt.plot(list(range(len(val_loss))), val_loss)
plt.show()
```

### Use Checkpoint
```python
import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer

task = './my_task'
ckpt = '1'
runner = flgo.init(task, fedavg, {'gpu':[0,],'log_file':True, 'num_epochs':1, 'save_checkpoint':ckpt, 'load_checkpoint':ckpt})
runner.run()
```
We save each checkpoint at `task_path/checkpoint/checkpoint_name/`. By specifying the name of checkpoints, the training can be automatically recovered from them.
```python
import flgo.algorithm.fedavg as fedavg
# the two methods need to be extended when using other algorithms
class Server(fedavg.Server): 
    def save_checkpoint(self):
        cpt = {
            'round': self.current_round,                           # current communication round
            'learning_rate': self.learning_rate,                   # learning rate
            'model_state_dict': self.model.state_dict(),           # model
            'early_stop_option': {                                 # early stop option
                '_es_best_score': self.gv.logger._es_best_score,
                '_es_best_round': self.gv.logger._es_best_round,
                '_es_patience': self.gv.logger._es_patience,
            },
            'output': self.gv.logger.output,                       # recorded information by Logger
            'time': self.gv.clock.current_time,                    # virtual time
        }
        return cpt

    def load_checkpoint(self, cpt):
        md = cpt.get('model_state_dict', None)
        round = cpt.get('round', None)
        output = cpt.get('output', None)
        early_stop_option = cpt.get('early_stop_option', None)
        time = cpt.get('time', None)
        learning_rate = cpt.get('learning_rate', None)
        if md is not None: self.model.load_state_dict(md)
        if round is not None: self.current_round = round + 1
        if output is not None: self.gv.logger.output = output
        if time is not None: self.gv.clock.set_time(time)
        if learning_rate is not None: self.learning_rate = learning_rate
        if early_stop_option is not None:
            self.gv.logger._es_best_score = early_stop_option['_es_best_score']
            self.gv.logger._es_best_round = early_stop_option['_es_best_round']
            self.gv.logger._es_patience = early_stop_option['_es_patience']
```
**Note**: different FL algorithms need to save different types of checkpoints. Here we only provide checkpoint save&load mechanism of FedAvg. We remain two APIs for customization above:

### Use Logger
We show how to use customized Logger [Here](https://flgo-xmu.github.io/Tutorials/1_Configuration/1.6_Logger_Configuration/)
## Tutorials and Documents :page_with_curl:
We have provided comprehensive [Tutorials](https://flgo-xmu.github.io/Tutorials/) and [Document](https://flgo-xmu.github.io/Docs/FLGo/) for FLGo. 
## Deployment To Real Machines :computer:
Our FLGo is able to be extended to real-world application. We provide a simple [Example](https://github.com/WwZzz/easyFL/tree/FLGo/example/realworld_case) to show how to run FLGo on multiple machines. 




# Overview :notebook:

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

* `num_threads` is the number of threads in the clients computing session that aims to accelerate the training process.

* `num_workers` is the number of workers of the torch.utils.data.Dataloader

Additional hyper-parameters for particular federated algorithms:

* `algo_para` is used to receive the algorithm-dependent hyper-parameters from command lines. Usage: 1) The hyper-parameter will be set as the default value defined in Server.__init__() if not specifying this term, 2) For algorithms with one or more parameters, use `--algo_para v1 v2 ...` to specify the values for the parameters. The input order depends on the dict `Server.algo_para` defined in `Server.__init__()`.

Logger's setting

* `logger` is used to selected the logger that has the same name with this term.

* `log_level` shares the same meaning with the LEVEL in the python's native module logging.

* `log_file` controls whether to store the running-time information into `.log` in `fedtask/taskname/log/`, default value is false.

* `no_log_console` controls whether to show the running time information on the console, and default value is false.

### More

To get more information and full-understanding of FLGo please refer to <a href='https://flgo-xmu.github.io/'>our website</a>.

In the website, we offer :

- API docs: Detailed introduction of packages, classes and methods.
- Tutorial: Materials that help user to master FLGo.

## Architecture

We seperate the FL system into five parts:`algorithm`, `benchmark`, `experiment`,  `simulator` and `utils`.
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
|  └─ runner.py                    //the class for generating experimental commands based on hyperparameter combinations and processor scheduling for all experimental 
├─ system_simulator                     //system heterogeneity simulation module
│  ├─ base.py							//the base class for simulate system heterogeneity
│  ├─ default_simulator.py				//the default class for simulate system heterogeneity
|  └─ ...
├─ utils
│  ├─ fflow.py							//option to read, initialize,...
│  └─ fmodule.py						//model-level operators
└─ requirements.txt 
```

### Benchmark

We have added many benchmarks covering several different areas such as CV, NLP, etc

### Algorithm
![image](https://github.com/WwZzz/myfigs/blob/master/fig0.png)
This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 


#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate_with()` them, who first `unpack()` the received package and then train the model with their local dataset by `train()`. After training the model, the clients `pack()` send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     


### Experiment

The experiment module contains experiment command generation and scheduling operation, which can help FL researchers more conveniently conduct experiments in the field of federated learning.

### simulator

The system_simulator module is used to realize the simulation of heterogeneous systems, and we set multiple states such as network speed and availability to better simulate the system heterogeneity of federated learning parties.

### Utils

Utils is composed of commonly used operations: 
1) model-level operation (we convert model layers and parameters to dictionary type and apply it in the whole FL system). 
2) API for the FL workflow like gen_benchmark, gen_task, init, ...
## Citation

Please cite our paper in your publications if this code helps your research.

```
@misc{wang2021federated,
      title={Federated Learning with Fair Averaging}, 
      author={Zheng Wang and Xiaoliang Fan and Jianzhong Qi and Chenglu Wen and Cheng Wang and Rongshan Yu},
      year={2021},
      eprint={2104.14937},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{wang2023flgo,
      title={FLGo: A Fully Customizable Federated Learning Platform}, 
      author={Zheng Wang and Xiaoliang Fan and Zhaopeng Peng and Xueheng Li and Ziqi Yang and Mingkuan Feng and Zhicheng Yang and Xiao Liu and Cheng Wang},
      year={2023},
      eprint={2306.12079},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contacts
Zheng Wang, zwang@stu.xmu.edu.cn


# Buy Me a Coffee :coffee:
Buy me a coffee if you'd like to support the development of this repo.
<center>
<img src="https://github.com/WwZzz/easyFL/assets/20792079/89050169-3927-4eb0-ac32-68d8bee12326" width=180/>
</center>

## References
<div id='refer-anchor-1'></div>

\[McMahan. et al., 2017\] [Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2017.](https://arxiv.org/abs/1602.05629)

<div id='refer-anchor-2'></div>

\[Cong Xie. et al., 2019\] [Cong Xie, Sanmi Koyejo, Indranil Gupta. Asynchronous Federated Optimization. ](https://arxiv.org/abs/1903.03934)

<div id='refer-anchor-3'></div>

\[John Nguyen. et al., 2022\] [John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Michael Rabbat, Mani Malek, Dzmitry Huba. Federated Learning with Buffered Asynchronous Aggregation. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2022.](https://arxiv.org/abs/2106.06639)

<div id='refer-anchor-4'></div>

\[Zheng Chai. et al., 2020\] [Zheng Chai, Ahsan Ali, Syed Zawad, Stacey Truex, Ali Anwar, Nathalie Baracaldo, Yi Zhou, Heiko Ludwig, Feng Yan, Yue Cheng. TiFL: A Tier-based Federated Learning System.In International Symposium on High-Performance Parallel and Distributed Computing(HPDC), 2020](https://arxiv.org/abs/2106.06639)

<div id='refer-anchor-5'></div>

\[Mehryar Mohri. et al., 2019\] [Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh. Agnostic Federated Learning.In International Conference on Machine Learning(ICML), 2019](https://arxiv.org/abs/1902.00146)

<div id='refer-anchor-6'></div>

\[Zheng Wang. et al., 2021\] [Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang, Rongshan Yu. Federated Learning with Fair Averaging. In International Joint Conference on Artificial Intelligence, 2021](https://arxiv.org/abs/2104.14937#)

<div id='refer-anchor-7'></div>

\[Zeou Hu. et al., 2022\] [Zeou Hu, Kiarash Shaloudegi, Guojun Zhang, Yaoliang Yu. Federated Learning Meets Multi-objective Optimization. In IEEE Transactions on Network Science and Engineering, 2022](https://arxiv.org/abs/2006.11489)

<div id='refer-anchor-8'></div>

\[Tian Li. et al., 2020\] [Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith. Federated Optimization in Heterogeneous Networks. In Conference on Machine Learning and Systems, 2020](https://arxiv.org/abs/1812.06127)

<div id='refer-anchor-9'></div>

\[Xinran Gu. et al., 2021\] [Xinran Gu, Kaixuan Huang, Jingzhao Zhang, Longbo Huang. Fast Federated Learning in the Presence of Arbitrary Device Unavailability. In Neural Information Processing Systems(NeurIPS), 2021](https://arxiv.org/abs/2106.04159)

<div id='refer-anchor-10'></div>

\[Yae Jee Cho. et al., 2020\] [Yae Jee Cho, Jianyu Wang, Gauri Joshi. Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies. ](https://arxiv.org/abs/2010.01243)

<div id='refer-anchor-11'></div>

\[Tian Li. et al., 2020\] [Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith. Fair Resource Allocation in Federated Learning. In International Conference on Learning Representations, 2020](https://arxiv.org/abs/1905.10497)

<div id='refer-anchor-12'></div>

\[Sai Praneeth Karimireddy. et al., 2020\] [Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh. SCAFFOLD: Stochastic Controlled Averaging for Federated Learning. In International Conference on Machine Learning, 2020](https://arxiv.org/abs/1910.06378)

