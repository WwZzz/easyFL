# How To Convert A Traditional ML Task Into A Federated One?
When someone wants to convert a traditional ML task into a federated one, the issues below will immediately arise:
* How to partition the dataset into different subdataset? How can a dataset be partitioned in the I.I.D. or non-I.I.D. manner?
* Should the existing federated optimization algorithms (e.g. fedavg) be rewrite to suit the new coming task?
* What if the metrics vary across different ML tasks? Will existing codes still work?
* Can a partitioned setting be reused by different algorithms?
* ...

Fortunately, we've developed this module to simultaneously address these problems. The main purpose of ours is to 1) provide a data pre-processing paradigm to distribute a ML dataset to different virtual clients, 2) decouple the task-specific part from federated optimization process. Now we in turn introduce how we achieve the two targets and finally take an example to help understand the whole procedure.

<p float="left">
   <img src="https://github.com/WwZzz/myfigs/blob/master/easyfl_benchmark_od.jpg" width="1000" />
</p>

## Generating FL Task From Traditional ML Task By `TaskGen` and `TaskPipe`
```
The architechture of benchmark
benchmark
├─ mnist_classification			//classification on mnist dataset
│  ├─ model                   //the models used in the original ML task
│  │  ├─cnn.py
│  │  ├─mlp.py
|  └─ core.py   
│  │  ├─TaskGen               //download dataset, load dataset into memory, partition dataset
│  │  ├─TaskPipe              //save dataset processed by TaskGen into the disk as fedtask, load the fedtask into training system
│  │  └─TaskCalculator        //a black box that receives (model, data) and then returns the necessary training\testing-time variables to federated algorithms
├─ ...
├─ RAW_DATA                   // storing the downloaded raw dataset
└─ toolkits.py						//the basic tools for generating federated dataset
```

To federalize a traditional ML task, we consider steps including loading the original ML dataset, setting the model (the backbone or the network), partitioning the dataset, saving as a stastic file, dynamically loading the saved file into federated training system. Now we illustrate each step in detail.
### TaskGen

**Loading Dataset.** The loading procedure must be complemented in TaskGen.load_data(). Since one dataset can be used to training different models for different purpose (segmentation, classification,...), download the original dataset to `./benchmark/RAW_DATA/dataset_name` and read it. The downloading manner depends on the selected dataset. For example, if the dataset has already been implemented in `torchvision` or `torchaudio` (e.g. mnist), then `torchvision.datasets.MNIST` will automatically download the data and load it into memory. For dataset like shakespeare, we directly download it from the website by url. After this step, the TaskGen should have attribute `train_data` and `test_data` so that it can be handled by `TaskGen.partition()`

**Setting Model.** The model should be defined in `./benchmark/benchmark_name/model/` with the name of the model. Each `model_name.py` (e.g. cnn.py) should contains a class named `Model()` inherenting from `utils.fmodule.FModule` where we pre-define a few operators like directly plusing one model's parameters to another's. Details can be found in `utils.fmodule`.   

**Partitioning Dataset.** After loading the dataset into the memory, we should partition the dataset into several parts so that it can be allocated to different virtual clients. There are mainly three kinds of datasets: testing dataset owned by Server, local training and validation dataset owned by clients. We pre-define  `TaskGen.partition()` to partition the training dataset and `TaskGen.local_holdout()` to hold out the validation dataset from each After partitioning, the partition information should be recorded by TaskGen itself and provide adequate information to reconstruct the federated dataset. Then the instance of `TaskGen` itself will be used to save the dataset by TaskPipe.

### TaskPipe

The `TaskPipe` is named as its function since it's like a pipe of TaskGen-Fedtask-FederatedTrainingSystem, where the generator uses it to save partitioned dataset into the disk as `fedtask` and the federated system uses it to read the `fedtask` from the disk. Each TaskPipe contains three parts: `save_task: @classmethod`, `load_task: @classmethod`, 'TaskDataset: class variable'.

**Loading Dataset.**


## Decoupling Task-Specific Calculation From Federated System by `TaskCalculator`
coming soon...
