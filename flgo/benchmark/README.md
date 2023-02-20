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

**Partitioning Dataset.** After loading the dataset into the memory, we should partition the dataset into several parts so that it can be allocated to different virtual clients. There are mainly three kinds of datasets: testing dataset owned by Server, local training and validation dataset owned by clients. We pre-define  `TaskGen.partition()` to partition the training dataset and `TaskGen.local_holdout()` to hold out the validation dataset from each local dataset. After partitioning, the partition information should be recorded by TaskGen itself and provide adequate information to reconstruct the federated dataset. Then the instance of `TaskGen` itself will be used to save the dataset by TaskPipe.

### TaskPipe

The `TaskPipe` is named as its function since it's like a pipe of TaskGen-Fedtask-FederatedTrainingSystem, where the generator uses it to save partitioned dataset into the disk as `fedtask` and the federated system uses it to read the `fedtask` from the disk. Each TaskPipe contains three parts: `save_task: @classmethod`, `load_task: @classmethod`, 'TaskDataset: class variable'.

**save_task.** this method will be only called by TaskGen, which converts the generator that has finished partition to the fedtask as .json file.   

**TaskDataset.** The TaskDataset inherents from torch.utils.data.Dataset and is an encapsulation for the splited sub-dataset of the original dataset. There are mainly two ways to design this class. If the numerical data (e.g. image) is stored in `fedtask` (e.g. `benchmark/toolkits.XYTaskPipe`), then `TaskDataset.__init__` should receive the stored features and labels to reconstruct the subdataset and `TaskDataset.__getitem__` should returns the indexed data at each momnet it is called. If the original indices of data in the original dataset is stored (e.g. `benchmark/toolkits.IDXTaskPipe`), then the `TaskDataset` must record the coresponding way to access the original dataset and returns the indexed data in original dataset. The second manner is comparably more faster than the first one and takes up less space in the memory. However, for synthetic data, only the first manner can be used. 

**load_task.** this method will be only called by `utils.fflow.initialize` when initalizing the federated sysptem, which read the `fedtask` to the client's training data, clients' validation data and server's testing data. The types of the loaded data are all `TaskPipe.TaskDataset`. 

## Decoupling Task-Specific Calculation From Federated System by `TaskCalculator`

It's difficult or even impossible to decouple all the task-specific parts from the federated optimization, since the models, the optimizers, the metrics, training procedures, the shape of data all vary across different ML tasks and federated algorithms. Therefore, we try to conclude a universal template that can be generalized to as more ML tasks as possible. Let's first see the local training procedure of `BasicClient.train()`:

```python
class BasicClient:
    ...

    def train(self, model):
        model.train_one_step()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate,
                                                  weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            loss.backward()
            optimizer.step()
        return

    def get_batch_data(self):
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size,
                                                                    num_workers=self.loader_num_workers))
            batch_data = next(self.data_loader)
        return batch_data
```

Here the `calculator` is the instance of `TaskCalculator` that is dynamically imported at `utils/fflow.initialize` according to the benchmark name of the fedtask. When local training the model, the function `Client.train()` accesses all the running-time variables necessary for local training by the calculator. For example, the batched data is generated from `calculator.get_data_loader()`, and `calculator.train_one_step()` returns a `dict` that contains the loss in the computing graph that can be used to computing the gradient. In this way, a lot of federated algorithms with additional loss term can be applied without changing the calculator (e.g. FedProx, FedDyn). There are also works that utilize the intermediate variables of the forward process when feeding data to the model (e.g. MOON). To handle this case, we allow the model to be defined in the algorithm file `fedxx.py` instead of `benchmark/benchmark_name/model/model_name.py`, which will be imported when the model_name cannot be found in `benchmark/benchmark_name/model/`. Therefore, the special term can be calculated by using the algorithm-specific model. (Remark: for now we've mainly considered about classification problems, and the other problems (e.g. Re-ID, NLP) will be implemented soon.)

## Example of Task-Converting
