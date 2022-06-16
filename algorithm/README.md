<div id='refer-anchor-1'></div>

# How to realize federated algorithms by easyFL?
Either you want to quickly implement your own federated algorithms, 
or you want to reproducing the experimental results of other papers, 
please carefully read this part as is almost created for this purpose. 
To show how easily one can use this module to reproduce a popular federated algorithm, 
we take FedProx and Scaffold as the examples and detail their implemention.
## Example 1: FedProx
Now we show how we realize FedProx with 5 lines.

Compared to FedAvg, FedProx is different in two ways. 
The first one is the client selection and corresponding 
model aggregation, where clients are sampled with probability 
p_k (the ratio of their data size) and the models are uniformly averaged.
The second one is that FedProx adds a proximal term when locally training the model.
Since easyFL already takes the MDsample with uniform aggregation as the default setting, 
the only issue you need is to change the function of local training. 
Thus, we realize FedProx as steps below:

**First, copy the codes we need from fedbase.**

Create classes of `Server` and `Client` respectively inheriting from `BasicServer` and `BasicClient` in `fedbase`. 
And copy the standard local training function `train()` from `BasicClient` to `fedprox.Client`. 

**Second, handling hyper-parameters of FedProx.**

add hyper-parameter `mu` as an attribute of `fedprox.Client`, and append the name `mu` to `fedprox.Server.paras_name`.

**Finally, add the proximal term to the loss.**

calculate and add the proximal loss to the original loss before backward propagation.

```python
def train(self, model):
 """the '"' hightlight additional lines of train() in fedprox compared to fedavg"""
 # 1
 """src_model = copy.deepcopy(model)"""
 # 2 (only for efficiency and can be removed)   
 """src_model.freeze_grad()"""
 model.train()
 data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
 optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate,
                                           weight_decay=self.weight_decay, momentum=self.momentum)
 for iter in range(self.num_steps):
  batch_data = self.get_batch_data()
  model.zero_grad()
  loss = self.calculator.train_one_step(model, batch_data)
  # 3
  """loss_proximal = 0"""
  # 4
  """
  for pm, ps in zip(model.parameters(), src_model.parameters()): loss_proximal+= torch.sum(torch.pow(pm-ps,2))
  """
  # 5
  """loss += 0.5 * self.mu * loss_proximal"""
  loss.backward()
  optimizer.step()
 return
```
 
 That is to say, you only need 5 lines 
 (the line #2 is for efficiency by avoiding backward propagation of the parameter of the global model.)
 to realize FedProx.
 
 Now let's take a look on the results of our implemention of FedProx.

<p float="left">
  <img src="https://github.com/WwZzz/myfigs/blob/master/fig01_trainloss_for_synthetic_0505_fedprox.png" width="500" />
  <img src="https://github.com/WwZzz/myfigs/blob/master/fig02_testacc_for_synthetic_0505_fedprox.png" width="500" /> 
</p>
 
To get the results, run the commands below:

```
# generate federated synthetic(0.5, 0.5) data
python generated_fedtask.py --dataset synthetic --dist 10 --skew 0.5 --num_clients 30

# run fedavg (default), weighted sample and uniform average
python main.py --task synthetic_cnum30_dist10_skew0.5_seed0 --num_epochs 20 --algorithm fedavg --model lr --learning_rate 0.01 --batch_size 10 --num_rounds 200 --proportion 0.34 --gpu 0 --lr_scheduler 0

# run fedprox with mu = 0.1
python main.py --task synthetic_cnum30_dist10_skew0.5_seed0 --num_epochs 20 --algorithm fedprox --mu 0.1 --model lr --learning_rate 0.01 --batch_size 10 --num_rounds 200 --proportion 0.34 --gpu 0 --lr_scheduler 0

# run fedprox with mu = 0.5
python main.py --task synthetic_cnum30_dist10_skew0.5_seed0 --num_epochs 20 --algorithm fedprox --mu 0.5 --model lr --learning_rate 0.01 --batch_size 10 --num_rounds 200 --proportion 0.34 --gpu 0 --lr_scheduler 0

# run fedprox with mu = 1
python main.py --task synthetic_cnum30_dist10_skew0.5_seed0 --num_epochs 20 --algorithm fedprox --mu 1 --model lr --learning_rate 0.01 --batch_size 10 --num_rounds 200 --proportion 0.34 --gpu 0 --lr_scheduler 0

# run fedavg (origin), uniform sample and weighted average
python main.py --task synthetic_cnum30_dist10_skew0.5_seed0 --num_epochs 20 --algorithm fedavg --aggregate weighted_com --sample uniform --model lr --learning_rate 0.01 --batch_size 10 --num_rounds 200 --proportion 0.34 --gpu 0 --lr_scheduler 0
```
 ## Example 2 : Scaffold
Scaffold is different from Fedavg during all the three stages in FL: communication stage (e.g. uploading\downloading parameters), aggregation, local trianing. Thus, we discuss these three issues respectively. 

**First, initialization of the server and the clients.**
Scaffold additionally uses clients' control variates `c_i` and the global one `cg` to help local training not leave away from the global updating direction. Therefore, the two variables should be initialized in the `__init__()` of both `Client` and `Server` (remark: here we setting the initial values to be zero vectors as mentioned in the original paper). The hyperparameter `eta_g` should also be initialized in the `Server` as the way of initializing `mu` of FedProx.
```python
class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.cg = self.model.zeros_like()
        self.cg.freeze_grad()
        self.eta = option['eta']
        self.paras_name = ['eta']
        
class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = fmodule.Model().zeros_like()
        self.c.freeze_grad()
```
**Second, change the communication procedure.**
In each communication round of Fedavg, the server sends the global model to the selected clients and the selected clients upload the locally updated models to the clients. Compared to Fedavg, Scaffold additionally exchange the global and local controlling variates in each communication. Thus, there are two communicating procedures to be modified. The first one is the communciation from the server to the clients, which consists of a pair of functions: `Server.pack()` and `Client.unpack()`. The server package is realized as `dict` of Python, and the additional information is added by putting new key and value to the package. The reverse direction from the client to the server is realized similarly, where `Client.pack()` and `Server.unpack()` is needed to be modified. Since `BasicServer.unpack()` has been realized as default for converting the packages from different clients to the lists that contains each value specified by the keys of the packages, it's not necessary to rewrite this function.

```python
    def Server.pack(self, client_id):
        return {"model": copy.deepcopy(self.model), "cg": self.cg}
    
    def BasicServer.unpack(self, received_packages):
        ...
        
    def Client.pack(self, dy, dc):
            return {"dy": dy, "dc": dc}
            
    def Client.unpack(self, received_pkg):
        return received_pkg['model'], received_pkg['cg']
```
**Third, change the local training procedure.**
Let's focus the local training procedure of Scaffold in Algorithm 1 of the original paper. In each step of updating the model, the delta is corrected by adding (c-c_i) for client i as the 10th line in Algo.1. Thus, we add the term to the gradient `for pm in model.parameters(): pm.grad...` to achieve the correction. Then, the local control variate is also updated as:
```
          12th:  ci+ <-- ci - c + 1 / K / eta_l * (x - yi)
          13th:  communicate (dy, dc) <-- (yi - x, ci+ - ci)
          14th:  ci <-- ci+
```
We implement this by instead calculating the terms below to obtain the same results:
```
          dy = yi - x
          dc <-- ci+ - ci = -1/K/eta_l * (yi - x) - c = -1 / K /eta_l *dy - c
          ci <-- ci+ = ci + dc
          communicate (dy, dc)
```
The whole codes for this part is shown as below (`train()` is copied from `BasicClient.train()` and modified by adding/changing total 7 lines):

```python
    def train(self, model, cg):
 model.train()
 # 1
 src_model = copy.deepcopy(model)
 # 2
 src_model.freeze_grad()
 optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate,
                                           weight_decay=self.weight_decay, momentum=self.momentum)
 for iter in range(self.num_steps):
  batch_data = self.get_batch_data()
  model.zero_grad()
  loss = self.calculator.train_one_step(model, batch_data)
  loss.backward()
  # 3
  for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()): pm.grad = pm.grad - pc + pcg
  optimizer.step()
 # 4    
 dy = model - src_model
 # 5
 dc = -1.0 / (self.num_steps * self.learning_rate) * dy - cg
 # 6
 self.c = self.c + dc
 # 7
 return dy, dc
```

**Fourth, change the aggregation way of the server.**
The model is directly updated by averaging the local models of clients, and the global control variate is updated similarly according to th 16th line in Algo.1.
```python
    def aggregate(self, dys, dcs):
        dx = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dx
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c
```
**Finally, modify the main function of the server and the clients.**
In each communication round, the action of the server and clients are respectively decided by `Server.iterate()` and `Client.reply()`. Since a few changes are take in the three important stages during communication, the two main functions should also be changed for alignment. 
```python
    def Server.iterate(self, t):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        # 1
        dys, dcs = res['dy'], res['dc']
        # 2
        self.model, self.cg = self.aggregate(dys, dcs)
        return
    
    def Client.reply(self, svr_pkg):
        model, c_g = self.unpack(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc)
        return cpkg
```
Now let's take a look on the results of our implemention of Scaffold.

# How to make observations during training time?
<div id='refer-anchor-2'></div>
To make additional observations about the whole training procedure for different optimization algorithms, we encourage the user to create another new algorithm file  and use `MyLogger(utils.fmodule.Logger)` to insert codes to the proper position to show the particular computing results. For example, if someone wants to see how testing loss changes during the local training process of a specified client i (e.g. client 0) when using fedavg to optimize the targets, what he needs to do including:

* create a new file `algorithm/fedavg_localtest_example.py` and copy the necessary parts from `fedbase.py` (e.g. copy Client.train() into the new file since the codes should be inserted into the local training process to show the testing loss during local training time).
* create a new class `class MyLogger(utils.fflow.Logger)` and add the instance method `local_test(server, model): {return the server's testing loss of the model}` to the class. The instance of this class will be generated in `utils.fflow.logger`.
* use `utils.fflow.logger` to make observations on client 0 in the training process as below, and the results will be stored by the logger into the log file in `fedtask/task_name/record/`.

```python
# 0
import utils.fflow as flw

...


class Client(BasicClient):
 def train(self, model):
  model.train()
  optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate,
                                            weight_decay=self.weight_decay, momentum=self.momentum)
  # 1
  rec_test_loss = []
  for iter in range(self.num_steps):
   batch_data = self.get_batch_data()
   # 2
   if self.name == 'Client00':
    # 3
    test_loss = flw.logger.local_test(self.server, model)
    # 4
    rec_test_loss.append(test_loss)
   model.zero_grad()
   loss = self.calculator.train_one_step(model, batch_data)
   loss.backward()
   optimizer.step()
  # 5
  flw.logger.write('client00_local_testing_loss', rec_test_loss)
  return


class MyLogger(flw.Logger):
 def __init__(self):
  super(MyLogger, self).__init__()

 def local_test(self, server, model):
  test_metric = server.test(model)
  return test_metric['loss']
```
In this way, the codes of `fedavg.py` is preserved, since all the addtional operations are made in another independent algorithm file with a different name. 
