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

Create classes of `Server` and `Client` that adaptlively inherit from `BasicServer` and `BasicClient` in `fedbase`. 
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
    optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
    for iter in range(self.epochs):
        for batch_idx, batch_data in enumerate(data_loader):
            model.zero_grad()
            loss = self.calculator.get_loss(model, batch_data)
            # 3
            """loss_prox = 0"""
            # 4
            """
            for pm, ps in zip(model.parameters(), src_model.parameters()): loss_prox+= torch.sum(torch.pow(pm-ps,2))
            """
            # 5
            """loss += 0.5 * self.mu * loss_prox""" 
       
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
