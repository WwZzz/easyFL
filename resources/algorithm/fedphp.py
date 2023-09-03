"""
This is a non-official implementation of personalized FL method FedPHP (https://dl.acm.org/doi/abs/10.1007/978-3-030-86486-6_36).
The original implementation is in github repo (https://github.com/lxcnju/FedRepo)
"""
import flgo.algorithm.fedbase
import copy
import torch
import flgo.utils.fmodule as fmodule

class Server(flgo.algorithm.fedbase.BasicServer):
    """
    Hyper-parameters:
        lmbd (float): the coefficient of the regularization term
        mu (float): the coeeficient of moving averaging among localpersonalized models
        reg (str): the type of regularization terms in {'mmd', 'l2', 'prox', 'kd'}
    """
    def initialize(self):
        self.init_algo_para({'lmbd': 0.1, 'mu':0.9, 'reg':'mmd'})
        for c in self.clients:
            c.Q = 1.0*self.clients_per_round/self.num_clients

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model)
        self.zk = 0

    def cal_mu(self, t):
        return max(min(self.mu*self.zk/(self.Q*t), 1.0),0.0)

    @fmodule.with_multi_gpus
    def train(self, model):
        self.zk += 1
        model.train()
        self.model.to(self.device)
        self.model.freeze_grad()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.calculator.to_device(self.get_batch_data())
            model.zero_grad()
            hs = model.encoder(batch_data[0])
            logits = model.head(hs)
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss_erm = self.calculator.criterion(logits, batch_data[-1])
            phs = self.model.encoder(batch_data[0])
            plogits = self.model.head(phs)
            phs = phs.detach()
            plogits = plogits.detach()
            if self.reg=='mmd':
                loss_reg = mmd_rbf_noaccelerate(hs, phs)
            elif self.reg=='l2':
                loss_reg = 0.5*torch.sum((phs-hs)**2)/len(batch_data[-1])
            elif self.reg=='prox':
                loss_reg = 0.0
                for p, pp in zip(model.parameters(), self.model.parameters()):
                    loss_reg += torch.sum((p-pp)**2)
            elif self.reg=='kd':
                loss_reg = (-1.0 * (plogits / 4.0).softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()
            loss = (1 - self.lmbd) * loss_erm + self.lmbd * loss_reg
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        # Equation (5) in paper\line 9 in ClientProcedure() in Algorithm 1
        mu_k = self.cal_mu(self.server.current_round)
        # Update Personalized model
        self.model = (1-mu_k)*model + mu_k*self.model
        return

def guassian_kernel(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((
        total.unsqueeze(dim=1) - total.unsqueeze(dim=0)
    ) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    # print("Bandwidth:", bandwidth)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
