"""
This is a non-official implementation of personalized FL method FedBABU (http://arxiv.org/abs/2106.06042).
The original implementation is in github repo (https://github.com/jhoon-oh/FedBABU)
"""
import copy
import torch
import flgo.algorithm.fedbase
import torch.nn as nn
import flgo.utils.fmodule as fmodule

class Server(flgo.algorithm.fedbase.BasicServer):
    """
    Hyper-parameters:
        local_upt_part (str): of values ['body', 'head', 'full']
        head_init (str): of values ['he', 'xavier', 'orth']
    """
    def initialize(self):
        self.init_algo_para({'local_upt_part':'body', 'head_init':'he'})
        self.init_model_head()

    def init_model_head(self):
        with torch.no_grad():
            if self.head_init=='he':
                nn.init.kaiming_uniform_(self.model.head.weight)
            elif self.head_init=='xavier':
                nn.init.xavier_uniform_(self.model.head.weight)
            elif self.head_init=='orth':
                nn.init.orthogonal_(self.model.head.weight)

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model)

    def reply(self, svr_pkg):
        self.unpack(svr_pkg)
        self.train(self.model)
        return self.pack(self.model)

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        if self.local_upt_part=='body':
            for n,p in model.named_parameters():
                p.requires_grad = (n.split('.')[0]!='head')
        elif self.local_upt_part=='head':
            for n,p in model.named_parameters():
                p.requires_grad = (n.split('.')[0]=='head')
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return