"""
This is a non-official implementation of 'Federated Learning Based on Dynamic Regularization'
(http://arxiv.org/abs/2111.04263). The official implementation is at 'https://github.com/alpemreacar/FedDyn'
"""
from .fedbase import BasicServer, BasicClient
import copy
from flgo.utils import fmodule
import torch

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'alpha': 0.1})
        self.h = self.model.zeros_like()

    def aggregate(self, models):
        self.h = self.h - self.alpha * (1.0 / self.num_clients * fmodule._model_sum(models) - self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.gradL = None

    @ fmodule.with_multi_gpus
    def train(self, model):
        if self.gradL == None:self.gradL = model.zeros_like()
        self.gradL.to(model.get_device())
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            l1 = self.calculator.compute_loss(model, batch_data)['loss']
            l2 = 0
            l3 = 0
            for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                l2 += torch.dot(pgl.view(-1), pm.view(-1))
                l3 += torch.sum(torch.pow(pm - ps, 2))
            loss = l1 - l2 + 0.5 * self.alpha * l3
            loss.backward()
            optimizer.step()
        # update grad_L
        self.gradL = self.gradL - self.alpha * (model-src_model)
        self.gradL.to(torch.device('cpu'))
        return