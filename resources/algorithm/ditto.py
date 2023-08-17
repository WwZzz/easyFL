"""
This is a non-official implementation of 'Ditto: Fair and Robust Federated Learning Through Personalization' (https://arxiv.org/abs/2012.04221)
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
from flgo.utils import fmodule

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':1.0})
        self.sample_option = 'md'
        self.aggregation_option = 'uniform'

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.model = None

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        if self.model is None: self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
        self.model.train()
        # global solver
        optimizer_global = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        optimizer_local = self.calculator.get_optimizer(self.model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            # global solver
            model.zero_grad()
            loss_global = self.calculator.compute_loss(model, batch_data)['loss']
            loss_global.backward()
            optimizer_global.step()
            # local solver
            self.model.zero_grad()
            loss_local = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss_proximal = 0
            for pm, ps in zip(self.model.parameters(), src_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            loss_local = loss_local + 0.5 * self.mu * loss_proximal
            loss_local.backward()
            optimizer_local.step()
        self.model = self.model.to('cpu')
        return