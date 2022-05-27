from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule
import torch

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['alpha']
        self.alpha = option['alpha']
        self.h  = self.model.zeros_like()

    def aggregate(self, models, p=[]):
        self.h = self.h - self.alpha * (1.0 / self.num_clients * fmodule._model_sum(models) - self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.gradL = None
        self.alpha = option['alpha']

    def train(self, model):
        if self.gradL == None:self.gradL = model.zeros_like()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            l1 = self.calculator.train_one_step(model, batch_data)['loss']
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
        return

