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

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # local training
        models, train_losses = self.communicate(selected_clients)
        # aggregate
        self.model = self.aggregate(models)
        # output info
        return selected_clients

    def aggregate(self, models):
        self.h = self.h - self.alpha/self.num_clients*(fmodule._model_sum(models) - self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, drop_rate=-1):
        super(Client, self).__init__(option, name, train_data, valid_data, drop_rate)
        self.gradL = None
        self.alpha = option['alpha']

    def train(self, model):
        if self.gradL == None:
            self.gradL = model.zeros_like()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                l1 = self.calculator.get_loss(model, batch_data)
                l2 = 0
                l3 = 0
                for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                    l2 += torch.dot(pgl.view(-1), pm.view(-1))
                    l3 += torch.sum(torch.pow(pm-ps,2))
                loss = l1 - l2 + 0.5 * self.alpha * l3
                loss.backward()
                optimizer.step()
        # update grad_L
        self.gradL = self.gradL - self.alpha * (model-src_model)
        return

