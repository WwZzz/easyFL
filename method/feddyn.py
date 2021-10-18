from .fedbase import BaseServer, BaseClient
from torch.utils.data import DataLoader
from utils.fmodule import device, lossfunc, Optim
import copy
from utils import fmodule
import torch

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest=None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.paras_name = ['alpha']
        self.alpha = option['alpha']
        self.h  = self.model.zeros_like()

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # local training
        ws, losses = self.communicate(selected_clients)
        # aggregate
        self.model = self.aggregate(ws)
        # output info
        return selected_clients

    def aggregate(self, ws):
        self.h = self.h - self.alpha/self.num_clients*(fmodule._model_sum(ws) - self.model)
        w_new = fmodule._model_average(ws) - 1.0 / self.alpha * self.h
        return w_new


class Client(BaseClient):
    def __init__(self, option, name='', data_train_dict={'x': [], 'y': []}, data_val_dict={'x': [], 'y': []}, train_rate=0.8, drop_rate=0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)
        self.w = None
        self.gradL = None
        self.alpha = option['alpha']

    def train(self, model):
        if self.gradL == None:
            self.gradL = model.zeros_like()
        # global parameters
        src_model = copy.deepcopy(model)
        model.train()
        if self.batch_size == -1:
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = fmodule.get_optimizer(self.optimizer, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(images)
                l1 = lossfunc(outputs, labels)
                l2 = 0
                l3 = 0
                for pgl, pm, ps in zip(self.gradL.parameters(), model.parameters(), src_model.parameters()):
                    l2 += torch.dot(pgl.view(-1), pm.view(-1))
                    l3 += torch.sum(torch.pow(pm-ps,2))
                loss = l1 - l2 + 0.5 * self.alpha * l3
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # update grad_L
        self.gradL = self.gradL - self.alpha * (model-src_model)
        return sum(epoch_loss) / len(epoch_loss)

