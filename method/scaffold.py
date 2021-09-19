import torch

from .fedbase import BaseServer, BaseClient
from torch.utils.data import DataLoader
from utils.fmodule import device, lossfunc, Optim
import copy
from utils import fmodule


class Server(BaseServer):
    def __init__(self, option, model, clients, dtest=None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.eta = option['eta']
        self.cg = self.model.zeros_like()
        self.paras_name = ['eta']

    def pack(self, cid):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def unpack(self, pkgs):
        dys = [p["dy"] for p in pkgs]
        dcs = [p["dc"] for p in pkgs]
        return dys, dcs

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # local training
        dys, dcs = self.communicate(selected_clients)
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        # output info
        return selected_clients

    def aggregate(self, dys, dcs):  # c_list is c_i^+
        dw = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dw
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c


class Client(BaseClient):
    def __init__(self, option, name='', data_train_dict={'x': [], 'y': []}, data_val_dict={'x': [], 'y': []}, train_rate=0.8, drop_rate=0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)
        self.c = None
        
    def train(self, model, cg):
        if not self.c:
            self.c = model.zeros_like()
            self.c.freeze_grad()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        model.train()
        if self.batch_size == -1:
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = Optim(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        epoch_loss = []
        num_batches = 0
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(images)
                loss = lossfunc(outputs, labels)
                loss.backward()
                for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                    pm.grad = pm.grad - pc + pcg
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
                num_batches += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # update local control variate c
        K = self.epochs * num_batches
        dy = model - src_model
        dc = -1.0 / (K * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy,dc

    def reply(self, svr_pkg):
        model, c_g = self.unpack(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc)
        return cpkg

    def pack(self, dy, dc):
        return {
            "dy": dy,
            "dc": dc,
        }

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']
