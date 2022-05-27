"""
This is a non-official implementation of Scaffold proposed in 'Stochastic
Controlled Averaging for Federated Learning' (ICML 2020).
"""

from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.cg = self.model.zeros_like()
        self.eta = option['eta']
        self.paras_name = ['eta']

    def pack(self, client_id):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def iterate(self, t):
        # sample clients
        self.selected_clients = self.sample()
        # local training
        res = self.communicate(self.selected_clients)
        dys, dcs = res['dy'], res['dc']
        # aggregate
        self.model, self.cg = self.aggregate(dys, dcs)
        return

    def aggregate(self, dys, dcs):
        # x <-- x + eta_g * dx = x + eta_g * average(dys)
        # c <-- c + |S|/N * dc = c + |S|/N * average(dcs)
        dx = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dx
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = fmodule.Model().zeros_like()
        self.c.freeze_grad()
        
    def train(self, model, cg):
        """
        The codes of Algorithm 1 that updates the control variate
          12:  ci+ <-- ci - c + 1 / K / eta_l * (x - yi)
          13:  communicate (dy, dc) <-- (yi - x, ci+ - ci)
          14:  ci <-- ci+
        Our implementation for efficiency
          dy = yi - x
          dc <-- ci+ - ci = -1/K/eta_l * (yi - x) - c = -1 / K /eta_l *dy - c
          ci <-- ci+ = ci + dc
          communicate (dy, dc)
        """
        model.train()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            loss.backward()
            # y_i <-- y_i - eta_l ( g_i(y_i)-c_i+c )  =>  g_i(y_i)' <-- g_i(y_i)-c_i+c
            for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                pm.grad = pm.grad - pc + pcg
            optimizer.step()
        dy = model - src_model
        dc = -1.0 / (self.num_steps * self.learning_rate) * dy - cg
        self.c = self.c + dc
        return dy, dc

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
