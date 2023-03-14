from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
from flgo.utils import fmodule
import torch

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'eta': 1.0})
        self.cg = self.model.zeros_like()
        self.sample_option = 'uniform'

    def pack(self, client_id, *args, **kwargs):
        return {
            "model": copy.deepcopy(self.model),
            "cg": self.cg,
        }

    def iterate(self):
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
        # c <-- c + |S|/N * dc = c + 1/N * sum(dcs)
        new_model = self.model + self.eta * fmodule._model_average(dys)
        new_c = self.cg + fmodule._model_sum(dcs)/self.num_clients
        return new_model, new_c


class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.c = None

    @fmodule.with_multi_gpus
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
        # if self.c is None: self.c = copy.deepcopy(cg)
        if self.c is None: self.c = cg.zeros_like()
        self.c.freeze_grad()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            # y_i <-- y_i - eta_l ( g_i(y_i)-c_i+c )  =>  g_i(y_i)' <-- g_i(y_i)-c_i+c
            for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                pm.grad = pm.grad - pc + pcg
            optimizer.step()
        dy = model - src_model
        dc = -dy/(self.num_steps * self.learning_rate) - cg
        self.c = self.c + dc
        return dy, dc

    def pack(self, dy, dc):
        return {
            "dy": dy,
            "dc": dc,
        }

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model'], received_pkg['cg']

    def reply(self, svr_pkg):
        model, c_g = self.unpack(svr_pkg)
        dy, dc = self.train(model, c_g)
        cpkg = self.pack(dy, dc)
        return cpkg