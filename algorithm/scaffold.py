from .fedbase import BasicServer, BasicClient
import copy
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.eta = option['eta']
        self.cg = self.model.zeros_like()
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
        dw = fmodule._model_average(dys)
        dc = fmodule._model_average(dcs)
        new_model = self.model + self.eta * dw
        new_c = self.cg + 1.0 * len(dcs) / self.num_clients * dc
        return new_model, new_c


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.c = None
        
    def train(self, model, cg):
        model.train()
        if not self.c:
            self.c = model.zeros_like()
            self.c.freeze_grad()
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        cg.freeze_grad()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train(model, batch_data)
            loss.backward()
            for pm, pcg, pc in zip(model.parameters(), cg.parameters(), self.c.parameters()):
                pm.grad = pm.grad - pc + pcg
            optimizer.step()
        # update local control variate c
        K = self.num_steps
        dy = model - src_model
        dc = -1.0 / (K * self.learning_rate) * dy - cg
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
