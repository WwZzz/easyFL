from utils import fmodule
from .fedbase import BaseServer, BaseClient
import numpy as np
import copy

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        # algorithm hyper-parameters
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['learning_rate']
        self.learning_rate_lambda = option['learning_rate_lambda']
        self.result_modeldict = copy.deepcopy(self.model.state_dict())
        self.paras_name=['learning_rate_lambda']

    def iterate(self, t):
        ws, losses, grads = [], [], []
        # training
        for cid in range(self.num_clients):
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
            grads.append(fmodule.modeldict_scale(fmodule.modeldict_sub(self.model.state_dict(), w), 1.0 / self.learning_rate))

        # aggregate grads
        grad = self.aggregate(grads, self.dynamic_lambdas)
        w_new = fmodule.modeldict_sub(self.model.state_dict(), fmodule.modeldict_scale(grad, self.learning_rate))
        self.model.load_state_dict(w_new)
        # update lambdas
        for lid in range(len(self.dynamic_lambdas)):
            self.dynamic_lambdas[lid] += self.learning_rate_lambda * losses[lid]
        self.dynamic_lambdas = self.project(self.dynamic_lambdas)
        # record resulting model
        self.result_modeldict = fmodule.modeldict_add(fmodule.modeldict_scale(self.result_modeldict, t), w_new)
        self.result_modeldict = fmodule.modeldict_scale(self.result_modeldict, 1.0 / (t + 1))
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def aggregate(self, ws, p):
        return fmodule.modeldict_weighted_average(ws, p)

    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0/(i + 1)) * (1 - np.sum(np.asarray(u)[:i+1]))) > 0:
                rho = i + 1
        lmbd = (1.0/rho) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i]+lmbd, 0))
        return res

    def test_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            self.trans_model.load_state_dict(self.result_modeldict)
            c.setModel(self.trans_model)
            acc, loss = c.test()
            accs.append(acc)
            losses.append(loss)
        return accs, losses

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, partition = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, partition, drop_rate)