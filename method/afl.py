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
        self.result_model = copy.deepcopy(self.model)
        self.paras_name=['learning_rate_lambda']

    def iterate(self, t):
        # full sampling
        # training
        ws, losses = self.communicate([cid for cid in range(self.num_clients)])
        grads = [(self.model - w)/self.learning_rate for w in ws]
        # aggregate grads
        grad = fmodule.average(grads, self.dynamic_lambdas)
        self.model = self.model - self.learning_rate*grad
        # update lambdas
        self.dynamic_lambdas = [lmb_i+self.learning_rate_lambda*loss_i for lmb_i,loss_i in zip(self.dynamic_lambdas, losses)]
        self.dynamic_lambdas = self.project(self.dynamic_lambdas)
        # record resulting model
        self.result_model = (t*self.result_model + self.model)/(t+1)
        return [c for c in range(self.num_clients)]

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

    def test_on_clients(self, round, dataflag='valid'):
        accs, losses = [], []
        for c in self.clients:
            acc, loss = c.test(self.result_model, dataflag)
            accs.append(acc)
            losses.append(loss)
        return accs, losses

    def test_on_dtest(self):
        if self.dtest:
            return fmodule.test(self.result_model, self.dtest)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, train_rate = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)