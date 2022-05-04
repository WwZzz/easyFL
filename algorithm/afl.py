from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np
import copy
import collections

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        # algorithm hyper-parameters
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['learning_rate']
        self.learning_rate_lambda = option['learning_rate_lambda']
        self.result_model = copy.deepcopy(self.model)
        self.paras_name = ['learning_rate_lambda']

    def iterate(self, t):
        # full sampling
        # training
        res = self.communicate([cid for cid in range(self.num_clients)])
        models, train_losses = res['model'], res['loss']
        grads = [(self.model - model) / self.learning_rate for model in models]
        # aggregate grads
        grad = fmodule._model_average(grads, self.dynamic_lambdas)
        self.model = self.model - self.learning_rate * grad
        # update lambdas
        self.dynamic_lambdas = [lmb_i + self.learning_rate_lambda * loss_i for lmb_i, loss_i in
                                zip(self.dynamic_lambdas, train_losses)]
        self.dynamic_lambdas = self.project(self.dynamic_lambdas)
        # record resulting model
        self.result_model = (t * self.result_model + self.model) / (t + 1)
        return

    def project(self, p):
        u = sorted(p, reverse=True)
        res = []
        rho = 0
        for i in range(len(p)):
            if (u[i] + (1.0 / (i + 1)) * (1 - np.sum(np.asarray(u)[:i + 1]))) > 0:
                rho = i + 1
        lmbd = (1.0 / rho) * (1 - np.sum(np.asarray(u)[:rho]))
        for i in range(len(p)):
            res.append(max(p[i] + lmbd, 0))
        return res

    def test_on_clients(self, round, dataflag='valid'):
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.result_model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None):
        if model == None: model = self.result_model
        if self.test_data:
            return self.calculator.test(model, self.test_data)
        else:
            return None


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        train_loss = self.test(model, 'train')['loss']
        self.train(model)
        cpkg = self.pack(model, train_loss)
        return cpkg

    def pack(self, model, loss):
        return {
            "model": model,
            "loss": loss,
        }
