from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np
import copy

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        # algorithm hyper-parameters
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = option['learning_rate']
        self.learning_rate_lambda = option['learning_rate_lambda']
        self.result_model = copy.deepcopy(self.model)
        self.paras_name=['learning_rate_lambda']

    def iterate(self, t):
        # full sampling
        # training
        models, train_losses = self.communicate([cid for cid in range(self.num_clients)])
        grads = [(self.model - model)/self.learning_rate for model in models]
        # aggregate grads
        grad = fmodule._model_average(grads, self.dynamic_lambdas)
        self.model = self.model - self.learning_rate*grad
        # update lambdas
        self.dynamic_lambdas = [lmb_i+self.learning_rate_lambda*loss_i for lmb_i,loss_i in zip(self.dynamic_lambdas, train_losses)]
        self.dynamic_lambdas = self.project(self.dynamic_lambdas)
        # record resulting model
        self.result_model = (t*self.result_model + self.model)/(t+1)
        return

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

    def test(self, model=None):
        if model == None: model = self.result_model
        if self.test_data:
            model.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            return eval_metric, loss
        else:
            return -1, -1

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)