"""
This is a non-official implementation of AFL proposed in 'Agnostic
Federated Learning (http://arxiv.org/abs/1902.00146)'. We refer to the
Github repo fair_flearn ('https://github.com/litian96/fair_flearn') when
implementing this algorithm.
"""

import flgo.utils.fmodule as fmodule
from flgo.algorithm.fedbase import BasicServer, BasicClient
import numpy as np
import copy
import collections

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'learning_rate_lambda': 0.01})
        self.dynamic_lambdas = np.ones(self.num_clients) * 1.0 / self.num_clients
        self.learning_rate = self.option['learning_rate']
        self.result_model = copy.deepcopy(self.model)

    def iterate(self):
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
        self.result_model = (self.current_round * self.result_model + self.model) / (self.current_round + 1)
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

    def global_test(self, flag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        Args:
            flag: choose train data or valid data to evaluate
        Returns:
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.result_model, flag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None, flag='test'):
        if model == None: model = self.result_model
        data = self.test_data if flag=='test' else self.valid_data
        if data is None: return {}
        else:
            return self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])

class Client(BasicClient):
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