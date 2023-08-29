"""
This is a non-official implementation of personalized FL method FedAMP (http://arxiv.org/abs/2007.03797).
"""
import copy
import torch
import flgo.utils.fmodule as fmodule
import flgo.algorithm.fedbase

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'lmbd':1.0, 'alpha':10000, 'sigma':25.0, 'heur':True, 'sa':1.0/self.num_clients, 'alpha_decay_rate':0.1, 'alpha_decay_interval':30})
        self.U = [copy.deepcopy(self.model) for _ in self.clients]
        self.sample_option = 'full'

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {'model': self.U[client_id]}

    def get_weights(self, models, heur=False):
        n = len(models)
        weights = torch.zeros((n, n))
        if heur:
            for i in range(n):
                for j in range(i):
                    weights[i][j] = weights[j][i] = torch.exp(self.sigma*fmodule._model_cossim(models[i], models[j]))
            for i in range(n):
                weights[i] = weights[i]/weights[i].sum()
                weights[i] = weights[i]*(1-self.sa)
                weights[i][i] = self.sa
        else:
            def dA(x):
                return torch.exp(-x/self.sigma)/self.sigma
            weights = torch.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    weights[i][j] = weights[j][i] = self.alpha* dA(fmodule._model_norm(models[i]-models[j])**2)
            for i in range(n):
                weights[i][i] = 1.0-weights[i].sum()
        return weights

    def iterate(self):
        models = self.communicate([i for i in range(self.num_clients)])['model']
        # compute aggregation weights for each client
        weights = self.get_weights(models, self.heur)
        # aggregate
        for i,ci in enumerate(self.received_clients):
            self.U[ci] = fmodule._model_sum([wj*mj for wj,mj in zip(weights[i], models)])
        # decay alpha
        if self.current_round%self.alpha_decay_interval==0:
            self.alpha = self.alpha*self.alpha_decay_rate
            for c in self.clients: c.alpha = self.alpha

class Client(flgo.algorithm.fedbase.BasicClient):
    def train(self, model):
        self.model = copy.deepcopy(model)
        u = model
        optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(self.model, batch_data)['loss']
            for pm, pu in zip(self.model.parameters(), u.parameters()):
                loss += 0.5*self.lmbd/self.alpha*torch.sum((pm - pu)**2)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for pm, pl in zip(model.parameters(), self.model.parameters()):
                pm.data = pl.data.clone()
        return