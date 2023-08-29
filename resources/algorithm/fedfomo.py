"""
This is a non-official implementation of personalized FL method FedFomo (http://arxiv.org/abs/2012.08565).
The original implementation is at https://github.com/NVlabs/FedFomo
"""
import warnings

import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule
import copy
import torch
import numpy as np

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'M':10})
        self.P = np.ones((self.num_clients, self.num_clients))
        self.client_start = [False for _ in range(self.num_clients)]
        self.new_models = [None for _ in range(self.num_clients)]
        self.sends_list = [[] for _  in range(self.num_clients)]

    def pack(self, client_id, mtype):
        if not self.client_start[client_id]:
            # initialize model for the client when it was first selected
            self.client_start[client_id] = True
            return {'model': copy.deepcopy(self.model)}
        else:
            # select others' models according to P
            pi = self.P[client_id]
            available_clients = [cid for cid in range(self.num_clients) if cid!=client_id and self.new_models[cid] is not None and not self.new_models[cid].has_nan()]
            m = min(self.M, len(available_clients))
            # select top m clients' model
            self.sends_list[client_id] = [available_clients[i] for i in np.argsort(pi[available_clients])[:m]]
            return {
                'models': [self.new_models[cid] for cid in self.sends_list[client_id]]
            }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        models = res['model']
        ws = res['w']
        # update model pool and P
        for cid, mi, wi in zip(self.received_clients, models, ws):
            self.new_models[cid] = mi
            for j,wj in zip(self.sends_list[cid], wi):
                self.P[cid][j] += wj.float()


class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model_old = None
        self.model = None
        self.w = []

    def unpack(self, svr_pkg):
        if self.model_old is None:
            model = svr_pkg['model']
            self.model_old = copy.deepcopy(model)
            self.model = copy.deepcopy(model)
            return self.model
        else:
            models = svr_pkg['models']
            models.append(self.model)
            val_loss_local = self.test(self.model_old, 'val')['loss']
            weights = torch.zeros(len(models))
            for i,mi in enumerate(models):
                val_loss_m = self.test(mi, 'val')['loss']
                dl = val_loss_local - val_loss_m
                if dl>0: weights[i] = dl/(fmodule._model_norm(self.model_old-mi)+1e-6)
            self.w = weights
            wsum = weights.sum()
            w = weights/wsum if weights.sum()>0 else weights
            self.model_old = self.model
            self.model = self.model_old + fmodule._model_sum([wi * mi for wi, mi in zip(w, models)])
        return self.model

    def pack(self, model):
        return {
            'model': self.model,
            'w': self.w[:-1]
        }



