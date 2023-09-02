"""
This is a non-official implementation of personalized FL method LG-Fedavg (http://arxiv.org/abs/2001.01523).
The original implementation is in github repo (https://github.com/pliang279/LG-FedAvg)
"""
import copy
import torch
import flgo.algorithm.fedbase

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'Kp':2})

class Client(flgo.algorithm.fedbase.BasicClient):
    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        if self.model is None: self.model = copy.deepcopy(global_model)
        # only update the final p layers for the local model
        num_layers = len(list(global_model.parameters()))
        with torch.no_grad():
            for lid,(pg, pl) in enumerate(zip(global_model.parameters(), self.model.parameters())):
                if lid>=num_layers-self.Kp:break
                pl.data = pg.data.clone()
        return self.model