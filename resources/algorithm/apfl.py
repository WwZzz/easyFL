"""
This is a non-official implementation of personalized FL method APFL (https://arxiv.org/abs/2003.13461).
"""
import copy

import torch
import flgo.algorithm.fedbase
import flgo.utils.fmodule as fmodule

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'alpha':0.01})

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.v = copy.deepcopy(self.server.model)
        self.model = copy.deepcopy(self.v)

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        self.v.train()
        self.v.to(self.device)
        self.model.to(self.device)
        w_optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss_w = self.calculator.compute_loss(model, batch_data)['loss']
            loss_lm = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss = loss_w + loss_lm
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            w_optimizer.step()
            with torch.no_grad():
                for pv, plm in zip(self.v.parameters(),self.model.parameters()):
                    pv.data = pv.data - self.learning_rate*plm.grad.data
            self.model = self.alpha*self.v + (1-self.alpha)*model
            loss_new = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss_new.backward()
            with torch.no_grad():
                dv = self.v - model
                dalpha = 0.0
                for pdv, plm in zip(dv.parameters(), self.model.parameters()):
                    dalpha += (pdv*plm.grad).sum()
                self.alpha = self.alpha - self.learning_rate * dalpha
                self.alpha = max(min(self.alpha, 1.0), 0.0)
        return