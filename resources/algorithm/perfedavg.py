"""
This is a non-official implementation of personalized FL method Per-FedAvg (http://arxiv.org/abs/2002.07948).
Our implementation considers both the two efficient versions of Per-FedAvg (FO) and Per-FedAvg (HF)
"""
import copy
import torch
import flgo.utils.fmodule as fmodule
import flgo.algorithm.fedbase

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        # ver in ['fo', 'hf']
        self.init_algo_para({'alpha':0.01, 'ver':'hf', 'delta':0.001})
        self.aggregation_option = 'uniform'

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = None
        self.beta = self.learning_rate

    def train(self, model):
        optimizer = self.calculator.get_optimizer(model, lr=self.alpha, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            origin_model = copy.deepcopy(model)
            final_model = copy.deepcopy(model)
            # Step 1. w_tmp = w - alpha * ▽w(D) where D is a batch of data
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            optimizer.step()

            # Step 2. estimate the gradient at w_tmp by ▽w_tmp(D') where D' is another batch of data independent to D
            optimizer.zero_grad()
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
            grads = torch.cat([g.flatten() for g in grads])

            # Step 3. Compute Hessian of w on another batch of data D'' that is independent to both D and D''
            if self.ver.lower()=='fo':
                # ignore the hessian term
                H_mul_grads = 0.0
            elif self.ver.lower()=='hf':
                # approximatiion of hessian of original model given batched data
                batch_data = self.get_batch_data()
                # estimate H@g by (▽f(w+δg, D'') - ▽f(w-δg, D''))/(2δ) according to
                add_model = fmodule._model_from_tensor(fmodule._model_to_tensor(origin_model) + self.delta*grads, model.__class__)
                sub_model = fmodule._model_from_tensor(fmodule._model_to_tensor(origin_model) - self.delta*grads, model.__class__)
                add_model.train()
                sub_model.train()
                loss_add = self.calculator.compute_loss(add_model, batch_data)['loss']
                grads_add = torch.autograd.grad(loss_add, add_model.parameters(), retain_graph=True, create_graph=True)
                grads_add = torch.cat([g.flatten() for g in grads_add])
                loss_sub = self.calculator.compute_loss(sub_model, batch_data)['loss']
                grads_sub = torch.autograd.grad(loss_sub, sub_model.parameters(), retain_graph=True, create_graph=True)
                grads_sub = torch.cat([g.flatten() for g in grads_sub])
                H_mul_grads = (grads_add - grads_sub)/(2.0*self.delta)

            # Step 4. Update w by w <- w - beta * (I - alpha*hessian) * ▽w_tmp(D')
            w = torch.cat([p.data.flatten() for p in final_model.parameters()])
            new_w = w - self.beta*(grads - self.alpha*H_mul_grads)
            final_model = fmodule._model_from_tensor(new_w, final_model.__class__)

        # set local model for evaluation
        self.model = copy.deepcopy(final_model)
        self.model.train()
        self.model.zero_grad()
        optimizer = self.calculator.get_optimizer(self.model, lr=self.alpha, weight_decay=self.weight_decay, momentum=self.momentum)
        loss = self.calculator.compute_loss(self.model, self.get_batch_data())['loss']
        loss.backward()
        optimizer.step()
        return final_model
