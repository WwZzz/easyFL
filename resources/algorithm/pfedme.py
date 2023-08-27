"""
This is a non-official implementation of personalized FL method pFedMe (http://arxiv.org/abs/2006.08848).
The original implementation is in github repo (https://github.com/CharlieDinh/pFedMe/)
"""
import copy
import torch
import flgo.utils.fmodule as fmodule
import flgo.algorithm.fedbase

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'beta':1.0, 'lambda_reg':15.0, 'K':5})

    def aggregate(self, models):
        return (1-self.beta)*self.model+self.beta*fmodule._model_average(models)

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = None

    def train(self, model):
        if self.model is None:
            self.model = copy.deepcopy(model)
        optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # line 7 in Algorithm 1 pFedMe
            batch_data = self.get_batch_data()
            self.model.load_state_dict(model.state_dict())
            self.model.train()
            self.model.zero_grad()
            model.freeze_grad()
            for _ in range(self.K):
                loss = self.calculator.compute_loss(self.model, batch_data)['loss']
                for param_theta_i, param_wi in zip(self.model.parameters(), model.parameters()):
                    loss += self.lambda_reg*0.5*torch.sum((param_theta_i-param_wi)**2)
                loss.backward()
                if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_grad)
                optimizer.step()
            model.enable_grad()
            # line 8
            with torch.no_grad():
                for param_wi, param_thetai in zip(model.parameters(), self.model.parameters()):
                    param_wi.data = param_wi.data - self.learning_rate * self.lambda_reg * (param_wi.data - param_thetai)
            self.model.load_state_dict(model.state_dict())
        return

