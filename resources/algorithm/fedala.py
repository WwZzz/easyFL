"""
This is a non-official implementation of personalized FL method FedALA (https://ojs.aaai.org/index.php/AAAI/article/view/26330).
The implementation here is directly transfered from their github repo (https://github.com/TsingZ0/FedALA/tree/main)
"""
import flgo.algorithm.fedbase
import copy
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import numpy as np

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'p':0, 's':1.0, 'eta':0.1})

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = None
        self.weights = None
        self.start_phase = True
        self.num_pre_loss = 10
        self.threshold = 0.1

    def pack(self, model, *args, **kwargs):
        r"""
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.

        Args:
            model: the locally trained model

        Returns:
            package: a dict that contains the necessary information for the server
        """
        return {
            "model": copy.deepcopy(self.model)
        }

    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        # initialize local model to the global model if local model is None,
        # and deactivate ALA at the 1st communication iteration by recoginizing the first round automatically
        if self.model is None:
            self.model = copy.deepcopy(global_model)
            return self.model

        # load the global encoder into local model
        params_global = list(global_model.parameters())
        params_local = list(self.model.parameters())
        for param, param_g in zip(params_local[:-self.p], params_global[:-self.p]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        local_model_temp = copy.deepcopy(self.model)
        params_local_temp = list(local_model_temp.parameters())

        # only consider higher layers
        params_local_head = params_local[-self.p:] # local model
        params_global_head = params_global[-self.p:] # global model
        params_local_temp_head = params_local_temp[-self.p:] # copy of local model

        # frozen the graident of the encoder in temp local model for efficiency
        for param in params_local_temp[:-self.p]: param.requires_grad = False

        # adaptively aggregate local model and global model by the weight into local temp model's head
        if self.weights is None: self.weights = [torch.ones_like(param.data).to(self.device) for param in params_local_head]
        for param_t, param, param_g, weight in zip(params_local_temp_head, params_local_head, params_global_head, self.weights):
            param_t.data = param + (param_g - param) * weight
        # weight learning
        # randomly sample partial local training data
        rand_num = int(self.s * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(Subset(self.train_data, list(range(rand_idx, rand_idx + rand_num))), self.batch_size, drop_last=True)
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        # train local aggregation weights (line 8-14)
        while True:
            for batch_data in rand_loader:
                loss = self.calculator.compute_loss(local_model_temp, batch_data)['loss']
                loss.backward()
                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_local_temp_head, params_local_head, params_global_head, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_local_temp_head, params_local_head,
                                                           params_global_head, self.weights):
                    param_t.data = param + (param_g - param) * weight
            losses.append(loss.item())
            cnt += 1
            # only train one epoch in the subsequent iterations
            if not self.start_phase: break
            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                self.gv.logger.info('Client:{}\tStd:{}\tALA epochs:{}'.format(self.id, np.std(losses[-self.num_pre_loss:]), cnt))
                break
        self.start_phase = False

        # copy the aggregated head into local model (line 15)
        for param, param_t in zip(params_local_head, params_local_temp_head):
            param.data = param_t.data.clone()
        return self.model
