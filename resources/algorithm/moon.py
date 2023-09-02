"""
This is a non-official implementation of MOON proposed in 'Model-Contrastive
Federated Learning (https://arxiv.org/abs/2103.16257)'. The official implementation is in https://github.com/QinbinLi/MOON.
********************************************Note***********************************************
The model used by this algorithm should be formulated by two submodules: encoder and head
"""
from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
import torch.nn.functional as F
import flgo.utils.fmodule as fmodule

def model_contrastive_loss(z, z_glob, z_prev, temperature=0.5):
    pos_sim = F.cosine_similarity(z, z_glob, dim=-1)
    logits = pos_sim.reshape(-1, 1)
    if z_prev is not None:
        neg_sim = F.cosine_similarity(z, z_prev, dim=-1)
        # neg_sim = self.cos(z, z_prev)
        logits = torch.cat((logits, neg_sim.reshape(-1, 1)), dim=1)
    logits /= temperature
    return F.cross_entropy(logits, torch.zeros(z.size(0)).long().to(logits.device))

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu': 0.1, 'tau':0.5})

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.model = None

    @fmodule.with_multi_gpus
    def train(self, model):
        # init global model and local model
        global_model = copy.deepcopy(model)
        global_model.freeze_grad()
        if self.model is not None:
            self.model.to(model.get_device())
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            model.zero_grad()
            z = model.encoder(batch_data[0])
            loss = self.calculator.criterion(model.head(z), batch_data[-1])
            # calculate model contrastive loss
            z_glob = global_model.encoder(batch_data[0])
            z_prev = self.model.encoder(batch_data[0]) if self.model is not None else None
            loss_con = model_contrastive_loss(z, z_glob, z_prev, self.tau)
            loss = loss + self.mu * loss_con
            loss.backward()
            optimizer.step()
        # update local model (move local model to CPU memory for saving GPU memory)
        self.model = copy.deepcopy(model).to(torch.device('cpu'))
        self.model.freeze_grad()
        return

