"""
This is a non-official implementation of MOON proposed in 'Model-Contrastive
Federated Learning (https://arxiv.org/abs/2103.16257)'. The official
implementation is in https://github.com/QinbinLi/MOON. 
"""
from .fedbase import BasicServer, BasicClient
import copy
import torch
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        if not "get_embedding" in dir(model):
            raise NotImplementedError("the model used by Moon should have the method `get_embedding` to obtain the intermediate result of forward")
        self.paras_name = ['mu']

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.mu = option['mu']
        # the temperature (tau) is set 0.5 as default
        # self.tau = option['tau']
        self.tau = 0.5
        self.local_model = None
        self.contrastive_loss = ModelContrastiveLoss(self.tau)

    def train(self, model):
        # init global model and local model
        global_model = copy.deepcopy(model)
        global_model.freeze_grad()
        if self.local_model:
            self.local_model.to(fmodule.device)
        model.train()
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.train_one_step(model, batch_data)['loss']
            # calculate model contrastive loss
            batch_data = self.calculator.data_to_device(batch_data)
            z = model.get_embedding(batch_data[0])
            z_glob = global_model.get_embedding(batch_data[0])
            z_prev = self.local_model.get_embedding(batch_data[0]) if self.local_model else None
            loss_con = self.contrastive_loss(z, z_glob, z_prev)
            loss = loss + self.mu * loss_con
            loss.backward()
            optimizer.step()
        # update local model (move local model to CPU memory for saving GPU memory)
        self.local_model = copy.deepcopy(model).to(torch.device('cpu'))
        self.local_model.freeze_grad()
        return

class ModelContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ModelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, z, z_glob, z_prev):
        pos_sim = self.cos(z, z_glob)
        logits = pos_sim.reshape(-1, 1)
        if z_prev is not None:
            neg_sim = self.cos(z, z_prev)
            logits = torch.cat((logits, neg_sim.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        return self.cross_entropy(logits, torch.zeros(z.size(0)).long().to(fmodule.device))
