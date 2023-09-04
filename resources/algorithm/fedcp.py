"""
This is a non-official implementation of personalized FL method FedCP (https://github.com/TsingZ0/FedCP.).
The original implementation is in github repo (https://github.com/TsingZ0/FedCP)
"""
import flgo.algorithm.fedbase
import copy
import torch
import flgo.utils.fmodule as fmodule
import torch.nn as nn
import torch.nn.functional as F
class Server(flgo.algorithm.fedbase.BasicServer):
    """
    Hyper-parameters:
        lmbd (float): the coefficient of the regularization term
    """
    def initialize(self):
        self.init_algo_para({'lmbd': 1.0})
        with torch.no_grad():
            x = self.test_data[0]
            self.model.to('cpu')
            h = self.model.encoder(x[0].unsqueeze(0))
            self.dim = h.shape[-1]
        self.cpn = self.init_cpn()

    def pack(self, client_id, mtype=0):
        if mtype==0:
            return {}
        elif mtype==1:
            return {
                'model': copy.deepcopy(self.model),
                'cpn': copy.deepcopy(self.cpn),
            }

    def iterate(self):
        self.selected_clients = self.sample()
        # ask clients to train
        res = self.communicate(self.selected_clients)
        models, cpns = res['model'], res['cpn']
        self.model = self.aggregate(models)
        self.cpn = self.aggregate(cpns)
        # broadcast model and cpn to clients
        self.communicate(self.selected_clients, mtype=1)
        return

    def init_cpn(self):
        dim = self.dim
        class CPN(fmodule.FModule):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(dim, 2*dim)
                self.relu = nn.ReLU()
                self.ln = nn.LayerNorm([2*dim])

            def forward(self, x):
                x = self.fc(x)
                x = self.ln(x)
                x = self.relu(x)
                # x = x.view(2, -1)
                return x
        return CPN()

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.local_model = copy.deepcopy(self.server.model)
        self.cpn = copy.deepcopy(self.server.cpn)
        self.v =  torch.rand(1, self.server.dim).to(self.device)
        self.global_model = copy.deepcopy(self.server.model)
        self.model = self.init_infer_model()
        self.model.set_module(copy.deepcopy(self.global_model.head), self.local_model, self.cpn, self.v)
        self.actions = {0: self.reply, 1:self.update_local_module}

    def pack(self):
        return {
            'model': self.global_model,
            'cpn': self.cpn,
        }

    def reply(self, svr_pkg):
        self.train(self.global_model)
        return self.pack()

    def unpack(self, svr_pkg):
        return self.global_model

    def update_local_module(self, svr_pkg):
        global_model = svr_pkg['model']
        # update local model's encoder
        for pg, pm in zip(global_model.encoder.parameters(), self.local_model.encoder.parameters()):
            pm.data = pg.data.clone()
        # update v
        self.v = self.local_model.head.weight.data.clone().sum(dim=0)
        # update cpn
        self.cpn = svr_pkg['cpn']
        # update inference model
        self.model.set_module(copy.deepcopy(global_model.head), self.local_model, self.cpn, self.v)
        self.global_model = copy.deepcopy(global_model)
        return

    @fmodule.with_multi_gpus
    def train(self, global_model):
        # no updating on global model
        global_model.freeze_grad()
        self.local_model.to(self.device)
        self.cpn.to(self.device)
        self.v.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD([{"params": self.local_model.parameters()}, {"params": self.cpn.parameters()}], lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.calculator.to_device(self.get_batch_data())
            hi = self.local_model.encoder(batch_data[0])
            ci = self.v / torch.norm(self.v, 2) * hi
            mask = self.cpn(ci)
            mask = mask.view(-1, 2,mask.shape[-1]//2)
            mask = F.gumbel_softmax(mask, dim=1, tau=1, hard=False)
            ri,si = mask[:,0,:], mask[:,1,:]
            y_local = self.local_model.head(si * hi)
            y_g = global_model.head(ri * hi)
            outputs = y_g+y_local
            loss_erm = self.calculator.criterion(outputs, batch_data[-1])
            # regularization term
            hg = global_model.encoder(batch_data[0])
            loss_reg = mmd_rbf_noaccelerate(hi, hg, device=self.device)
            loss = loss_erm + self.lmbd * loss_reg
            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=self.local_model.parameters(), max_norm=self.clip_grad)
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=self.cpn(), max_norm=self.clip_grad)
            optimizer.step()
        with torch.no_grad():
            for pg, pm in zip(global_model.encoder.parameters(), self.local_model.encoder.parameters()):
                pg.data = pm.data.clone()
            for pg, pm in zip(global_model.head.parameters(), self.local_model.head.parameters()):
                pg.data = 0.5*(pg.data+pm.data)
        return

    def init_infer_model(self):
        class InferModel(fmodule.FModule):
            def __init__(self, ghead=None, local_model=None, cpn=None, v=None):
                super(InferModel, self).__init__()
                self.ghead = ghead
                self.local_model = local_model
                self.cpn = cpn
                self.v = v

            def forward(self, x):
                hi = self.local_model.encoder(x)
                ci = self.v/torch.norm(self.v, 2)*hi
                mask = self.cpn(ci)
                mask = mask.view(-1, 2, mask.shape[-1] // 2)
                mask = F.gumbel_softmax(mask, dim=1, tau=1, hard=False)
                ri, si = mask[:, 0, :], mask[:, 1, :]
                y_local = self.local_model.head(si * hi)
                y_g = self.ghead(ri * hi)
                return y_g+y_local

            def set_module(self, ghead, local_model, cpn, v):
                self.ghead = ghead
                self.local_model = local_model
                self.cpn = cpn
                self.v = v
        return InferModel()

def mmd_rbf_noaccelerate(x, y, kernel='rbf', device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)
