"""
This is a non-official implementation of personalized FL method GPFL (https://arxiv.org/abs/2308.10279).
The original implementation at in https://github.com/TsingZ0/GPFL
"""
import flgo.algorithm.fedbase
import copy
import torch
import flgo.utils.fmodule as fmodule
import torch.nn as nn
import collections
import torch.nn.functional as F
class Server(flgo.algorithm.fedbase.BasicServer):
    """
    Hyper-parameters:
        lmbd (float): the coefficient of the regularization term
        mu (float): the coefficient of the weight decay on parameters C and CoV
    """
    def initialize(self):
        self.init_algo_para({'lmbd': 1e-2, 'mu':1e-2})
        self.num_classes = len(collections.Counter([d[-1] for d in self.test_data]))
        with torch.no_grad():
            x = self.test_data[0]
            self.model.to('cpu')
            h = self.model.encoder(x[0].unsqueeze(0))
            self.dim = h.shape[-1]
        dec_model = self.get_decorate_model()
        self.model = dec_model

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        self.model = self.aggregate(models)
        return len(models) > 0

    def get_decorate_model(self):
        dim = self.dim
        encoder = self.model.encoder
        head = self.model.head
        cov = self.init_cov()
        C = torch.nn.Embedding(self.num_classes, dim)
        g = C.weight.data.mean(dim=0)
        p = C.weight.data.mean(dim=0)
        class DecModel(fmodule.FModule):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.head = head
                self.cov = cov
                self.C = C
                self.g = g
                self.p = p

            def forward(self, x):
                fi = self.encoder(x)
                # compute {gamma, beta} and {gamma_i, beta_i}
                self.g = self.g.to(x.device)
                self.p = self.p.to(x.device)
                gamma, beta = self.cov(fi, self.g)
                gamma_i, beta_i = self.cov(fi, self.p)
                fi_G = F.relu((gamma + 1) * fi + beta)
                fi_P = F.relu((gamma_i + 1) * fi + beta_i)
                outputs = self.head(fi_P)
                if self.training:
                    return outputs, fi_G, fi_P
                else:
                    return outputs

            def set_module(self, encoder=None, head=None, cov=None, C=None, g=None, p=None):
                if encoder is not None: self.encoder = encoder
                if head is not None: self.head = head
                if cov is not None: self.cov = cov
                if C is not None: self.C = C
                if g is not None:
                    self.g = g
                    g.requires_grad = False
                if p is not None:
                    p.requires_grad = False
        return DecModel()

    def init_cov(self):
        dim = self.dim
        class CPN(fmodule.FModule):
            def __init__(self):
                super().__init__()
                self.gamma_gen = nn.Sequential(
                    nn.Linear(2*dim, dim),
                    nn.ReLU(),
                    nn.LayerNorm(dim)
                )
                self.beta_gen = nn.Sequential(
                    nn.Linear(2*dim, dim),
                    nn.ReLU(),
                    nn.LayerNorm(dim)
                )
            def forward(self, x, conditional_input):
                x = torch.cat([x, torch.tile(conditional_input, dims=(len(x),1))], dim=1)
                return self.gamma_gen(x), self.beta_gen(x)

        return CPN()

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.model = copy.deepcopy(self.server.model)
        label_counter = collections.Counter([d[-1] for d in self.train_data])
        self.sizes_label = torch.zeros(self.server.num_classes)
        for lb in range(self.server.num_classes):
            if lb in label_counter.keys():
                self.sizes_label[lb] = label_counter[lb]
        self.alpha = self.sizes_label/self.sizes_label.sum()

    def unpack(self, svr_pkg):
        # copy encoder, CoV, C's parameters into local model
        gmodel = svr_pkg['model']
        for pg, pl in zip(gmodel.encoder.parameters(), self.model.encoder.parameters()):
            pl.data = pg.data.clone()
        for pg, pl in zip(gmodel.cov.parameters(), self.model.cov.parameters()):
            pl.data = pg.data.clone()
        self.model.C.weight.data = gmodel.C.weight.data.clone()
        self.model.set_module(g=self.model.C.weight.data.mean(dim=0), p=(self.alpha.view(1, -1)@self.model.C.weight.data)/self.server.num_classes)
        return self.model

    @fmodule.with_multi_gpus
    def train(self, model):
        # model.C = model.C.to(self.device)
        gC = copy.deepcopy(model.C)
        gC.requires_grad = False
        # no updating on the local head
        for p in model.head.parameters(): p.requires_grad = False
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate, momentum=self.momentum
        )
        model.train()
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.calculator.to_device(self.get_batch_data())
            outputs, fG, fP = model(batch_data[0])
            # compute loss_alg = -log[exp(cossim(f_G, C(yi)))/sum(exp_k(cossim(fi_G, C(k))))]
            loss_alg = 0.
            for i in range(len(batch_data[-1])):
                loss_alg += angle_level_guidence(fG[i], batch_data[-1][i].item(), model.C.weight.data)
            loss_alg /= len(batch_data[-1])
            # compute loss_mlg = ||f_G - gC(y)||_2^2
            gC_batch = gC(batch_data[-1])
            loss_mlg = ((fG-gC_batch)**2).sum()/len(batch_data[-1])
            # compute ERM risk
            loss_erm = self.calculator.criterion(outputs, batch_data[-1])
            # compute l2 regularizaiton
            loss_l2 = torch.norm(model.C.weight.data)
            for p in model.cov.parameters(): loss_l2 += torch.norm(p)
            loss = loss_erm + loss_alg + self.lmbd *loss_mlg  + self.mu*loss_l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return

def angle_level_guidence(fi, y, C):
    exp_cos_sim = torch.exp(torch.cosine_similarity(fi, C, dim=-1))
    return -torch.log(exp_cos_sim[y]/exp_cos_sim.sum())