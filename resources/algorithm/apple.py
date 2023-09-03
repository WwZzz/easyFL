"""
This is a non-official implementation of personalized FL method APPLE (https://www.ijcai.org/proceedings/2022/301).
The original implementation is in github repo https://github.com/ljaiverson/pFL-APPLE
"""
import copy
import torch
import numpy as np
import flgo.utils.fmodule as fmodule
import flgo.algorithm.fedbase

class Server(flgo.algorithm.fedbase.BasicServer):
    def initialize(self):
        self.init_algo_para({'M':self.num_clients-1, 'mu':0.1, 'lr_dr':1e-3, 'ratioL':0.1, 'type_loss_scheduler':'cos', })
        self.L = self.num_rounds*self.ratioL
        for cid,c in enumerate(self.clients):
            c.L = self.L
            c.cid = cid
        self.core_models = [copy.deepcopy(self.model) for _ in self.clients]
        client_data_vols = torch.FloatTensor([len(c.train_data) for c in self.clients])
        self.p0 = client_data_vols/client_data_vols.sum()
        self.select_mat = np.eye(self.num_clients)
        self.pss = torch.stack([self.p0.clone() for _ in self.clients])
        self.sample_option = 'full'

    def pack(self, client_id, mtype=0):
        # select core models for client_id
        res = []
        # select clients that never have been selected
        zero_clients= np.where(self.select_mat[client_id]==0.0)[0]
        if len(zero_clients)>0:
            if len(zero_clients)>self.M:
                client_js = np.random.choice(zero_clients, self.M, replace=False).tolist()
            else:
                client_js = zero_clients.tolist()
            res.extend(client_js)
            for j in res:
                self.select_mat[client_id][j] = 1.
        if len(res)<self.M:
            br = max(1.5, self.current_round*self.M/self.num_clients)
            have_been_selected_clients = [cid for cid in range(self.num_clients) if (cid !=client_id and cid not in res)]
            probs = np.array([np.power(br, np.abs(pij.item())) for pij in self.pss[client_id][have_been_selected_clients]])
            probs /= probs.sum()
            append_clients = np.random.choice(have_been_selected_clients, self.M-len(res), replace=False, p=probs).tolist()
            res.extend(append_clients)
        res = sorted(res)
        return {
            'core_models': [self.core_models[j] for j in res],
            'core_ids': res,
            'r': self.current_round,
        }

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        models, ps = res['model'], res['p']
        # update core models and p
        for cid, mi, pi in zip(self.received_clients, models, ps):
            self.core_models[cid] = mi
            self.pss[cid] = pi

class Client(flgo.algorithm.fedbase.BasicClient):
    def initialize(self):
        self.p0 = self.server.p0
        self.p = copy.deepcopy(self.p0)
        self.model = copy.deepcopy(self.server.model)
        self.core_model = copy.deepcopy(self.server.model)
        self.core_models = [None for _ in self.server.clients]
        self.core_models[self.cid] = self.core_model
        # init local core models
        for i in range(len(self.core_models)):
            if self.core_models[i] is None:
                self.core_models[i] = self.server.core_models[i]

    def reply(self, svr_pkg):
        core_ids = self.unpack(svr_pkg)
        self.train(None)
        return self.pack(core_ids)

    def unpack(self, svr_pkg):
        for j, mj in zip(svr_pkg['core_ids'], svr_pkg['core_models']):
            self.core_models[j] = mj
        self.current_round = svr_pkg['r']
        return svr_pkg['core_ids']

    def loss_scheduler(self, r):
        if self.type_loss_scheduler=='cos':
            return 0.5*(np.cos(r*np.pi/self.L)+1) if r<= self.L else 0.0
        else:
            return np.pow(1e-3, 1.0*r/self.L) if r<=self.L else 0.0

    def pack(self, core_ids):
        return {
            'model':copy.deepcopy(self.core_model),
            'p': self.p,
        }

    def train(self, model=None):
        # compute personalized model w_p
        pi = self.p.detach().to(self.device)
        pi.requires_grad = True
        self.p0 = self.p0.to(self.device)
        optimizer = self.calculator.get_optimizer(self.core_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        pi_optim = torch.optim.SGD([{'params':pi}], lr=self.lr_dr)
        for i, m in enumerate(self.core_models):
            if i==self.cid: continue
            m.freeze_grad()
        for _ in range(self.num_steps):
            self.core_model.zero_grad()
            pi_optim.zero_grad()
            self.model = fmodule._model_average(self.core_models, self.p)
            batch_data = self.get_batch_data()
            loss = self.calculator.compute_loss(self.model, batch_data)['loss']
            coef = 0.5*self.mu*self.loss_scheduler(self.current_round)
            loss += coef * torch.sum((pi - self.p0)**2)
            loss.backward()
            # compute gradient of local core model
            pself = self.p[self.cid]
            for pm, psc in zip(self.model.parameters(), self.core_model.parameters()):
                psc.grad = pself * pm.grad.data.clone()
            # fix gradient of pi
            with torch.no_grad():
                pi_grad = torch.zeros_like(pi)
                for j in range(len(self.core_models)):
                    for pm, pj in zip(self.model.parameters(), self.core_models[j].parameters()):
                        pi_grad[j] += torch.sum(pm.grad.data*pj)
                pi.grad += pi_grad
            pi_optim.step()
            optimizer.step()
        self.p = pi
        self.model = fmodule._model_average(self.core_models, self.p)
        return