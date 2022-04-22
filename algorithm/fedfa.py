from utils import fmodule
from .fedbase import BasicServer, BasicClient
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.frequency = 0
        self.m = fmodule._modeldict_zeroslike(self.model.state_dict())
        self.beta = option['beta']
        self.alpha = 1.0 - self.beta
        self.gamma = option['gamma']
        self.eta = option['learning_rate']
        self.paras_name=['beta','gamma']

    def iterate(self, t):
        # sample clients
        self.selected_clients = self.sample()
        # training
        res = self.communicate(self.selected_clients)
        models, losses, ACC, F = res['model'], res['loss'], res['acc'], res['freq']
        # aggregate
        # calculate ACCi_inf, fi_inf
        sum_acc = np.sum(ACC)
        sum_f = np.sum(F)
        ACCinf = [-np.log2(1.0*acc/sum_acc+0.000001) for acc in ACC]
        Finf = [-np.log2(1-1.0*f/sum_f+0.00001) for f in F]
        sum_acc = np.sum(ACCinf)
        sum_f = np.sum(Finf)
        ACCinf = [acc/sum_acc for acc in ACCinf]
        Finf = [f/sum_f for f in Finf]
        # calculate weight = αACCi_inf+βfi_inf
        p = [self.alpha*accinf+self.beta*finf for accinf,finf in zip(ACCinf,Finf)]
        wnew = self.aggregate(models, p)
        dw = wnew -self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma*self.m, self.gamma + (1 - self.gamma)*dw
        self.model = wnew - self.m * self.eta
        return

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.frequency = 0
        self.momentum = option['gamma']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        metrics = self.test(model,'train')
        acc, loss = metrics['accuracy'], metrics['loss']
        self.train(model)
        cpkg = self.pack(model, loss, acc)
        return cpkg

    def pack(self, model, loss, acc):
        self.frequency += 1
        return {
            "model":model,
            "loss":loss,
            "acc":acc,
            "freq":self.frequency,
        }
