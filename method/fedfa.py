from utils import fmodule
import copy
from .fedbase import BaseServer, BaseClient
import numpy as np

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.m = fmodule.modeldict_zeroslike(self.model.state_dict())
        self.beta = option['beta']
        self.alpha = 1.0 - self.beta
        self.gamma = option['gamma']
        self.learning_rate = option['learning_rate']
        self.paras_name=['beta','gamma','momentum']

    def unpack(self, pkgs):
        ws = [p["model"].state_dict() for p in pkgs]
        losses = [p["train_loss"] for p in pkgs]
        ACC = [p["acc"] for p in pkgs]
        freq = [p["freq"] for p in pkgs]
        return ws, losses, ACC, freq

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # training
        ws, losses, ACC, F = self.communicate(selected_clients)
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
        wnew = self.aggregate(ws, p)
        dw = wnew -self.model
        # calculate m = γm+(1-γ)dw
        self.m = self.gamma*self.m, self.gamma + (1 - self.gamma)*dw
        self.model = wnew - self.m * self.learning_rate
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, partition = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, partition, drop_rate)
        self.frequency = 0

    def pack(self, model):
        self.frequency += 1
        acc, loss = self.test(model,'train')
        return {
            "model":model,
            "train_loss":loss,
            "acc":acc,
            "freq":self.frequency,
        }
