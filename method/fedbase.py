import numpy as np
from task import datafuncs, modelfuncs
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from task.modelfuncs import device,lossfunc,optim

class BaseServer():
    def __init__(self, option, model, clients):
        # basic setting
        self.model = model
        self.trans_model = copy.deepcopy(self.model).to(modelfuncs.device)
        self.eval_interval = option['eval_interval']
        # clients settings
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_vols = [ck.datasize for ck in self.clients]
        self.data_vol = sum(self.client_vols)
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        # sampling and aggregating methods
        self.sample_option = option['sample']
        self.agg_option = option['aggregate']
        self.lr=option['learning_rate']
        # names of additional parameters
        self.paras_name=[]

    def run(self):
        client_accuracy=[]
        std_accuracy=[]
        ave_accuracy=[]
        ave_train_loss=[]
        for round in range(self.num_rounds+1):
            print("Round {}".format(round))
            if self.eval_interval>0 and (round==0 or round%self.eval_interval==0):
                accs, _ = self.test_on_clients(round)
                client_accuracy.append(accs)
                ave_accuracy.append(np.mean(accs))
                print("Mean of test accuracy: {}".format(ave_accuracy[-1]))
                std_accuracy.append(np.std(accs))
            train_losses = self.iterate(round)
            ave_train_loss.append(train_losses)
        accs, _ = self.test_on_clients(round)
        client_accuracy.append(accs)
        ave_accuracy.append(np.mean(accs))
        std_accuracy.append(np.std(accs))
        outdict={
            "acc_dist":accs,
            "mean_curve":ave_accuracy,
            "var_curve":std_accuracy,
            "train_loss":ave_train_loss,
            "client_accs":{}
            }
        for cid in range(self.num_clients):
            outdict['client_accs'][self.clients[cid].name]=[client_accuracy[i][cid] for i in range(len(client_accuracy))]
        return outdict

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # wait for replying of the update and loss
        return self.clients[cid].reply()

    def iterate(self, t):
        ws, losses = [], []
        # sample clients
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
        # aggregate
        w_new = self.aggregate(ws, p=[1.0*self.client_vols[id]/self.data_vol for id in selected_clients])
        self.model.load_state_dict(w_new)
        # output info
        loss_avg = sum(losses) / len(losses)
        return loss_avg

    def sample(self, replacement=False):
        cids = [i for i in range(self.num_clients)]
        selected_cids = []
        if self.sample_option == 'uniform':
            selected_cids = np.random.choice(cids, self.clients_per_round, replace=replacement)
        elif self.sample_option =='prob':
            selected_cids = np.random.choice(cids, self.clients_per_round, replace=replacement, p=[nk/self.data_vol for nk in self.client_vols])
        return list(selected_cids)

    def aggregate(self, ws, p=[]):
        """
         weighted_scale                 |uniform                    |weighted_com
        ============================================================================================
        N/K * Σpk * wk                 |1/K * Σwi                  |(1-Σpk) * w_old + Σpk * wk
        """
        if self.agg_option == 'weighted_scale':
            K = len(ws)
            N = self.num_clients
            q = [1.0*pk*N/K for pk in p]
            return modelfuncs.modeldict_weighted_average(ws, q)
        elif self.agg_option == 'uniform':
            return modelfuncs.modeldict_weighted_average(ws)
        elif self.agg_option == 'weighted_com':
            return modelfuncs.modeldict_add(modelfuncs.modeldict_scale(self.model.state_dict(), 1 - sum(p)), modelfuncs.modeldict_weighted_average(ws, p))

    def test_on_clients(self, round):
        accs, losses = [], []
        for c in self.clients:
            c.setModel(self.model)
            acc, loss = c.test()
            accs.append(acc)
            losses.append(loss)
        return accs, losses


class BaseClient():
    def __init__(self,  option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        self.name = name
        self.frequency = 0
        # client's dataset
        if not partition:
            self.train_data = datafuncs.XYDataset(data_train_dict['x'], data_train_dict['y'])
            self.test_data = datafuncs.XYDataset(data_test_dict['x'], data_test_dict['y'])
            self.val_data = datafuncs.XYDataset(data_test_dict['x'], data_test_dict['y'])
        else:
            data_x = data_train_dict['x'] + data_test_dict['x']
            data_y = data_train_dict['y'] + data_test_dict['y']
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
            self.train_data = datafuncs.XYDataset(data_x[:int(len(data_x)*0.8)], data_y[:int(len(data_y)*0.8)])
            self.test_data = datafuncs.XYDataset(data_x[int(len(data_x)*0.8):int(len(data_x)*0.9)], data_y[int(len(data_x)*0.8):int(len(data_x)*0.9)])
            self.val_data = datafuncs.XYDataset(data_x[int(len(data_x)*0.9):], data_y[int(len(data_x)*0.9):])
        self.datasize = len(self.train_data)
        # hyper-parameters for training
        self.epochs = option['num_epochs']
        self.learning_rate = option['learning_rate']
        self.batch_size = option['batch_size']
        self.regularzation = option['regularzation']
        self.momentum = option['momentum']
        self.model = None

    def setModel(self, model):
        self.model = model

    def train(self):
        self.model.train()
        if self.batch_size == -1:
            # full gradient descent
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                self.model.zero_grad()
                outputs = self.model(images)
                loss = lossfunc(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def test(self, dataflag='test'):
        if dataflag == 'test':
            return modelfuncs.test(self.model, self.test_data)
        elif dataflag == 'train':
            return modelfuncs.test(self.model, self.train_data)
        elif dataflag =='validate':
            return modelfuncs.test(self.model, self.val_data)

    def reply(self):
        loss = self.train()
        return copy.deepcopy(self.model.state_dict()), loss