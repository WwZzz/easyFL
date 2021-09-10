import numpy as np
from utils import fmodule
import copy
from torch.utils.data import DataLoader
from utils.fmodule import device,lossfunc,Optim
from multiprocessing.dummy import Pool as ThreadPool
import time

class BaseServer():
    def __init__(self, option, model, clients, dtest = None):
        # basic setting
        self.task = option['task']
        self.name = option['method']
        self.model = model
        self.dtest = fmodule.XYDataset(dtest['x'], dtest['y']) if dtest else None
        self.eval_interval = option['eval_interval']
        self.num_threads = option['num_threads']
        # clients settings
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_vols = [c.datavol for c in self.clients]
        self.data_vol = sum(self.client_vols)
        self.clients_buffer = [{} for _ in range(self.num_clients)]
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.decay_rate = option['learning_rate_decay']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        self.lr_scheduler_type = option['lr_scheduler']
        self.current_round = -1
        # sampling and aggregating methods
        self.sample_option = option['sample']
        self.agg_option = option['aggregate']
        self.lr=option['learning_rate']
        # names of additional parameters
        self.paras_name=[]
        self.option = option

    def run(self):
        accs = []
        c_accs = []
        train_losses = []
        test_losses = []
        std_c_accs = []
        mean_c_accs = []
        valid_accs = []
        test_accs = []
        time_costs = []
        temp = "{:<30s}{:.4f}"
        global_timestamp_start = time.time()
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            self.current_round+=1
            timestamp_start = time.time()
            # train
            selected_clients = self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)
            timestamp_end = time.time()
            time_costs.append(timestamp_end - timestamp_start)
            print(temp.format("Time Cost:",time_costs[-1])+'s')
            # free the cache of gpu
            # torch.cuda.empty_cache()
            if self.eval_interval>0 and (round==0 or round%self.eval_interval==0 or round== self.num_rounds):
                # train
                _, train_loss = self.test_on_clients(round, dataflag='train')
                # validate
                accs, _ = self.test_on_clients(round, dataflag='valid')
                # test
                test_acc, test_loss = self.test_on_dtest()
                # record
                train_losses.append(1.0*sum([ck * closs for ck, closs in zip(self.client_vols, train_loss)])/self.data_vol)
                c_accs.append(accs)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                valid_accs.append(1.0*sum([ck * acc for ck, acc in zip(self.client_vols, accs)])/self.data_vol)
                mean_c_accs.append(np.mean(accs))
                std_c_accs.append(np.std(accs))
                # print
                print(temp.format("Training Loss:", train_losses[-1]))
                print(temp.format("Testing Loss:", test_losses[-1]))
                print(temp.format("Testing Accuracy:", test_accs[-1]))
                print(temp.format("Validating Accuracy:",valid_accs[-1]))
                print(temp.format("Mean of Client Accuracy:", mean_c_accs[-1]))
                print(temp.format("Std of Client Accuracy:", std_c_accs[-1]))
        global_timestamp_end = time.time()
        print("=================End==================")
        print(temp.format("Total Time Cost", global_timestamp_end-global_timestamp_start) + 's')
        print(temp.format("Mean Time Cost Of Each Round", float(np.mean(time_costs))) + 's')
        # create_outdict
        outdict={
            "meta":self.option,
            "acc_dist":accs,
            "mean_curve":mean_c_accs,
            "var_curve":std_c_accs,
            "train_losses":train_losses,
            "test_accs":test_accs,
            "test_losses":test_losses,
            "valid_accs":valid_accs,
            "client_accs":{},
            }
        for cid in range(self.num_clients):
            outdict['client_accs'][self.clients[cid].name]=[c_accs[i][cid] for i in range(len(c_accs))]
        return outdict

    def iterate(self, t):
        # sample clients
        selected_clients = self.sample()
        # training
        ws, losses = self.communicate(selected_clients)
        # aggregate: pk = nk/n as default
        self.model = self.aggregate(ws, p=[1.0*self.client_vols[id]/self.data_vol for id in selected_clients])
        # output info
        return selected_clients

    def communicate(self, cids):
        cpkgs = []
        if self.num_threads <= 1:
            # computing iteratively
            for cid in cids:
                rp = self.communicate_with(cid)
                cpkgs.append(rp)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(cids)))
            cpkgs = pool.map(self.communicate_with, cids)
            pool.close()
            pool.join()
        return self.unpack(cpkgs)

    def communicate_with(self, cid):
        svr_pkg = self.pack(cid)
        return self.clients[cid].reply(svr_pkg)

    def pack(self, cid):
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, cpkgs):
        ws = [cp["model"] for cp in cpkgs]
        losses = [cp["train_loss"] for cp in cpkgs]
        return ws, losses

    def global_lr_scheduler(self, round):
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{t+1} = DecayRate * eta_{t}"""
            self.lr = float(self.lr * np.power(self.decay_rate, self.option['num_epochs']))

    def sample(self, replacement=False):
        cids = [i for i in range(self.num_clients)]
        selected_cids = []
        if self.sample_option == 'uniform': # original sample
            selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=False))
        elif self.sample_option =='md': # default
            selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=False, p=[nk / self.data_vol for nk in self.client_vols]))
            # selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=True, p=[nk/self.data_vol for nk in self.client_vols]))
        # client dropout
        selected_cids = [cid for cid in selected_cids if self.clients[cid].is_available()]
        return selected_cids

    def aggregate(self, ws, p=[]):
        if not ws: return self.model
        """
        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==============================================================================================|============================
        N/K * Σpk * wk                 |1/K * Σwk                  |(1-Σpk) * w_old + Σpk * wk     |Σ(pk/Σpk) * wk
        """
        if self.agg_option == 'weighted_scale':
            K = len(ws)
            N = self.num_clients
            return fmodule.sum([wk*pk for wk,pk in zip(ws, p)])*N/K
        elif self.agg_option == 'uniform':
            return fmodule.average(ws)
        elif self.agg_option == 'weighted_com':
            w = fmodule.sum([wk*pk for wk,pk in zip(ws, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule.sum([wk * pk for wk, pk in zip(ws, p)])

    def test_on_clients(self, round, dataflag='valid'):
        """ Validate accuracies and losses """
        accs, losses = [], []
        for c in self.clients:
            acc, loss = c.test(self.model, dataflag)
            accs.append(acc)
            losses.append(loss)
        return accs, losses

    def test_on_dtest(self):
        if self.dtest:
            return fmodule.test(self.model, self.dtest)
        else: return -1,-1

class BaseClient():
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, train_rate = 0.8, drop_rate = -1):
        self.name = name
        self.frequency = 0
        # client's benchmark
        if train_rate==0:
            self.train_data = fmodule.XYDataset(data_train_dict['x'], data_train_dict['y'])
            self.valid_data = fmodule.XYDataset(data_val_dict['x'], data_val_dict['y'])
        else:
            data_x = data_train_dict['x'] + data_val_dict['x']
            data_y = data_train_dict['y'] + data_val_dict['y']
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
            if train_rate==1:
                self.train_data = fmodule.XYDataset(data_x, data_y)
                self.valid_data = self.train_data
            else:
                k = int(len(data_x) * train_rate)
                self.train_data = fmodule.XYDataset(data_x[:k], data_y[:k])
                self.valid_data = fmodule.XYDataset(data_x[k:], data_y[k:])
        self.datavol = len(self.train_data)
        self.drop_rate = drop_rate if drop_rate>0.01 else 0
        # hyper-parameters for training
        self.optimizer = option['optimizer']
        self.epochs = option['num_epochs']
        self.learning_rate = option['learning_rate']
        self.batch_size = option['batch_size']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.model = None

    def setModel(self, model):
        self.model = model

    def set_learning_rate(self, lr = 0):
        self.learning_rate = lr if lr else self.learning_rate

    def is_available(self):
        if self.drop_rate==0: return True
        else: return (np.random.rand() < self.drop_rate)

    def train(self, model):
        model.train()
        if self.batch_size == -1:
            # full gradient descent
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = fmodule.get_optimizer(self.optimizer, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (features, labels) in enumerate(ldr_train):
                features, labels = features.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(features)
                loss = lossfunc(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def test(self, model, dataflag='valid'):
        if dataflag == 'valid' and self.valid_data:
            return fmodule.test(model, self.valid_data)
        elif dataflag == 'train' and self.train_data:
            return fmodule.test(model, self.train_data)
        else: return -1, -1

    def train_loss(self, model):
        return self.test(model,'train')[1]

    def valid_loss(self, model):
        return self.test(model)[1]

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model):
        loss = self.train_loss(model)
        return {
            "model" : model,
            "train_loss": loss,
        }
