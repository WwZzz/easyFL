import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
import time

class BasicServer():
    def __init__(self, option, model, clients, test_data = None):
        # basic setting
        self.task = option['task']
        self.name = option['algorithm']
        self.model = model
        self.test_data = test_data
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
        # server calculator
        self.calculator = fmodule.TaskCalculator(fmodule.device)

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
                test_acc, test_loss = self.test()
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
        models, train_losses = self.communicate(selected_clients)
        # aggregate: pk = nk/n as default
        self.model = self.aggregate(models, p=[1.0*self.client_vols[id]/self.data_vol for id in selected_clients])
        # output info
        return selected_clients

    def communicate(self, client_ids):
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in client_ids:
                rp = self.communicate_with(client_id)
                packages_received_from_clients.append(rp)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(client_ids)))
            packages_received_from_clients = pool.map(self.communicate_with, client_ids)
            pool.close()
            pool.join()
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        svr_pkg = self.pack(client_id)
        return self.clients[client_id].reply(svr_pkg)

    def pack(self, client_id):
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return models, train_losses

    def global_lr_scheduler(self, current_round):
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)

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
            return fmodule._model_sum([wk * pk for wk, pk in zip(ws, p)]) * N / K
        elif self.agg_option == 'uniform':
            return fmodule._model_average(ws)
        elif self.agg_option == 'weighted_com':
            w = fmodule._model_sum([wk * pk for wk, pk in zip(ws, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([wk * pk for wk, pk in zip(ws, p)])

    def test_on_clients(self, round, dataflag='valid'):
        """ Validate accuracies and losses """
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(self.model, dataflag)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

    def test(self):
        if self.test_data:
            self.model.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(self.model, batch_data)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            return eval_metric, loss
        else: return -1,-1

class BasicClient():
    def __init__(self, option, name='', train_data=None, valid_data=None, drop_rate=-1):
        self.name = name
        self.frequency = 0
        # create local dataset
        self.train_data = train_data
        self.valid_data = valid_data
        self.datavol = len(self.train_data)

        # system setting
        self.drop_rate = drop_rate if drop_rate>0.01 else 0

        # local calculator
        self.calculator = fmodule.TaskCalculator(device=fmodule.device)
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.epochs = option['num_epochs']
        self.learning_rate = option['learning_rate']
        self.batch_size = len(self.train_data) if option['batch_size']==-1 else option['batch_size']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.model = None

    def set_model(self, model):
        self.model = model

    def set_learning_rate(self, lr = 0):
        self.learning_rate = lr if lr else self.learning_rate

    def is_available(self):
        if self.drop_rate==0: return True
        else: return (np.random.rand() < self.drop_rate)

    def train(self, model):
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        return

    def test(self, model, dataflag='valid'):
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def train_loss(self, model):
        return self.test(model,'train')[1]

    def valid_loss(self, model):
        return self.test(model)[1]

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg

    def pack(self, model, loss):
        return {
            "model" : model,
            "train_loss": loss,
        }
