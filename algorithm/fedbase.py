import numpy as np
from utils import fmodule
import copy
from multiprocessing import Pool as ThreadPool
from main import logger
import os
import utils.fflow as flw
import utils.network_simulator as ns

class BasicServer:
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
        self.selected_clients = []
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
        # virtual clock
        self._max_waiting_time = 10
        self.virtual_clock = []

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)
            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)
            logger.time_end('Time Cost')
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))
        return

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        models, train_losses = self.communicate(self.selected_clients)
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return

    @ns.under_network_latency
    def communicate(self, selected_clients):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in selected_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(selected_clients)))
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
            pool.close()
            pool.join()
        self.selected_clients = [selected_clients[k] for k in range(len(selected_clients)) if packages_received_from_clients[k]]
        packages_received_from_clients = [pk for pk in packages_received_from_clients if pk]
        return self.unpack(packages_received_from_clients)

    @ns.drop_client
    def communicate_with(self, client_id):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response
        return self.clients[client_id].reply(svr_pkg)

    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return: a list of local computing results [models, losses, ...] where
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        res = []
        if len(packages_received_from_clients)==0: return res
        for key in packages_received_from_clients[0]:
            res.append([cp[key] for cp in packages_received_from_clients])
        return res

    def global_lr_scheduler(self, current_round):
        """
        Control the step size (i.e. learning rate) of local training
        :param
            current_round: the current communication round
        """
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{round+1} = eta_0/(round+1)"""
            self.lr = self.option['learning_rate']*1.0/(current_round+1)
            for c in self.clients:
                c.set_learning_rate(self.lr)

    @ns.deactivate_client
    def sample(self):
        """Sample the clients.
        :param
        :return
            a list of the ids of the selected clients
        """
        all_clients = [cid for cid in range(self.num_clients)]
        # sample clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        return selected_clients

    def aggregate(self, models, p=[]):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
            p: a list of weights for aggregating
        :return
            the averaged result

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k
        """
        if not models: return self.model
        if self.agg_option == 'weighted_scale':
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.agg_option == 'uniform':
            return fmodule._model_average(models)
        elif self.agg_option == 'weighted_com':
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def test_on_clients(self, round, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            round: the current communication round
            dataflag: choose train data or valid data to evaluate
        :return
            evals: the evaluation metrics of the global model on each client's dataset
            loss: the loss of the global model on each client's dataset
        """
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss = c.test(self.model, dataflag)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: model=self.model
        if self.test_data:
            model.eval()
            loss = 0
            eval_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
                loss += bmean_loss * len(batch_data[0])
                eval_metric += bmean_eval_metric * len(batch_data[0])
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            return eval_metric, loss
        else: return -1,-1

class BasicClient():
    def __init__(self, option, name='', train_data=None, valid_data=None):
        self.name = name
        self.frequency = 0
        # create local dataset
        self.train_data = train_data
        self.valid_data = valid_data
        self.datavol = len(self.train_data)
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
        # system setting
        # the probability of dropout obey distribution beta(drop, 1). The larger 'drop' is, the more possible for a device to drop
        if option['net_active']>99998:
            self.active_rate = 1
        elif option['net_active']>=100:
            rho = 1.0 * option['net_active']/1000
            miny = min(self.train_data.get_all_labels())
            self.active_rate = (1-rho)*miny/9.0+rho
        else:
            self.active_rate =  np.random.beta(option['net_active'], 1, 1).item()
        if option['net_drop'] >=100:
            rho = 1.0 * option['net_drop'] / 1000
            miny = min(self.train_data.get_all_labels())
            self.drop_rate = (1-rho)*miny/9.0 + rho
        elif option['net_drop'] <=0.01:
            self.drop_rate=0
        else:
            self.drop_rate =np.random.beta(option['net_drop'], 1, 1).item()

        self.latency_amount = np.random.normal(0, option['net_latency'], 1)
        self.latency_amount = 1.0/(1.0-self.latency_amount) if self.latency_amount<0 else 1.0*(1.0+self.latency_amount)

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
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
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            loss += bmean_loss * len(batch_data[0])
            eval_metric += bmean_eval_metric * len(batch_data[0])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg

    def pack(self, model, loss):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "train_loss": loss,
        }

    def is_active(self):
        """
        Check if the client is active to participate training.
        :param
        :return
            True if the client is active according to the active_rate else False
        """
        if self.active_rate==1: return True
        else: return (np.random.rand() <= self.active_rate)

    def is_drop(self):
        """
        Check if the client drops out during communicating.
        :param
        :return
            True if the client drops out according to the drop_rate else False
        """
        if self.drop_rate==0: return False
        else: return (np.random.rand() < self.drop_rate)

    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[1]

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[1]

    def set_model(self, model):
        """
        set self.model
        :param model:
        :return:
        """
        self.model = model

    def set_learning_rate(self, lr = None):
        """
        set the learning rate of local training
        :param lr:
        :return:
        """
        self.learning_rate = lr if lr else self.learning_rate

    def get_network_latency(self):
        """
        Get the latency amount of the client
        :return: self.latency_amount
        """
        return self.latency_amount