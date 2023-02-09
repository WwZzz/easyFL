import numpy as np
import flgo.utils.fmodule
from flgo.utils import fmodule
import copy
import os
import flgo.system_simulator.base as ss
import math
import collections
import torch.multiprocessing as mp
import config as cfg
import torch

class BasicServer:
    def __init__(self, option={}):
        # initialize the global model
        self.model = cfg.Model()
        if not option['server_with_cpu']:
            self.model = self.model.to(cfg.dev_list[0])
        self.device = self.model.get_device()
        if option['pretrain'] != '':
            self.model.load_state_dict(torch.load(option['pretrain'])['model'])
            cfg.logger.info('The pretrained model parameters in {} will be loaded'.format(option['pretrain']))
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_threads = option['num_threads']
        # server calculator
        self.calculator = cfg.TaskCalculator(self.device, optimizer_name = option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.proportion = option['proportion']
        self.decay_rate = option['learning_rate_decay']
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.sample_option = option['sample']
        self.aggregation_option = option['aggregate']
        # systemic option
        self.tolerance_for_latency = 999999
        self.sending_package_buffer = [None for _ in range(9999)]
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option

    def initialize(self, *args, **kwargs):
        return

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        cfg.logger.time_start('Total Time Cost')
        cfg.logger.info("--------------Initial Evaluation--------------")
        cfg.logger.time_start('Eval Time Cost')
        cfg.logger.log_once()
        cfg.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            cfg.clock.step()
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                cfg.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if cfg.logger.check_if_log(self.current_round, self.eval_interval):
                    cfg.logger.time_start('Eval Time Cost')
                    cfg.logger.log_once()
                    cfg.logger.time_end('Eval Time Cost')
                # check if early stopping
                if cfg.logger.early_stop(): break
                self.current_round += 1
            # decay learning rate
            self.global_lr_scheduler(self.current_round)
            # clear package buffer
            self.sending_package_buffer = [None for _ in self.clients]
        cfg.logger.info("=================End==================")
        cfg.logger.time_end('Total Time Cost')
        # save results as .json file
        cfg.logger.save_output_as_json()
        return

    def iterate(self):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return len(models)>0

    @ss.with_dropout
    @ss.with_clock
    def communicate(self, selected_clients, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for cid in communicate_clients:
            received_package_buffer[cid] = None
        try:
            for cid in communicate_clients:
                self.sending_package_buffer[cid] = self.pack(cid)
        except Exception as e:
            if str(self.device) != 'cpu':
                self.model.to(torch.device('cpu'))
                for cid in communicate_clients:
                    self.sending_package_buffer[cid] = self.pack(cid)
                self.model.to(self.device)
            else:
                raise e
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_threads)
            for client_id in communicate_clients:
                self.clients[client_id].update_device(next(cfg.dev_manager))
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=(int(client_id),)))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): received_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # listen for the client's response
        return self.clients[client_id].reply(self.sending_package_buffer[client_id])

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
        :return:
            res: collections.defaultdict that contains several lists of the clients' reply
        """
        if len(packages_received_from_clients)==0: return collections.defaultdict(list)
        res = {pname:[] for pname in packages_received_from_clients[0]}
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
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

    @ss.with_availability
    def sample(self):
        """Sample the clients.
        :param
        :return
            a list of the ids of the selected clients
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in range(self.num_clients)]
        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients
        # sample clients
        elif 'uniform' in self.sample_option:
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, min(self.clients_per_round, len(all_clients)), replace=False))
        elif 'md' in self.sample_option:
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local data sizes
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols)/total_data_vol
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=p))
        return selected_clients

    def aggregate(self, models: list, *args, **kwargs):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
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
        if len(models) == 0: return self.model
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * local_data_vols[cid] /total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def global_test(self, dataflag='valid'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None, flag='test'):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.model
        data = self.test_data if flag=='test' else self.valid_data
        if data is None: return {}
        else:
            return self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])

    def init_algo_para(self, algo_para: dict):
        """
        Initialize the algorithm-dependent hyper-parameters for the server and all the clients.
        :param
            algo_paras: the dict that defines the hyper-parameters (i.e. name, value and type) for the algorithm.

        Example 1:
            calling `self.init_algo_para({'u':0.1})` will set the attributions `server.u` and `c.u` as 0.1 with type float where `c` is an instance of `CLient`.
        Note:
            Once `option['algo_para']` is not `None`, the value of the pre-defined hyperparameters will be replaced by the list of values in `option['algo_para']`,
            which requires the length of `option['algo_para']` is equal to the length of `algo_paras`
        """
        self.algo_para = algo_para
        if len(self.algo_para)==0: return
        # initialize algorithm-dependent hyperparameters from the input options
        if self.option['algo_para'] is not None:
            # assert len(self.algo_para) == len(self.option['algo_para'])
            keys = list(self.algo_para.keys())
            for i,pv in enumerate(self.option['algo_para']):
                if i==len(self.option['algo_para']): break
                para_name = keys[i]
                try:
                    self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
                except:
                    self.algo_para[para_name] = pv
        # register the algorithm-dependent hyperparameters as the attributes of the server and all the clients
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
            for c in self.clients:
                c.__setattr__(para_name, value)
        return

    def get_tolerance_for_latency(self):
        return self.tolerance_for_latency

    def wait_time(self, t=1):
        ss.clock.step(t)
        return

    @property
    def available_clients(self):
        """
        Return all the available clients at current round.
        :param
        :return: a list of indices of currently available clients
        """
        return [cid for cid in range(self.num_clients) if self.clients[cid].is_idle()]

    def register_clients(self, clients):
        self.clients = clients
        self.num_clients = len(clients)
        for cid, c in enumerate(self.clients):
            c.client_id = cid
        for c in self.clients:c.register_server(self)
        self.clients_per_round = max(int(self.num_clients * self.proportion), 1)
        self.selected_clients = []
        self.dropped_clients = []

    def set_data(self, data, flag='test'):
        setattr(self, flag+'_data', data)

class BasicClient:
    def __init__(self, option={}):
        self.id = None
        # create local dataset
        self.data_loader = None
        # local calculator
        self.device = next(cfg.dev_manager)
        self.calculator = cfg.TaskCalculator(self.device, option['optimizer'])
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        # system setting
        self._effective_num_steps = self.num_steps
        self._latency = 0
        # server
        self.server = None

    def initialize(self):
        return

    @ ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            optimizer.step()
        return

    @ fmodule.with_multi_gpus
    def test(self, model, dataflag='valid'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        if dataset is not None:
            return self.calculator.test(model, dataset, self.test_batch_size)
        else:
            return {}

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
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
        }

    def is_idle(self):
        """
        Check if the client is active to participate training.
        :param
        :return
            True if the client is active according to the active_rate else False
        """
        return cfg.state_updater.client_states[self.id]=='idle'

    def is_dropped(self):
        """
        Check if the client drops out during communicating.
        :param
        :return
            True if the client was being dropped
        """
        return cfg.state_updater.client_states[self.id]=='dropped'

    def is_working(self):
        return cfg.state_updater.client_states[self.id]=='working'

    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')['loss']

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)['loss']

    def set_model(self, model):
        """
        set self.model
        :param model:
        :return:
        """
        self.model = model

    def register_server(self, server=None):
        if server is not None:
            self.server = server

    def set_local_epochs(self, epochs=None):
        if epochs is None: return
        self.epochs = epochs
        self.num_steps = self.epochs * math.ceil(len(self.train_data)/self.batch_size)
        return

    def set_batch_size(self, batch_size=None):
        if batch_size is None: return
        self.batch_size = batch_size

    def set_learning_rate(self, lr = None):
        """
        set the learning rate of local training
        :param lr:
        :return:
        """
        self.learning_rate = lr if lr else self.learning_rate

    def get_time_response(self):
        """
        Get the latency amount of the client
        :return: self.latency_amount if client not dropping out
        """
        return np.inf if self.dropped else self.time_response

    def get_batch_data(self):
        """
        Get the batch of data
        :return:
            a batch of data
        """
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size, num_workers=self.loader_num_workers))
            batch_data = next(self.data_loader)
        # clear local DataLoader when finishing local training
        self.current_steps = (self.current_steps+1) % self.num_steps
        if self.current_steps == 0:self.data_loader = None
        return batch_data

    def update_device(self, dev):
        """
        Update running-time GPU device to the inputted dev, including change the client's device and the task_calculator's device
        :param
            dev: target dev
        :return:
        """
        self.device = dev
        self.calculator = cfg.TaskCalculator(dev, self.calculator.optimizer_name)

    def set_data(self, data, flag='train'):
        setattr(self, flag+'_data', data)
        if flag=='train':
            self.datavol = len(data)
            # reset batch_size
            if self.batch_size<0: self.batch_size = len(self.train_data)
            elif self.batch_size>=1: self.batch_size = int(self.batch_size)
            else: self.batch_size = int(self.datavol * self.batch_size)
            # reset num_steps
            if self.num_steps > 0:
                self.num_epochs = 1.0 * self.num_steps/(math.ceil(self.datavol / self.batch_size))
            else:
                self.num_steps = self.num_epochs * math.ceil(self.datavol / self.batch_size)
