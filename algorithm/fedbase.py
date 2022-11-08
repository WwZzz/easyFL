import numpy as np
import utils.fmodule
from utils import fmodule
import copy
import os
import utils.fflow as flw
import utils.system_simulator as ss
import math
import collections
import torch.multiprocessing as mp

class BasicServer:
    def __init__(self, option, model, clients, test_data=None):
        # basic configuration
        self.task = option['task']
        self.name = option['algorithm']
        self.model = model
        self.device = self.model.get_device()
        self.test_data = test_data
        self.eval_interval = option['eval_interval']
        self.num_threads = option['num_threads']
        # server calculator
        self.calculator = fmodule.TaskCalculator(self.device, optimizer_name = option['optimizer'])
        # clients configuration
        self.clients = clients
        self.num_clients = len(self.clients)
        self.local_data_vols = [c.datavol for c in self.clients]
        self.total_data_vol = sum(self.local_data_vols)
        for cid, c in enumerate(self.clients): c.client_id = cid
        self.selected_clients = []
        self.dropped_clients = []
        for c in self.clients:c.set_server(self)
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.decay_rate = option['learning_rate_decay']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.sample_option = option['sample']
        self.aggregation_option = option['aggregate']
        # systemic option
        self.tolerance_for_latency = 1000
        self.tolerance_for_availability = 0
        self.asynchronous = False
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            if flw.logger.check_if_log(round, self.eval_interval):
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()
                flw.logger.time_end('Eval Time Cost')
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        return

    @ss.time_step
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
        return

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
        client_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        for cid in communicate_clients:client_package_buffer[cid] = None
        if self.num_threads <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_threads)
            for client_id in communicate_clients:
                self.clients[client_id].update_device(next(utils.fmodule.dev_manager))
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=(int(client_id),)))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): client_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [client_package_buffer[cid] for cid in selected_clients if client_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    @ss.with_latency
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
        all_clients = [cid for cid in range(self.num_clients)]
        # full sampling with unlimited communication resources of the server
        if self.sample_option == 'full':
            return all_clients
        # sample clients
        elif self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local data sizes
            p = np.array(self.local_data_vols)/self.total_data_vol
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
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.received_clients]
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def test_on_clients(self, dataflag='valid'):
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

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.test(model, self.test_data, batch_size = self.option['test_batch_size'])
        else:
            return None

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
                self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
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
        return [cid for cid in range(self.num_clients) if self.clients[cid].is_available()]

class BasicClient():
    def __init__(self, option, name='', train_data=None, valid_data=None):
        self.name = name
        self.id = None
        # create local dataset
        self.train_data = train_data
        self.valid_data = valid_data
        if option['train_on_all']:
            self.train_data = self.train_data + self.valid_data
        self.datavol = len(self.train_data)
        self.data_loader = None
        # local calculator
        self.device = next(fmodule.dev_manager)
        self.calculator = fmodule.TaskCalculator(self.device, option['optimizer'])
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.batch_size = len(self.train_data) if option['batch_size']<0 else option['batch_size']
        self.batch_size = int(self.batch_size) if self.batch_size>=1 else int(len(self.train_data)*self.batch_size)
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        if option['num_steps']>0:
            self.num_steps = option['num_steps']
            self.epochs = 1.0 * self.num_steps/(math.ceil(len(self.train_data)/self.batch_size))
        else:
            self.epochs = option['num_epochs']
            self.num_steps = self.epochs * math.ceil(len(self.train_data) / self.batch_size)
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        # system setting
        # 1) availability
        self.available = True
        # 2) connectivity
        self.dropped = False
        # 3) completeness
        self._effective_num_steps = self.num_steps
        # 4) timeliness
        self._latency = 0
        # server
        self.server = None

    @ss.with_completeness
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
            loss = self.calculator.train_one_step(model, batch_data)['loss']
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
        return self.calculator.test(model, dataset, self.test_batch_size)

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

    def is_available(self):
        """
        Check if the client is active to participate training.
        :param
        :return
            True if the client is active according to the active_rate else False
        """
        return self.available

    def is_dropped(self):
        """
        Check if the client drops out during communicating.
        :param
        :return
            True if the client drops out according to the drop_rate else False
        """
        return self.dropped

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

    def set_server(self, server=None):
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
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, num_workers=self.loader_num_workers))
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
        self.calculator = fmodule.TaskCalculator(dev, self.calculator.optimizer_name)