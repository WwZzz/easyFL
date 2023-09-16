import math
import copy
import collections
from typing import Any
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import numpy as np

import flgo.benchmark.base
from flgo.utils import fmodule
import flgo.simulator.base as ss

# The BasicParty
class BasicParty:
    def __init__(self, *args, **kwargs):
        self.actions = {}  # the message-action map that is used to customize the communication process
        self.id = None  # the id for communicating
        self._object_map = {} # mapping objects according to their ids
        self._data_names = []

    def register_action_to_mtype(self, action_name: str, mtype):
        r"""
        Register an existing method as the action corresponding to the message type.

        Args:
            action_name: the name of the instance method
            mtype: the message type
        """
        if action_name not in self.__dict__.keys():
            raise NotImplementedError("There is no method named `{}` in the class instance.".format(action_name))
        self.actions[mtype] = self.__dict__[action_name]

    def message_handler(self, package):
        r"""
        Handling the received message by excuting the corresponding action.

        Args:
            package (dict): the package received from other parties (i.e. the content of the message)

        Returns:
            action_reult
        """
        try:
            mtype = package['__mtype__']
        except:
            raise KeyError("__mtype__ must be a key of the package")
        if mtype not in self.actions.keys():
            raise NotImplementedError("There is no action corresponding to message type {}.".format(mtype))
        return self.actions[mtype](package)

    def set_data(self, data, flag:str='train') -> None:
        r"""
        Set self's attibute 'xxx_data' to be data where xxx is the flag. For example,
        after calling self.set_data([1,2,3], 'test'), self.test_data will be [1,2,3].
        Particularly, If the flag is 'train', the batchsize and the num_steps will be
        reset.

        Args:
            data: anything
            flag (str): the name of the data
        """
        setattr(self, flag + '_data', data)
        if flag not in self._data_names:
            self._data_names.append(flag)
        if flag == 'train':
            self.datavol = len(data)
            if hasattr(self, 'batch_size'):
                # reset batch_size
                if self.batch_size < 0:
                    self.batch_size = len(self.get_data(flag))
                elif self.batch_size >= 1:
                    self.batch_size = int(self.batch_size)
                else:
                    self.batch_size = int(self.datavol * self.batch_size)
            # reset num_steps
            if hasattr(self, 'num_steps') and hasattr(self, 'num_epochs'):
                if self.num_steps > 0:
                    self.num_epochs = 1.0 * self.num_steps / (math.ceil(self.datavol / self.batch_size))
                else:
                    self.num_steps = self.num_epochs * math.ceil(self.datavol / self.batch_size)

    def get_data(self, flag:str='val')->Any:
        r"""
        Get self's attibute '{flag}_data' if this attribute exists.

        Args:
            flag (str): the name of the data

        Returns:
            flag_data (Any): self.{flag}_data
        """
        dname = (flag+'_data')
        if flag not in self._data_names: return None
        else:
            return getattr(self, dname)

    def get_data_names(self)->list:
        """
        Get the names of data hold by self.

        Returns:
            data_names (list): the names of data hold by self
        """
        return self._data_names

    def get_classname(self)->str:
        """
        Get the class name of self.

        Returns:
            class_name (str): the class name
        """
        return self.__class__.__name__

    def set_model(self, model, model_name: str = 'model'):
        r"""
        Set self's attibute 'model_name' to be model. For example,
        after calling self.set_model(my_model, 'model'), self.model will be my_model.
        """
        # set self.__dict__[model_name] = model
        setattr(self, model_name, model)

    def set_id(self, id=None):
        r"""
        Set self's attibute 'id' to be id where self.id = id
        """
        if id is not None:
            self.id = id

    def set_message(self, mtype:Any, package:dict={})->dict:
        """Set the message type of a package.

        Args:
            mtype (Any): the message type
            package (dict): a dict

        Returns:
            package_with_mtype (dict): a dict with the message type
        """
        if type(package) is not dict:
            raise TypeError('The type of the package should be dict')
        package.update({'__mtype__': mtype})
        return package

    def register_objects(self, parties:list, parties_name='parties'):
        r"""
        Set self's attribute party_names (e.g. parties as default) to be parties if
        self has no attribute named party_names. Otherwise, parties will be extend to
        the attribute party_names of self.
        
        Args:
            parties (list): a list of objects
            parties_name (str): the name of attribute to store parties

        Example:
        ```python
            >>> a = BasicParty()
            >>> b = BasicParty()
            >>> c = BasicParty()
            >>> a.register_objects([b, c], 'parties')
            >>> a.parties # will be [b,c]
            >>> d = BasicParty()
            >>> a.register_objects([d], 'parties')
            >>> a.parties # will be [b,c,d]
        ```
        """
        if type(parties) is not list:
            raise TypeError("parties should be a list")
        if not hasattr(self, parties_name):
            setattr(self, parties_name, parties)
        else:
            tmp = getattr(self, parties_name)
            if tmp is None: tmp = []
            elif type(tmp) is not list: tmp = list(tmp)
            tmp.extend(parties)
            setattr(self, parties_name, tmp)
        self._object_map.update({p.id:p for p in parties if p.id is not None})

    def communicate_with(self, target_id, package={}):
        r"""
        Send the package to target object according to its id, and receive the response from it

        Args:
            target_id (int): the id of the object to communicate with
            package (dict): the package to be sended to the object
        Returns:
            client_package (dict): the reply from the target object and will be 'None' if losing connection
        """
        return self.gv.communicator.request(self.id, target_id, package)

    def initialize(self, *args, **kwargs):
        r"""API for customizing the initializing process of the object"""
        return

class BasicServer(BasicParty):
    TaskCalculator = flgo.benchmark.base.BasicTaskCalculator
    def __init__(self, option={}):
        super().__init__()
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
        self.clients = []
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # server calculator
        self.device = self.gv.apply_for_device() if not option['server_with_cpu'] else torch.device('cpu')
        self.calculator = self.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.proportion = option['proportion']
        self.decay_rate = option['learning_rate_decay']
        self.lr_scheduler_type = option['lr_scheduler']
        self.lr = option['learning_rate']
        self.sample_option = option['sample']
        self.aggregation_option = option['aggregate']
        # systemic option
        self.tolerance_for_latency = 999999
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option
        self.id = -1

    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        if self.eval_interval>0:
            # evaluating initial model performance
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            self.gv.clock.step()
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
                # decay learning rate
                self.global_lr_scheduler(self.current_round)
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        """
        The standard iteration of each federated communication round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.

        Returns:
            False if the global model is not updated in this iteration
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return len(models) > 0

    @ss.with_clock
    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.

        Args:
            selected_clients (list of int): the clients to communicate with
            mtype (anytype): type of message
            asynchronous (bool): asynchronous communciation or synchronous communcation

        Returns:
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = list(set(selected_clients))
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in tqdm(communicate_clients, desc="Local Training on {} Clients".format(len(communicate_clients)), leave=False):
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
                packages_received_from_clients.append(response_from_client_id)
        else:
            self.model = self.model.to(torch.device('cpu'))
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                server_pkg = self.pack(client_id, mtype)
                server_pkg['__mtype__'] = mtype
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (self.clients[client_id].id, server_pkg)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
            self.model = self.model.to(self.device)
            for pkg in packages_received_from_clients:
                for k,v in pkg.items():
                    if hasattr(v, 'to'):
                        try:
                            pkg[k] = v.to(self.device)
                        except:
                            continue
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, target_id, package={}):
        r"""Communicate with the object under system simulator that simulates the
        network latency. Send the package to target object according to its id,
        and receive the response from it

        Args:
            target_id (int): the id of the object to communicate with
            package (dict): the package to be sended to the object

        Returns:
            client_package (dict): the reply from the target object and
            will be 'None' if losing connection
        """
        return super(BasicServer, self).communicate_with(target_id, package)

    def pack(self, client_id, mtype=0, *args, **kwargs):
        r"""
        Pack the necessary information for the client's local_movielens_recommendation training.
        Any operations of compression or encryption should be done here.

        Args:
            client_id (int): the id of the client to communicate with
            mtype: the message type

        Returns:
            a dict contains necessary information (e.g. a copy of the global model as default)
        """
        return {
            "model": copy.deepcopy(self.model),
        }

    def unpack(self, packages_received_from_clients):
        r"""
        Unpack the information from the received packages. Return models and losses as default.

        Args:
            packages_received_from_clients (list): a list of packages

        Returns:
            res (dict): collections.defaultdict that contains several lists of the clients' reply
        """
        if len(packages_received_from_clients) == 0: return collections.defaultdict(list)
        res = {pname: [] for pname in packages_received_from_clients[0]}
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res

    def global_lr_scheduler(self, current_round):
        r"""
        Control the step size (i.e. learning rate) of local_movielens_recommendation training
        Args:
            current_round (int): the current communication round
        """
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr *= self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{round+1} = eta_0/(round+1)"""
            self.lr = self.option['learning_rate'] * 1.0 / (current_round + 1)
            for c in self.clients:
                c.set_learning_rate(self.lr)

    def sample(self):
        r"""
        Sample the clients. There are three types of sampling manners:
        full sample, uniform sample without replacement, and MDSample
        with replacement. Particularly, if 'available' is in self.sample_option,
        the server will only sample from currently available clients.

        Returns:
            a list of the ids of the selected clients

        Example:
        ```python
            >>> selected_clients=self.sample()
            >>> selected_clients
            >>> # The selected_clients is a list of clients' ids
        ```
        """
        all_clients = self.available_clients if 'available' in self.sample_option else [cid for cid in
                                                                                        range(self.num_clients)]
        # full sampling with unlimited communication resources of the server
        if 'full' in self.sample_option:
            return all_clients
        # sample clients
        elif 'uniform' in self.sample_option:
            # original sample proposed by fedavg
            selected_clients = list(
                np.random.choice(all_clients, min(self.clients_per_round, len(all_clients)), replace=False)) if len(
                all_clients) > 0 else []
        elif 'md' in self.sample_option:
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local_movielens_recommendation data sizes
            local_data_vols = [self.clients[cid].datavol for cid in all_clients]
            total_data_vol = sum(local_data_vols)
            p = np.array(local_data_vols) / total_data_vol
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=p)) if len(
                all_clients) > 0 else []
        return selected_clients

    def aggregate(self, models: list, *args, **kwargs):
        r"""
        Aggregate the locally trained models into the new one. The aggregation
        will be according to self.aggregate_option where

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k


        Args:
            models (list): a list of local_movielens_recommendation models

        Returns:
            the aggregated model

        Example:
        ```python
            >>> models = [m1, m2] # m1, m2 are models with the same architecture
            >>> m_new = self.aggregate(models)
        ```
        """
        if len(models) == 0: return self.model
        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")
            self.gv.logger.info('Warning("There exists nan-value in local models, which will be automatically removed from the aggregatino list.")')
            new_models = []
            received_clients = []
            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0 - sum(p)) * self.model + w
        else:
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            sump = sum(p)
            p = [pk / sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def global_test(self, model=None, flag:str='val'):
        r"""
        Collect local_movielens_recommendation testing result of all the clients.

        Args:
            model (flgo.utils.fmodule.FModule|torch.nn.Module): the model to be sevaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metrics (dict): a dict contains key-value pairs like (metric_name,
            the lists of metric results of the clients)
        """
        if model is None: model=self.model
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(model, flag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None, flag:str='test'):
        r"""
        Evaluate the model on the test dataset owned by the server.

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metrics (dict): the dict contains the evaluating results
        """
        if model is None: model = self.model
        dataset = getattr(self, flag+'_data') if hasattr(self, flag+'_data') else None
        if dataset is None:
            return {}
        else:
            return self.calculator.test(model, dataset, batch_size=min(self.option['test_batch_size'], len(dataset)),
                                        num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])

    def init_algo_para(self, algo_para: dict):
        """
        Initialize the algorithm-dependent hyper-parameters for the server and all the clients.

        Args:
            algo_paras (dict): the dict that defines the hyper-parameters (i.e. name, value and type) for the algorithm.

        Example:
        ```python
            >>> # s is an instance of Server and s.clients are instances of Client
            >>> s.u # will raise error
            >>> [c.u for c in s.clients] # will raise errors too
            >>> s.init_algo_para({'u': 0.1})
            >>> s.u # will be 0.1
            >>> [c.u for c in s.clients] # will be [0.1, 0.1,..., 0.1]
        ```
        Note:
            Once `option['algo_para']` is not `None`, the value of the pre-defined hyperparameters will be replaced by the list of values in `option['algo_para']`,
            which requires the length of `option['algo_para']` is equal to the length of `algo_paras`
        """
        self.algo_para = algo_para
        if len(self.algo_para) == 0: return
        # initialize algorithm-dependent hyperparameters from the input options
        if self.option['algo_para'] is not None:
            # assert len(self.algo_para) == len(self.option['algo_para'])
            keys = list(self.algo_para.keys())
            for i, pv in enumerate(self.option['algo_para']):
                if i == len(self.option['algo_para']): break
                para_name = keys[i]
                try:
                    self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
                except:
                    self.algo_para[para_name] = pv
        # register the algorithm-dependent hyperparameters as the attributes of the server and all the clients
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
            for p in self._object_map.values():
                p.__setattr__(para_name, value)
        return

    def get_tolerance_for_latency(self):
        r"""
        Get the tolerance for latency of waiting for clients' responses

        Returns:
            a int number (i.e. self.tolerance_for_latency)
        """
        return self.tolerance_for_latency

    def set_tolerance_for_latency(self, tolerance:int):
        r"""
        Set the tolerance for latency of waiting for clients' responses

        Args:
            tolerance (int): the amounts of virtual time units
        """
        self.tolerance_for_latency = tolerance

    def wait_time(self, t=1):
        r"""
        Wait for the time of the virtual clock to pass t units
        """
        ss.clock.step(t)
        return

    @property
    def available_clients(self):
        """
        Return all the available clients at the current round.

        Returns:
            a list of indices of currently available clients
        """
        return [cid for cid in range(self.num_clients) if self.clients[cid].is_idle()]

    def register_clients(self, clients):
        """
        Regiser clients to self.clients, and update related attributes (e.g. self.num_clients)
        
        Args:
            clients (list): a list of objects
        """
        self.register_objects(clients, 'clients')
        self.num_clients = len(clients)
        for cid, c in enumerate(self.clients):
            c.client_id = cid
        for c in self.clients: c.register_server(self)
        self.clients_per_round = max(int(self.num_clients * self.proportion), 1)
        self.selected_clients = []
        self.dropped_clients = []

class BasicClient(BasicParty):
    TaskCalculator = flgo.benchmark.base.BasicTaskCalculator
    def __init__(self, option={}):
        super().__init__()
        self.id = None
        # create local_movielens_recommendation dataset
        self.data_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
        # local_movielens_recommendation calculator
        self.device = self.gv.apply_for_device()
        self.calculator = self.TaskCalculator(self.device, option['optimizer'])
        self._train_loader = None
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.clip_grad = option['clip_grad']
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        # system setting
        self._effective_num_steps = self.num_steps
        self._latency = 0
        # server
        self.server = None
        # actions of different message type
        self.option = option
        self.actions = {0: self.reply}

    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local_movielens_recommendation training procedure. Train the transmitted model with
        local_movielens_recommendation training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, flag='val'):
        r"""
        Evaluate the model on the dataset owned by the client

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metric (dict): the evaluating results (e.g. metric = {'loss':1.02})
        """
        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        return self.calculator.test(model, dataset, min(self.test_batch_size, len(dataset)), self.option['num_workers'])

    def unpack(self, received_pkg):
        r"""
        Unpack the package received from the server

        Args:
            received_pkg (dict): a dict contains the global model as default

        Returns:
            the unpacked information
        """
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        r"""
        Reply a package to the server. The whole local_movielens_recommendation procedure should be defined here.
        The standard form consists of three procedure: unpacking the
        server_package to obtain the global model, training the global model,
        and finally packing the updated model into client_package.

        Args:
            svr_pkg (dict): the package received from the server

        Returns:
            client_pkg (dict): the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model, *args, **kwargs):
        r"""
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.

        Args:
            model: the locally trained model

        Returns:
            package: a dict that contains the necessary information for the server
        """
        return {
            "model": model,
        }

    def is_idle(self):
        r"""
        Check if the client is available to participate training.

        Returns:
            True if the client is available according to the active_rate else False
        """
        return self.gv.simulator.client_states[self.id] == 'idle'

    def is_dropped(self):
        r"""
        Check if the client drops out during communicating.

        Returns:
            True if the client was being dropped
        """
        return self.gv.simulator.client_states[self.id] == 'dropped'

    def is_working(self):
        r"""
        Check if the client is training the model.

        Returns:
            True if the client is working
        """

        return self.gv.simulator.client_states[self.id] == 'working'

    def train_loss(self, model):
        r"""
        Get the loss value of the model on local_movielens_recommendation training data

        Args:
            model (flgo.utils.fmodule.FModule|torch.nn.Module): model

        Returns:
            the training loss of model on self's training data
        """
        return self.test(model, 'train')['loss']

    def val_loss(self, model):
        r"""
        Get the loss value of the model on local_movielens_recommendation validating data

        Args:
            model (flgo.utils.fmodule.FModule|torch.nn.Module): model

        Returns:
            the validation loss of model on self's validation data
        """
        return self.test(model)['loss']

    def register_server(self, server=None):
        r"""
        Register the server to self.server
        """
        self.register_objects([server], 'server_list')
        if server is not None:
            self.server = server

    def set_local_epochs(self, epochs=None):
        r"""
        Set local_movielens_recommendation training epochs
        """
        if epochs is None: return
        self.epochs = epochs
        self.num_steps = self.epochs * math.ceil(len(self.train_data) / self.batch_size)
        return

    def set_batch_size(self, batch_size=None):
        r"""
        Set local_movielens_recommendation training batch size

        Args:
            batch_size (int): the training batch size
        """
        if batch_size is None: return
        self.batch_size = batch_size

    def set_learning_rate(self, lr=None):
        """
        Set the learning rate of local_movielens_recommendation training
        Args:
            lr (float): a real number
        """
        self.learning_rate = lr if lr else self.learning_rate

    def get_time_response(self):
        """
        Get the latency amount of the client

        Returns:
            self.latency_amount if client not dropping out
        """
        return np.inf if self.dropped else self.time_response

    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'])
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self._train_loader)
            batch_data = next(self.data_loader)
        # clear local_movielens_recommendation DataLoader when finishing local_movielens_recommendation training
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def update_device(self, dev):
        """
        Update running-time GPU device to dev

        Args:
            dev (torch.device): target dev
        """
        self.device = dev
        self.calculator = self.gv.TaskCalculator(dev, self.calculator.optimizer_name)
