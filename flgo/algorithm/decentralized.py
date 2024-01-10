from flgo.algorithm.fedbase import BasicParty
import collections
from abc import ABCMeta, abstractmethod
from flgo.utils import fmodule
import copy
import torch

class AbstractProtocol(metaclass=ABCMeta):
    r"""
    Abstract Protocol
    """
    @abstractmethod
    def get_clients_for_iteration(self, *args, **kwarg):
        """Return clients that should perform iteration at the current moment"""
        pass

class BasicProtocol(AbstractProtocol, BasicParty):
    def __init__(self, option):
        super().__init__()
        self.test_data = None
        self.val_data = None
        self.model = None
        self.clients = []
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # server calculator
        self.device = self.gv.apply_for_device() if not option['server_with_cpu'] else torch.device('cpu')
        self.calculator = self.gv.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        # algorithm-dependent parameters
        self.algo_para = {}
        self.current_round = 1
        # all options
        self.option = option
        self.id = -1

    def get_clients_for_iteration(self, *args, **kwarg):
        return self.clients

    def run(self):
        self.gv.logger.time_start('Total Time Cost')
        if self.eval_interval > 0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        for c in self.clients:
            c.init_state()
        while self.current_round <= self.num_rounds:
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
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        clients_for_iteration = self.get_clients_for_iteration()
        # compute next state for each client in Sc
        for c in clients_for_iteration:
            c.nstate = c.update_state()
        # flush the states of all the clients
        for c in clients_for_iteration:
            c.state = copy.deepcopy(c.nstate)
        return

class LineProtocol(BasicProtocol):
    def __init__(self, option):
        super(LineProtocol, self).__init__(option)
        self.crt_client = 0
        self.num_rounds = min(len(self.clients), self.num_rounds)

    def get_clients_for_iteration(self, *args, **kwarg):
        ret = [self.clients[self.crt_client]]
        self.crt_client += 1
        return ret

class RingProtocol(BasicProtocol):
    def __init__(self, option):
        super(RingProtocol, self).__init__(option)
        self.crt_client = 0

    def get_clients_for_iteration(self, *args, **kwarg):
        ret = [self.clients[self.crt_client]]
        self.crt_client = (self.crt_client+1)%len(self.clients)
        return ret

class BasicClient(BasicParty):
    def __init__(self, option):
        super().__init__(option)
        self.clients = None
        self.topology = 'mesh'
        self.model = None
        self.state = {}
        self.pstate = {}
        self.nstate = {}
        self.id = None
        self.data_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.device = self.gv.apply_for_device()
        self.calculator = self.gv.TaskCalculator(self.device, option['optimizer'])
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
        # actions of different message type
        self.option = option
        self.actions = {0: self.send_state}

    def init_state(self):
        # self.state will be shared with other clients and any direct change on self.state should not impact self.model
        self.state = copy.deepcopy({'model': self.model})

    def update_state(self):
        # sample clients
        selected_clients = self.sample()
        # request prior states from selected neighbors
        received_models = self.communicate(selected_clients)['model']
        # aggregate neighbors' states
        self.model = self.aggregate(received_models)
        # local train
        self.train(self.model)
        return {'model': self.model}

    def sample(self):
        if self.topology=='mesh':
            return list(range(len(self.clients)))
        elif self.topology=='line' or self.topology=='ring':
            return [0]
        else:
            raise NotImplementedError('Not Supported topology')

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        packages_received_from_clients = []
        received_package_buffer = {}
        communicate_clients = selected_clients
        # prepare packages for clients
        for client_id in communicate_clients:
            received_package_buffer[client_id] = None
        # communicate with selected clients
        for client_id in communicate_clients:
            server_pkg = self.pack(client_id, mtype)
            server_pkg['__mtype__'] = mtype
            response_from_client_id = self.communicate_with(self.clients[client_id].id, package=server_pkg)
            packages_received_from_clients.append(response_from_client_id)
        for i, client_id in enumerate(communicate_clients): received_package_buffer[client_id] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if
                                          received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, target_id, package={}):
        return self.gv.communicator.request(self.id, target_id, package)

    def aggregate(self, models:list):
        return fmodule._model_average(models+[self.model])

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

    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'], drop_last = not self.option.get('no_drop_last', False))
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

    def pack(self, *args, **kwargs):
        return {}

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

    def send_state(self, *args, **kwargs):
        return copy.deepcopy(self.state)

    def test(self, model=None, flag='val'):
        r"""
        Evaluate the model on the dataset owned by the client

        Args:
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
            flag (str): choose the data to evaluate the model

        Returns:
            metric (dict): the evaluating results (e.g. metric = {'loss':1.02})
        """
        if model is None: model = self.model
        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        return self.calculator.test(model, dataset, min(self.test_batch_size, len(dataset)), self.option['num_workers'])

Client = BasicClient
Protocol = BasicProtocol