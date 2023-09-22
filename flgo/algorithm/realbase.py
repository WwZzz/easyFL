import collections
import pickle

import torch.cuda
import torch.multiprocessing as mlp
import flgo
import flgo.algorithm.fedavg as fedavg
import zmq
import time
import flgo.utils.fmodule
import threading

class Server(fedavg.Server):
    def __init__(self, option={}):
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
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
        self._data_names = []
        self._exit = False

    def is_exit(self):
        self._lock_exit.acquire()
        res = self._exit
        self._lock_exit.release()
        return res

    def set_exit(self):
        self._lock_exit.acquire()
        self._exit = True
        self._lock_exit.release()
        return

    def add_buffer(self, x):
        self._lock_buffer.acquire()
        self._buffer.put(x)
        self._lock_buffer.release()
        return

    def clear_buffer(self):
        self._lock_buffer.acquire()
        res = {}
        while not self._buffer.empty():
            d = self._buffer.get_nowait()
            res[d['name']] = d['package']
        self._lock_buffer.release()
        return res

    def size_buffer(self):
        self._lock_buffer.acquire()
        buffer_size = len(self._buffer)
        self._lock_buffer.release()
        return buffer_size

    def register(self):
        self.logger.info("Waiting for registrations...")
        self.register_poller = zmq.Poller()
        self.register_poller.register(self.registrar, zmq.POLLIN)
        self.listen_registration_thread = threading.Thread(target=self.listen_for_registration, )
        self.listen_registration_thread.start()
        while True:
            time.sleep(1)
            if self.if_start():
                self.logger.info("Start training...")
                break
        return

    def if_start(self):
        if not hasattr(self, '_check_timestamp'):
            self._check_timestamp = time.time()
        return len(self.clients)>=10 or time.time()-self._check_timestamp>=30

    def listen_for_registration(self):
        while not self.is_exit():
            time.sleep(0.1)
            if len(self.register_poller.poll(10000)) > 0:
                reg_req = self.registrar.recv_pyobj()
                if reg_req["name"] not in self.clients.keys():
                    self.add_client(reg_req["name"])
                    l = len(self.clients)
                    self.logger.info("%s joined in the federation. The number of clients is %i" % (reg_req['name'], l))
                    valid_keys = ['num_steps', 'learning_rate', 'batch_size', 'momentum', 'weight_decay', 'num_epochs', 'optimizer']
                    self.registrar.send_pyobj({"client_idx": l, 'port_send': self.port_send, 'port_recv': self.port_recv, '__option__':{k:self.option[k] for k in valid_keys}})
                else:
                    self.logger.info("%s rebuilt the connection." % reg_req['name'])

    def listen_for_sender(self):
        # t = threading.current_thread()
        while not self.is_exit():
            time.sleep(0.5)
            if len(self.receiver_poller.poll(10000)) > 0:
                name = self.receiver.recv_string()
                package_msg = self.receiver.recv()
                package_size = len(package_msg)/1024.0/1024.0
                d = self.receiver._deserialize(package_msg, pickle.loads)
                d['__size__'] = package_size
                if '__mtype__' in d and d['__mtype__']=="close":
                    self.logger.info("{} was successfully closed.".format(name))
                else:
                    self.add_buffer({'name':name, 'package':d})
                    self.logger.info("Received package of size {}MB from {} at round {}".format(package_size, name, self.current_round))

    @property
    def clients(self):
        self._lock_registration.acquire()
        res = self._clients
        self._lock_registration.release()
        return res

    @property
    def num_clients(self):
        return len(self.clients)

    def add_client(self, name):
        self._lock_registration.acquire()
        i = len(self._clients)
        self._clients[i] = name
        self._lock_registration.release()
        return

    def pack(self, client_id, mtype=0, *args, **kwargs):
        if mtype=='close':
            return {}
        else:
            return {'model': self.model}

    def run(self, ip='*', port='5555'):
        self.logger = self.logger(task=self.option['task'], option=self.option, name=self.name+'_'+str(self.logger), level=self.option['log_level'])
        self.logger.register_variable(object=self, server=self)
        self._clients = {}
        self._lock_registration = threading.Lock()
        self._buffer = mlp.Queue()
        self._lock_buffer = threading.Lock()
        self._exit = False
        self._lock_exit = threading.Lock()
        self.ip = ip
        self.port = port
        self.context = zmq.Context()
        self.registrar = self.context.socket(zmq.REP)
        self.registrar.bind("tcp://%s:%s" % (ip, port))
        self.port_send = self.get_free_port()
        self.sender = self.context.socket(zmq.PUB)
        self.sender.bind("tcp://%s:%s" %(ip, self.port_send))
        self.port_recv = self.get_free_port()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("tcp://%s:%s"%(ip, self.port_recv))
        self.register()
        self.current_round = 1
        self.receiver_poller = zmq.Poller()
        self.receiver_poller.register(self.receiver, zmq.POLLIN)
        self.listen_receiver_thread = threading.Thread(target=self.listen_for_sender, )
        self.listen_receiver_thread.start()
        self.logger.time_start('Total Time Cost')
        if self.eval_interval>0:
            # evaluating initial model performance
            self.logger.info("--------------Initial Evaluation--------------")
            self.logger.time_start('Eval Time Cost')
            self.logger.log_once()
            self.logger.time_end('Eval Time Cost')
        while self.current_round<=self.num_rounds:
            updated = self.iterate()
            if updated is True or updated is None:
                self.logger.info("--------------Round {}--------------".format(self.current_round))
                if self.logger.check_if_log(self.current_round, self.eval_interval):
                    self.logger.time_start('Eval Time Cost')
                    self.logger.log_once()
                    self.logger.time_end('Eval Time Cost')
                # check if early stopping
                if self.logger.early_stop(): break
                self.current_round += 1
        self.logger.info("=================End==================")
        self.logger.time_end('Total Time Cost')
        # save results as .json file
        self.logger.save_output_as_json()
        self.set_exit()
        self.communicate([_ for _ in range(len(self.clients))], mtype='close')
        exit(0)

    def aggregate(self, models: list, *args, **kwargs):
        if len(models)==0: return self.model
        return flgo.utils.fmodule._model_average(models)

    def unpack(self, pkgs:dict):
        if len(pkgs)==0: return collections.defaultdict(list)
        keys = list(list(pkgs.values())[0].keys())
        res = {}
        for k in keys:
            res[k] = []
            for cip in pkgs:
                v = pkgs[cip][k]
                res[k].append(v)
        return res

    def communicate_with(self, target_id, package={}):
        self.sender.send_string(target_id, zmq.SNDMORE)
        self.sender.send_pyobj(package)

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
        selected_clients = [self.clients[i] for i in selected_clients]
        self.model.to('cpu')
        for i, name in enumerate(selected_clients):
            package = self.pack(i, mtype=mtype)
            package['__mtype__'] = mtype
            package['__round__'] = self.current_round
            package['name'] = name
            self.communicate_with(name, package)
        buffer = {}
        while not self.is_exit():
            new_comings = self.clear_buffer()
            buffer.update(new_comings)
            if asynchronous or all([(name in buffer) for name in selected_clients]):
                break
            time.sleep(0.1)
        return self.unpack(buffer)

    def global_test(self, model=None, flag: str = 'val'):
        all_metrics = self.communicate([_ for _ in range(len(self.clients))], mtype='%s_metric'%flag)
        return all_metrics

    def get_free_port(self):
        import socket
        sock = socket.socket()
        sock.bind(('', 0))
        ip, port = sock.getsockname()
        sock.close()
        return port

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.actions = {
            '0': self.reply,
            'val_metric': self.val_metric,
            'train_metric': self.train_metric,
            'test_metric': self.test_metric,
        }

    def val_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'val')
        return metrics

    def test_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'test')
        return metrics

    def train_metric(self, package):
        model = package['model']
        metrics = self.test(model, 'train')
        return metrics

    def register(self):
        self.logger.info("%s Registering..." % self.name)
        self.registrar.send_pyobj({"name": self.name})
        reply = self.registrar.recv_pyobj()
        if '__option__' in reply: self.set_option(reply['__option__'])
        return reply["port_recv"], reply["port_send"],

    def message_handler(self, package, *args, **kwargs):
        mtype = package['__mtype__']
        action = self.default_action if mtype not in self.actions else self.actions[mtype]
        response = action(package)
        assert isinstance(response, dict)
        response['__name__'] = self.name
        if hasattr(self, 'round'): response['__round__'] = self.round
        self.sender.send_string(self.name, zmq.SNDMORE)
        msg = pickle.dumps(response, pickle.DEFAULT_PROTOCOL)
        self.logger.info("Sending the package of size {}MB to the server...".format(len(msg)/1024/1024))
        self.sender.send(msg)
        # self.sender.send_pyobj(response)
        return

    def run(self, server_ip='127.0.0.1', server_port='5555'):
        self.logger = self.logger(task=self.option['task'], option=self.option, name=self.name+'_'+str(self.logger), level=self.option['log_level'])
        self.logger.register_variable(object=self, clients = [self])
        self.context = zmq.Context()
        # Registration Socket
        self.registrar = self.context.socket(zmq.REQ)
        self.registrar.connect("tcp://%s:%s"%(server_ip, server_port))
        port_svr_recv, port_svr_send = self.register()
        self.logger.info(f"Successfully Registered to {server_ip}:{server_port}")

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.connect("tcp://%s:%s" % (server_ip, port_svr_send))
        self.receiver.subscribe(self.name)

        self.sender = self.context.socket(zmq.PUSH)
        self.sender.connect("tcp://%s:%s" % (server_ip, port_svr_recv))
        self.logger.info("Ready For Training...")
        self.logger.info("---------------------------------------------------------------------")
        while True:
            name = self.receiver.recv_string()
            assert name == self.name
            try:
                package_msg = self.receiver.recv()
                package_size = len(package_msg)
                package = self.receiver._deserialize(package_msg, pickle.loads)
                # package = self.receiver.recv_pyobj()
                assert '__mtype__' in package
                if package['__mtype__']=='close':
                    self.sender.send_string(self.name, zmq.SNDMORE)
                    self.sender.send_pyobj({'__mtype__':"close"})
                    break
                package['__size__'] = package_size/1024/1024 #MB
                if '__round__' in package.keys():
                    self.round = package['__round__']
                    self.logger.info("Selected at round {} and received the package of {}MB".format(package['__round__'], package['__size__']))
                self.message_handler(package)
            except Exception as e:
                print(e)
                break
        self.logger.info("%s has been closed" % self.name)
        torch.cuda.empty_cache()
        exit(0)

    def set_option(self, option:dict={}):
        valid_keys = ['num_steps', 'learning_rate', 'batch_size', 'momentum', 'weight_decay', 'num_epochs', 'optimizer']
        types = [int, float, float, float, float, int, str]
        self.option.update(option)
        for k,t in zip(valid_keys, types):
            if k in option:
                try:
                    setattr(self, k, t(option[k]))
                except:
                    self.logger.info("Failed to set hyper-parameter {}={}".format(k, option[k]))
                    continue
        # correct hyper-parameters
        if hasattr(self, 'train_data'):
            import math
            self.datavol = len(self.train_data)
            if hasattr(self, 'batch_size'):
                # reset batch_size
                if self.batch_size < 0:
                    self.batch_size = self.datavol
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
        return

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
            import math
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
class algo:
    Server = Server
    Client = Client


if __name__=='__main__':
    mlp.set_start_method('spawn', force=True)
    mlp.set_sharing_strategy('file_system')
    import flgo.benchmark.mnist_classification as mnist
    import flgo.benchmark.partition as fbp
    flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=10), 'my_task')
    runner = flgo.init('my_task', algo, option={'proportion':0.2, 'gpu':[0,1,2,3,], 'server_with_cpu':True, 'num_rounds':10, 'num_steps':1, 'log_file':True}, scene='parallel_horizontal')
    runner.run()

