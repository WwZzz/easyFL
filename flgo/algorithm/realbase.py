import collections
import os.path
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
        while True:
            time.sleep(1)
            if self.if_start():
                self.logger.info("Start training...")
                break
        return

    def if_start(self):
        tmp = input("Press Y and Enter to start training: ")
        return (tmp.lower()=='y' or tmp.lower()=='yes')
        # if not hasattr(self, '_check_timestamp'):
        #     self._check_timestamp = time.time()
        # return len(self.clients)>=10 or time.time()-self._check_timestamp>=30

    def register_handler(self, worker_id, client_id, received_pkg):
        valid_keys = ['num_steps', 'learning_rate', 'batch_size', 'momentum', 'weight_decay', 'num_epochs', 'optimizer']
        if received_pkg["name"] not in self.clients.keys():
            self.add_client(received_pkg["name"])
            l = len(self.clients)
            self.logger.info("%s joined in the federation. The number of clients is %i" % (received_pkg['name'], l))

            d = {"client_idx": l, 'port_send': self.port_send, 'port_recv': self.port_recv,
                 '__option__': {k: self.option[k] for k in valid_keys}}
            self.registrar.send_multipart([worker_id, client_id, pickle.dumps(d, pickle.DEFAULT_PROTOCOL)])
        else:
            self.logger.info("%s rebuilt the connection." % received_pkg['name'])
            self.registrar.send_pyobj({"client_idx": len(self.clients), 'port_send': self.port_send, 'port_recv': self.port_recv, '__option__': {k: self.option[k] for k in valid_keys}})

    def task_pusher_handler(self, worked_id, client_id, *args, **kwargs):
        zipped_task = self._get_zipped_task()
        if zipped_task is None:
            self._read_zipped_task()
            zipped_task = self._get_zipped_task()
        for _data in zipped_task:
            self.task_pusher.send_multipart([worked_id, client_id, _data])
        return

    def _listen(self):
        while not self.is_exit():
            time.sleep(0.5)
            events = dict(self._poller.poll())
            if self.task_pusher in events and events[self.task_pusher]==zmq.POLLIN:
                worker_id = self.task_pusher.recv()
                client_id = self.task_pusher.recv()
                received_pkg = self.task_pusher.recv_pyobj()
                t = threading.Thread(target=self.task_pusher_handler, args=(worker_id, client_id, received_pkg))
                t.start()
            if self.registrar in events and events[self.registrar]==zmq.POLLIN:
                worker_id = self.registrar.recv()
                client_id = self.registrar.recv()
                received_pkg = self.registrar.recv_pyobj()
                t = threading.Thread(target=self.register_handler, args=(worker_id, client_id, received_pkg))
                t.start()
            if self.receiver in events and events[self.receiver]==zmq.POLLIN:
                name = self.receiver.recv_string()
                package_msg = self.receiver.recv()
                package_size = len(package_msg) / 1024.0 / 1024.0
                d = self.receiver._deserialize(package_msg, pickle.loads)
                d['__size__'] = package_size
                if '__mtype__' in d and d['__mtype__'] == "close":
                    self.logger.info("{} was successfully closed.".format(name))
                else:
                    self.add_buffer({'name': name, 'package': d})
                    self.logger.info("Received package of size {}MB from {} at round {}".format(package_size, name,
                                                                                                self.current_round))

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

    def _read_zipped_task(self, with_bmk=True):
        task_path = self.option['task']
        task_name = os.path.basename(task_path)
        task_dir = os.path.dirname(os.path.abspath(task_path))
        task_zip = task_name + '.zip'
        if not os.path.exists(task_zip):
            flgo.zip_task(task_path, target_path=task_dir, with_bmk=with_bmk)
        if not hasattr(self, '_zipped_task'): self._zipped_task = []
        CHUNK_SIZE = 1024
        with open(os.path.join(task_dir, task_zip), 'rb') as inf:
            while True:
                chunk = inf.read(CHUNK_SIZE)
                self._zipped_task.append(chunk)
                if not chunk:
                    break

    def _get_zipped_task(self, copy=True):
        if not hasattr(self, '_get_zipped_task'): return None
        return self._zipped_task.copy() if copy==True else self._zipped_task

    def run(self, ip='*', port='5555', protocol='tcp'):
        self._read_zipped_task()
        self.logger = self.logger(task=self.option['task'], option=self.option, name=self.name+'_'+str(self.logger), level=self.option['log_level'])
        self.logger.register_variable(object=self, server=self)
        self._clients = {}
        self.ip = ip
        self.port = port
        self._lock_registration = threading.Lock()
        self._buffer = mlp.Queue()
        self._lock_buffer = threading.Lock()
        self._exit = False
        self._lock_exit = threading.Lock()

        self.context = zmq.Context()
        self.registrar = self.context.socket(zmq.ROUTER)
        self.registrar.bind("%s://%s:%s" % (protocol, ip, port))

        self.port_task = self.get_free_port()
        self.task_pusher = self.context.socket(zmq.STREAM)
        self.task_pusher.bind("%s://%s:%s" % (protocol, ip, self.port_task))
        self.logger.info("Publish Task %s in %s://%s:%s"% (os.path.basename(self.option['task']),protocol, ip, self.port_task))

        self.port_send = self.get_free_port()
        self.sender = self.context.socket(zmq.PUB)
        self.sender.bind("%s://%s:%s" %(protocol, ip, self.port_send))

        self.port_recv = self.get_free_port()
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("%s://%s:%s"%(protocol, ip, self.port_recv))

        self._poller = zmq.Poller()
        self._poller.register(self.task_pusher, zmq.POLLIN)
        self._poller.register(self.registrar, zmq.POLLIN)
        self._poller.register(self.receiver, zmq.POLLIN)
        self._thread_listening = threading.Thread(target=self._listen)
        self._thread_listening.start()

        self.register()

        self.current_round = 1
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
