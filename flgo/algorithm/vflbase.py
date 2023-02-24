from .fedbase import BasicServer, BasicParty
import collections
import torch
import torch.multiprocessing as mp

class ActiveParty(BasicServer):
    def __init__(self, option):
        self.actions = {0: self.forward, 1: self.backward,2:self.forward_test}
        self.device = torch.device('cpu') if option['server_with_cpu'] else self.gv.apply_for_device()
        self.calculator = self.gv.TaskCalculator(self.device, optimizer_name = option['optimizer'])
        # basic configuration
        self.task = option['task']
        self.eval_interval = option['eval_interval']
        self.num_parallels = option['num_parallels']
        # server calculator
        # self.calculator = self.gv.TaskCalculator(self.device, optimizer_name=option['optimizer'])
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.proportion = option['proportion']
        self.batch_size = option['batch_size']
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
        self.id = -1

    def communicate(self, selected_clients, mtype=0, asynchronous=False):
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
                self.sending_package_buffer[cid] = self.pack(cid, mtype=mtype)
        except MemoryError as e:
            if str(self.device) != 'cpu':
                self.model.to(torch.device('cpu'))
                for cid in communicate_clients:
                    self.sending_package_buffer[cid] = self.pack(cid, mtype=mtype)
                self.model.to(self.device)
            else:
                raise e
        # communicate with selected clients
        if self.num_parallels <= 1:
            # computing iteratively
            for client_id in communicate_clients:
                response_from_client_id = self.communicate_with(client_id, package=self.sending_package_buffer[cid], mtype=mtype)
                packages_received_from_clients.append(response_from_client_id)
        else:
            # computing in parallel with torch.multiprocessing
            pool = mp.Pool(self.num_parallels)
            for client_id in communicate_clients:
                self.clients[client_id].update_device(self.gv.apply_for_device())
                args = (int(client_id), self.sending_package_buffer[cid], mtype)
                packages_received_from_clients.append(pool.apply_async(self.communicate_with, args=args))
            pool.close()
            pool.join()
            packages_received_from_clients = list(map(lambda x: x.get(), packages_received_from_clients))
        for i,cid in enumerate(communicate_clients): received_package_buffer[cid] = packages_received_from_clients[i]
        packages_received_from_clients = [received_package_buffer[cid] for cid in selected_clients if received_package_buffer[cid]]
        self.received_clients = selected_clients
        return self.unpack(packages_received_from_clients)

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
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
                self.current_round += 1
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

    def iterate(self):
        self._data_type='train'
        self.crt_batch = self.get_batch_data()
        activations = self.communicate([pid for pid in range(len(self.parties))], mtype=0)['activation']
        self.defusions = self.update_global_module(activations, self.global_module)
        _ = self.communicate([pid for pid in range(len(self.parties))], mtype=1)
        return True

    def pack(self, client_id, mtype=0):
        if mtype==0:
            return {'batch': self.crt_batch[2], 'data_type': self._data_type}
        elif mtype==1:
            return {'derivation': self.defusion[client_id]}
        elif mtype==2:
            return {'batch': self.crt_test_batch[2], 'data_type': self._data_type}

    def get_batch_data(self):
        """
        Get the batch of data
        :return:
            a batch of data
        """
        try:
            batch_data = next(self.data_loader)
        except:
            self.data_loader = iter(self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size))
            batch_data = next(self.data_loader)
        return batch_data

    def update_global_module(self, activations, model):
        self.fusion = self.fuse(activations)
        self.fusion.requires_grad=True
        optimizer = self.calculator.get_optimizer(self.global_module, lr=self.lr)
        loss = self.calculator.compute_loss(model, (self.fusion, self.crt_batch[1]))['loss']
        loss.backward()
        optimizer.step()
        self.defusion = self.defuse(self.fusion)

    def fuse(self, activations):
        return torch.stack(activations).mean(dim=0)

    def defuse(self, fusion):
        return [fusion.grad for _ in self.parties]

    def update_local_module(self, derivation, activation):
        optimizer = self.calculator.get_optimizer(self.local_module, self.lr)
        loss_surrogat = (derivation*activation).sum()
        loss_surrogat.backward()
        optimizer.step()
        return

    def forward(self, package):
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'valid': self.valid_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation.clone().detach()}

    def backward(self, package):
        derivation = package['derivation']
        self.update_local_module(derivation, self.activation)
        return

    def communicate_with(self, client_id, package={}, mtype=0):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
            package: the package to be sended to the client
            mtype: the type of the message that is used to decide the action of the client
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # listen for the client's response
        return self.gv.communicator.request(self.id-1, client_id-1, package, mtype)

    def test(self, flag='test'):
        self.set_model_mode('eval')
        flag_dict = {'test':self.test_data, 'train':self.train_data, 'valid':self.valid_data}
        dataset = flag_dict[flag]
        self._data_type = flag
        dataloader = self.calculator.get_dataloader(dataset, batch_size=128)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(dataloader):
            self.crt_test_batch = batch_data
            activations = self.communicate([pid for pid in range(len(self.parties))], mtype=2)['activation']
            fusion = self.fuse(activations)
            outputs = self.global_module(fusion.to(self.device))
            batch_mean_loss = self.calculator.criterion(outputs, batch_data[1].to(self.device)).item()
            y_pred = outputs.data.max(1, keepdim=True)[1].cpu()
            correct = y_pred.eq(batch_data[1].data.view_as(y_pred)).long().cpu().sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[1])
        self.set_model_mode('train')
        return {'accuracy': 1.0 * num_correct / len(dataset), 'loss': total_loss / len(dataset)}

    def forward_test(self, package):
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'valid': self.valid_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation}

    def set_model_mode(self,mode = 'train'):
        for party in self.parties:
            if party.local_module is not None:
                if mode == 'train':
                    party.local_module.train()
                else:
                    party.local_module.eval()
            if party.global_module is not None:
                if mode == 'train':
                    party.global_module.train()
                else:
                    party.global_module.eval()

class PassiveParty(BasicParty):
    def __init__(self, option):
        self.option = option
        self.actions = {0: self.forward, 1:self.backward, 2:self.forward_test}
        self.id = None
        # create local dataset
        self.data_loader = None
        # local calculator
        self.device = self.gv.apply_for_device()
        self.calculator = self.gv.TaskCalculator(self.device, option['optimizer'])
        # hyper-parameters for training
        self.optimizer_name = option['optimizer']
        self.lr = option['learning_rate']
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


    def forward(self, package):
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'valid': self.valid_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation.clone().detach()}

    def backward(self, package):
        derivation = package['derivation']
        self.update_local_module(derivation, self.activation)
        return

    def update_local_module(self, derivation, activation):
        optimizer = self.calculator.get_optimizer(self.local_module, self.lr)
        loss_surrogat = (derivation*activation).sum()
        loss_surrogat.backward()
        optimizer.step()
        return

    def forward_test(self, package):
        batch_ids = package['batch']
        tmp = {'train': self.train_data, 'valid': self.valid_data, 'test':self.test_data}
        dataset = tmp[package['data_type']]
        # select samples in batch
        self.activation = self.local_module(dataset.get_batch_by_id(batch_ids)[0].to(self.device))
        return {'activation': self.activation}