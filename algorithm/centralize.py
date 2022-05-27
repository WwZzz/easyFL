import time
import torch
from .fedbase import BasicServer, BasicClient
import utils.fflow as flw
import ujson
import os
import math

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.data_loader = None
        train_data = self.clients[0].train_data
        valid_data = self.clients[0].valid_data
        for i in range(1, self.num_clients):
            train_data = train_data + self.clients[i].train_data
            valid_data = valid_data + self.clients[i].valid_data
        self.train_data = train_data
        self.valid_data = valid_data
        self.batch_size = len(self.train_data) if option['batch_size'] == -1 else int(option['batch_size'])
        self.epochs = option['num_epochs']
        self.num_iters_per_epoch = math.ceil(len(self.train_data)/self.batch_size)
        self.num_iters = option['num_steps'] if option['num_steps'] > 0 else self.epochs * self.num_iters_per_epoch
        self.epochs = math.ceil(1.0*self.num_iters/self.num_iters_per_epoch)

    def run(self):
        # training
        self.model.train()
        optimizer = self.calculator.get_optimizer(self.option['optimizer'], self.model, lr=self.option['learning_rate'],
                                                  weight_decay=self.option['weight_decay'], momentum=self.option['momentum'])
        flw.logger.time_start('Total Time Cost')
        for iter in range(self.num_iters):
            # evaluate
            if flw.logger.check_if_log(iter, self.eval_interval):
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log(self, current_iter=iter)
                flw.logger.time_end('Eval Time Cost')
            batch_data = self.get_batch_data()
            self.model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(self.model, batch_data)
            loss.backward()
            optimizer.step()
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log(self, current_iter=self.num_iters)
        flw.logger.time_end('Eval Time Cost')
        flw.logger.time_end('Total Time Cost')
        flw.logger.save(self)

    def get_batch_data(self):
        if not self.data_loader:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size))
        try:
            batch_data = next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size))
            batch_data = next(self.data_loader)
        return batch_data

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

class MyLogger(flw.Logger):
    def log(self, server=None, current_iter=-1):
        if len(self.output) == 0:
            self.output['meta'] = server.option
        test_metrics = server.calculator.test(server.model, server.test_data, server.batch_size)
        train_metrics = server.calculator.test(server.model, server.train_data, server.batch_size)
        valid_metrics = server.calculator.test(server.model, server.valid_data, server.batch_size)
        for met_name, met_value in test_metrics.items():
            self.write('test_'+met_name, met_value)
        for met_name, met_value in train_metrics.items():
            self.write('train_'+met_name, met_value)
        for met_name, met_value in valid_metrics.items():
            self.write('valid_'+met_name, met_value)
        print('----------Iter {}/{} :: Epoch {}/{}-------------'.format(current_iter%server.num_iters_per_epoch, server.num_iters_per_epoch,current_iter//server.num_iters_per_epoch, server.epochs))
        for key, val in self.output.items():
            if key == 'meta': continue
            print(self.temp.format(key, val[-1]))

    def save(self, server):
        """Save the self.output as .json file"""
        rec_path = os.path.join('fedtask', server.option['task'], 'record', flw.output_filename(server.option, server))
        model_name = server.option['model']+'_iter'+str(server.num_iters)+'_'+server.option['algorithm']+'_'+time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())+'.pth'
        model_path = os.path.join('fedtask', server.option['task'], 'record', model_name)
        with open(rec_path, 'w') as outf:
            ujson.dump(dict(self.output), outf)
        torch.save({'model':server.model.state_dict()}, model_path)