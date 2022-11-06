import time
import torch
from .fedbase import BasicServer
from .fedavg import Client
import utils.fflow as flw
import utils.logger.basic_logger as bl
import ujson
import os
import math
from tqdm import tqdm

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
        optimizer = self.calculator.get_optimizer(self.model, lr=self.option['learning_rate'],
                                                  weight_decay=self.option['weight_decay'], momentum=self.option['momentum'])
        flw.logger.time_start('Total Time Cost')
        for epoch in tqdm(range(self.epochs)):
            # evaluate
            if flw.logger.check_if_log(epoch, self.eval_interval):
                flw.logger.info('Evaluate model of epoch {}'.format(epoch))
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once(current_iter=epoch)
                flw.logger.time_end('Eval Time Cost')
            flw.logger.info('Starting to train model at epoch {}'.format(epoch))
            for iter in tqdm(range(self.num_iters_per_epoch)):
                batch_data = self.get_batch_data()
                self.model.zero_grad()
                # calculate the loss of the model on batched dataset through task-specified calculator
                loss = self.calculator.train_one_step(self.model, batch_data)['loss']
                loss.backward()
                optimizer.step()
            flw.logger.info('Ending training model of epoch {}'.format(epoch))
        flw.logger.time_end('Total Time Cost')
        flw.logger.save_output_as_json(self)

    def get_batch_data(self):
        if not self.data_loader:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size))
        try:
            batch_data = next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size))
            batch_data = next(self.data_loader)
        return batch_data

class Logger(bl.Logger):
    def log_once(self, current_iter):
        test_metrics = self.server.calculator.test(self.server.model, self.server.test_data, self.server.batch_size)
        train_metrics = self.server.calculator.test(self.server.model, self.server.train_data, self.server.batch_size)
        valid_metrics = self.server.calculator.test(self.server.model, self.server.valid_data, self.server.batch_size)
        for met_name, met_value in test_metrics.items():
            self.write_var_into_output('test_'+met_name, met_value)
        for met_name, met_value in train_metrics.items():
            self.write_var_into_output('train_'+met_name, met_value)
        for met_name, met_value in valid_metrics.items():
            self.write_var_into_output('valid_'+met_name, met_value)
        self.show_current_output()

    def save_output_as_json(self, filepath=None):
        """Save the self.output as .json file"""
        filepath = os.path.join(self.get_output_path(), self.get_output_name())
        model_name = self.meta['model']+'_iter'+str(self.server.epochs)+'_'+self.meta['algorithm']+'_'+time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())+'.pth'
        model_path = os.path.join('fedtask', self.meta['task'], 'record', model_name)
        with open(filepath, 'w') as outf:
            ujson.dump(dict(self.output), outfs
        torch.save({'model':self.server.model.state_dict()}, model_path)