from .fedbase import BasicServer, BasicClient
import copy
import torch

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['mu']

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, drop_rate=-1):
        super(Client, self).__init__(option, name, train_data, valid_data, drop_rate)
        self.mu = option['mu']

    def train(self, model):
        # global parameters
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()
                l1 = self.calculator.get_loss(model, batch_data)
                # prox. item
                l2 = 0
                for pm, ps in zip(model.parameters(), src_model.parameters()):
                    l2 += torch.sum(torch.pow(pm-ps,2))
                loss = l1 + 0.5 * self.mu * l2
                loss.backward()
                optimizer.step()
        return

