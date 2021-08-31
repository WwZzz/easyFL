from .fedbase import BaseServer, BaseClient
from torch.utils.data import DataLoader
from utils.fmodule import device,lossfunc,Optim
import copy
from utils import fmodule

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)
        self.paras_name = ['mu']

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, train_rate = 0.8, drop_rate = 0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)
        self.mu = option['mu']

    def train(self, model):
        # global parameters
        src_model = copy.deepcopy(model)
        model.train()
        if self.batch_size == -1:
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = fmodule.get_optimizer(self.optimizer, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(images)
                loss = lossfunc(outputs, labels)
                loss+=self.mu/2 * ((model-src_model).norm() ** 2)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

