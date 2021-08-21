from .fedbase import BaseServer, BaseClient
import torch.nn.functional as F
from torch.utils.data import DataLoader
from task.modelfuncs import device,lossfunc,optim, modeldict_to_tensor1D
import copy
import torch

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)
        self.mu=option['mu']
        self.paras_name = ['mu']

    def train(self):
        # global parameters
        src_model = copy.deepcopy(self.model.state_dict())
        src_vec = modeldict_to_tensor1D(src_model)
        self.model.train()
        if self.batch_size == -1:
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                self.model.zero_grad()
                outputs = self.model(images)
                loss = lossfunc(outputs, labels)
                tmp_vec = torch.Tensor().to(device)
                for crt in self.model.parameters():
                    tmp_vec = torch.cat((tmp_vec, crt.view(-1)))
                reg_loss = torch.norm(tmp_vec - src_vec, 2)**2
                loss+=self.mu/2*reg_loss
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

