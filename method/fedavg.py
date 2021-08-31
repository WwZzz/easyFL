from .fedbase import BaseServer, BaseClient

class Server(BaseServer):
    def __init__(self, option, model, clients, dtest = None):
        super(Server, self).__init__(option, model, clients, dtest)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, train_rate = 0.8, drop_rate=0):
        super(Client, self).__init__(option, name, data_train_dict, data_val_dict, train_rate, drop_rate)


