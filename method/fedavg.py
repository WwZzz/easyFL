from .fedbase import BaseServer, BaseClient

class Server(BaseServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)

class Client(BaseClient):
    def __init__(self, option, name = '', data_train_dict = {'x':[],'y':[]}, data_test_dict={'x':[],'y':[]}, partition = True):
        super(Client, self).__init__(option, name, data_train_dict, data_test_dict, partition)


