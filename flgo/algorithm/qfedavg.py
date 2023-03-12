import flgo.algorithm.fedbase as fedbase
import flgo.utils.fmodule as fmodule
import copy

class Server(fedbase.BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'q': 1.0})

    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        self.model = self.model - fmodule._model_sum(res['dk']) / sum(res['hk'])
        return len(self.received_clients) > 0

class Client(fedbase.BasicClient):
    def unpack(self, package):
        model = package['model']
        self.global_model = copy.deepcopy(model)
        return model

    def pack(self, model):
        Fk = self.test(self.global_model, 'train')['loss'] + 1e-8
        L = 1.0 / self.learning_rate
        delta_wk = L * (self.global_model - model)
        dk = (Fk ** self.q) * delta_wk
        hk = self.q * (Fk ** (self.q - 1)) * (delta_wk.norm() ** 2) + L * (Fk ** self.q)
        self.global_model = None
        return {'dk': dk, 'hk': hk}