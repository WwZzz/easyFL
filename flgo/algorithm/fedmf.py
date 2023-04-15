"""
This is a non-official implementation of the work 'Secure Federated Matrix Factorization' (https://arxiv.org/abs/1906.05108).
"""
import collections
import torch
import copy
from .fedbase import BasicServer
from .fedbase import BasicClient
from phe import paillier
import numpy as np
from tqdm import tqdm

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        # hyper-parameters
        self.init_algo_para({'lambda_l2':1e-4, 'part':1})
        self.gv.public_key, self.gv.private_key = paillier.generate_paillier_keypair(n_length=1024, )
        # initialize item vectors as [0.01 ... 0.01] as the official code did (just for testing)
        # encrypt the item vectors
        if self.part < 0:
            # if part < 0, it won't encrypt item embeddings
            self.encrypted_item_vectors = self.model.get_embedding().detach().cpu().numpy()
        else:
            tmp = self.model.get_embedding().detach().cpu().numpy().tolist()
            self.encrypted_item_vectors = []
            self.gv.logger.info('Encrypt all the item embeddings...')
            for vector in tqdm(tmp):
                self.encrypted_item_vectors.append([self.gv.public_key.encrypt(e, precision=1e-5) for e in vector])

    def iterate(self):
        self.selected_clients = self.sample()
        en_grads = self.communicate(self.selected_clients)['encrypted_gradient']
        self.aggregate(en_grads)

        # update decrypted item embeddings

    def pack(self, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'encrypted_item_vectors': self.encrypted_item_vectors
        }

    def aggregate(self, encrypted_grads):
        if self.part<0:
            for g in encrypted_grads:
                self.encrypted_item_vectors -= g
        elif self.part==0:
                for eng in encrypted_grads:
                    for i in range(len(self.encrypted_item_vectors)):
                        for j in range(len(self.encrypted_item_vectors[i])):
                            self.encrypted_item_vectors[i][j] = self.encrypted_item_vectors[i][j] - eng[i][j]*1.0/len(encrypted_grads)
        elif self.part==1:
            for eng in encrypted_grads:
                for item_id in eng:
                    for j in range(len(self.encrypted_item_vectors[item_id])):
                        self.encrypted_item_vectors[item_id][j] = self.encrypted_item_vectors[item_id][j] - eng[item_id][j]*1.0/len(encrypted_grads)

    def global_test(self, model=None, flag:str='valid'):
        if model is None: model=self.model
        all_metrics = collections.defaultdict(list)
        for c in self.clients:
            client_metrics = c.test(model, flag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics

    def test(self, model=None, flag='test'):
        data = self.test_data if flag=='test' else self.valid_data
        if data is None: return {}
        # construct user embeddings
        clients = sorted(self.clients, key=lambda x: x.train_data.user_id[0])
        user_embeddings = np.array([c.model.get_embedding().view(-1).detach().cpu().numpy() for c in clients])
        local_model = self.model.__class__(len(clients), 20).to(self.device)
        local_model.set_embedding(user_embeddings)
        # construct item embeddings
        if self.part<0:
            tmp = self.encrypted_item_vectors
        else:
            tmp = np.array([[self.gv.private_key.decrypt(e) for e in vector] for vector in tqdm(self.encrypted_item_vectors)], dtype=np.float32)
        self.model.set_embedding(tmp)
        return self.calculator.test(models = (self.model, local_model), dataset=data, batch_size=self.option['test_batch_size'], num_workers=self.option['num_workers'], pin_memory=self.option['pin_memory'])

class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        grad = self.train(model)
        cpkg = self.pack(grad)
        return cpkg

    def unpack(self, received_pkg):
        """Return the decrypted item embeddings (i.e. the instance of ItemEmbedding())"""
        encrypted_item_vectors = received_pkg['encrypted_item_vectors']
        model = received_pkg['model']
        # local_movielens_recommendation decrypt
        if self.part<0:
            item_embeddings = encrypted_item_vectors
        else:
            item_vectors_np = []
            self.gv.logger.info('Client {} decrypt the item embeddings...'.format(self.id))
            for vector in tqdm(encrypted_item_vectors):
                item_vectors_np.append([self.gv.private_key.decrypt(e) for e in vector])
            item_embeddings = np.array(item_vectors_np)
        model.set_embedding(item_embeddings)
        return model

    def train(self, global_model):
        # self.gv.logger.info('Client {} locally train the models...'.format(self.id))
        # zero grad
        original_model = copy.deepcopy(global_model)
        global_model.zero_grad()
        self.model.zero_grad()
        # get a batch of data
        optimizer = torch.optim.SGD([{'params':self.model.parameters(), 'lr':self.learning_rate}, {'params':global_model.parameters(), 'lr':self.learning_rate}], weight_decay=self.lambda_l2)
        for _ in range(self.num_steps):
            batch_data = self.get_batch_data()
            self.model.to(self.device)
            global_model.to(self.device)
            loss = self.calculator.compute_loss((global_model, self.model), batch_data)['loss']
            # predict = global_model(batch_data, self.model.get_embedding())
            # loss = ((batch_data['rating']-predict)**2).sum()/len(batch_data['item_id'])
            loss.backward()
            optimizer.step()
        global_model.to(torch.device('cpu'))
        gradient = (global_model.get_embedding()-original_model.get_embedding()).detach().cpu().numpy()
        return gradient

    def pack(self, gradient):
        if self.part==0:
            encrypted_gradient = []
            for vector in tqdm(gradient):
                encrypted_gradient.append([self.gv.public_key.encrypt(float(e), precision=1e-5) for e in vector])
        elif self.part==1:
            self.gv.logger.info('Client {} encrypt the gradient...'.format(self.id))
            encrypted_gradient = {}
            item_list = list(set(self.train_data.item_id.tolist()))
            for item_id in tqdm(item_list):
                encrypted_gradient[item_id] = [self.gv.public_key.encrypt(float(e), precision=1e-5) for e in gradient[item_id]]
        else:
            encrypted_gradient = gradient
        return {'encrypted_gradient': encrypted_gradient}

    def test(self, global_model, flag='valid'):
        data = self.train_data if flag=='train' else self.valid_data
        if data is None: return {}
        return self.calculator.test(models=(global_model, self.model), dataset=data, batch_size=self.option['test_batch_size'], pin_memory=self.option['pin_memory'], num_workers=self.option['num_workers'])

