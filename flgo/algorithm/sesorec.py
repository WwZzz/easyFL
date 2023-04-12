"""
This is a non-official implementation of SeSoRec proposed in
'Secure Social Recommendation based on Secret Sharing' (https://arxiv.org/abs/2002.02088).
"""
import flgo.algorithm.vflbase as vflbase
import flgo.utils.fmodule as fmodule
import torch.nn

class MF(fmodule.FModule):
    def __init__(self, num_users, num_items, dim=5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.all_users = torch.LongTensor(list(range(num_users)))
        self.all_items = torch.LongTensor(list(range(num_items)))
        self.user_embedding = torch.nn.Embedding(num_users, dim)
        self.item_embedding = torch.nn.Embedding(num_items, dim)
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.normal(self.user_embedding.weight)
        torch.nn.init.normal(self.item_embedding.weight)

    def forward(self, batch_data):
        users, items, ratings  = batch_data
        Ub = self.user_embedding(users)
        Vb = self.item_embedding(items)
        output = (Ub*Vb).sum(dim=1)
        return 5*self.sigmoid(output)

    def get_embeddings(self, batch_data):
        users, items, ratings  = batch_data
        users = torch.unique(users)
        items = torch.unique(items)
        Ubmap = {user.item(): uid for uid, user in enumerate(users)}
        Vbmap = {item.item(): iid for iid, item in enumerate(items)}
        Ub = self.user_embedding(users).T
        Vb = self.item_embedding(items).T
        Rb = torch.zeros((len(users), len(items))).to(Ub.device)
        for u,i,r in zip(users, items, ratings):
            uid, iid = Ubmap[u.item()], Vbmap[i.item()]
            Rb[uid][iid] = r
        U = self.user_embedding.weight.T
        V = self.item_embedding.weight.T
        return Ub, Vb, Ubmap, Vbmap, Rb, U, V

def init_global_module(object):
    pass

def init_local_module(object):
    if object.name == 'Advertiser':
        object.local_module = MF(len(object.userID_data), len(object.itemID_data), 10)

class ActiveParty(vflbase.ActiveParty):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'gamma':1, 'lambda_l2':0.01})
        self.learning_rate = self.option['learning_rate']

    def iterate(self):
        with torch.no_grad():
            # sample a batch of data
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)
            Ub, Vb, Ubmap, Vbmap, Rb, U, V = self.local_module.get_embeddings(batch_data)
            Ib = torch.sign(Rb)
            # communicate with the MediaCompany to obtain Z1=Ub(Db'+Eb') and Z2=US' by SSMM
            Z1 = self.secret_sharing_matrix_multiplication(Ub, list(Ubmap.keys()), mtype='Z1')
            Z2 = self.secret_sharing_matrix_multiplication(U, list(Ubmap.keys()), mtype='Z2')
            # use Z1 and Z2 to compute grad(U) and grad(V)
            sigmoidUV = self.local_module.sigmoid(torch.matmul(Ub.T, Vb))
            dR = (Rb-5*sigmoidUV)*Ib*5*sigmoidUV*(1-sigmoidUV)
            gUb = -torch.matmul(Vb, dR.T) + 0.5*self.gamma*Z1 - self.gamma*Z2 + self.lambda_l2*Ub
            gVb = -torch.matmul(Ub, dR) + self.lambda_l2*Vb
            # update self.U and self.V according to (gU, gV) and batch_data
            U[:, list(Ubmap.keys())] -= self.learning_rate*gUb
            V[:, list(Vbmap.keys())] -= self.learning_rate*gVb
            self.update_local_module(U,V)
        return True

    def update_local_module(self, U, V):
        self.local_module.user_embedding.weight = U.T
        self.local_module.item_embedding.weight = V.T

    def secret_sharing_matrix_multiplication(self, P, users, mtype='Z1'):
        if P.shape[-1]%2==1: P = torch.cat((P, torch.zeros((P.shape[0], 1)).to(P.device)), 1)
        P_rdm = torch.randn_like(P)
        P_rdme, P_rdmo = self.split_matrix(P_rdm, by='col')
        P1 = P + P_rdm
        P2 = P_rdme+P_rdmo
        res = self.communicate_with(1, {'P1': P1, 'P2':P2, 'users':users, '__mtype__':mtype})
        Q1, Q2, N = res['Q1'], res['Q2'], res['N']
        M = torch.matmul(P+2*P_rdm,Q1) + torch.matmul(P2+P_rdmo,Q2)
        return M+N

    def split_matrix(self, M, by='row'):
        if by=='row':
            odd_idx = [i for i in range(0, M.shape[0], 2)]
            even_idx = [i for i in range(1, M.shape[0], 2)]
            return M[even_idx], M[odd_idx]
        else:
            tmp1, tmp2 = self.split_matrix(M.T, by='row')
            return tmp1.T, tmp2.T

    def update_local_module(self, U, V):
        self.local_module.U = U
        self.local_module.V = V

    def init_algo_para(self, algo_para: dict):
        """
        Initialize the algorithm-dependent hyper-parameters for the server and all the clients.
        :param
            algo_paras (dict): the dict that defines the hyper-parameters (i.e. name, value and type) for the algorithm.

        Example 1:
            calling `self.init_algo_para({'u':0.1})` will set the attributions `server.u` and `c.u` as 0.1 with type float where `c` is an instance of `CLient`.
        Note:
            Once `option['algo_para']` is not `None`, the value of the pre-defined hyperparameters will be replaced by the list of values in `option['algo_para']`,
            which requires the length of `option['algo_para']` is equal to the length of `algo_paras`
        """
        self.algo_para = algo_para
        if len(self.algo_para)==0: return
        # initialize algorithm-dependent hyperparameters from the input options
        if self.option['algo_para'] is not None:
            # assert len(self.algo_para) == len(self.option['algo_para'])
            keys = list(self.algo_para.keys())
            for i,pv in enumerate(self.option['algo_para']):
                if i==len(self.option['algo_para']): break
                para_name = keys[i]
                try:
                    self.algo_para[para_name] = type(self.algo_para[para_name])(pv)
                except:
                    self.algo_para[para_name] = pv
        # register the algorithm-dependent hyperparameters as the attributes of the server and all the clients
        for para_name, value in self.algo_para.items():
            self.__setattr__(para_name, value)
            for c in self.parties:
                if c.id!=self.id:
                    c.__setattr__(para_name, value)
        return

    def test(self, model=None, flag='test'):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model (flgo.utils.fmodule.FModule): the model need to be evaluated
        :return:
            metrics (dict): specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None: model=self.local_module
        data = getattr(self, flag+'_data')
        if data is None: return {}
        else:
            return self.calculator.test(model, data, batch_size = self.option['test_batch_size'])

class PassiveParty(vflbase.PassiveParty):
    def initialize(self, *args, **kwargs):
        self.actions={'Z1':self.reply_for_Z1, 'Z2':self.reply_for_Z2}
        self.edge_index = self.social_data.edge_index
        self.num_users = len(self.userID_data)
        self.S = torch.zeros((self.num_users, self.num_users))
        for us,ut in zip(self.edge_index[0], self.edge_index[1]):
            self.S[us][ut] = self.S[ut][us] = 1

    def reply_for_Z1(self, package):
        # set Q=(Db'+Eb')
        users = package['users']
        Db = torch.diag(self.S.sum(dim=1)[users])
        user_in_batch = torch.tensor([[1 if u in users else 0 for u in range(self.num_users)]])
        user_in_batch = torch.repeat_interleave(user_in_batch, self.num_users, dim=0).T
        Eb = torch.diag(((user_in_batch*self.S).sum(dim=0))[users])
        Q = (Db.T+Eb.T)
        Q1, Q2, N = self.f(Q, package['P1'], package['P2'])
        res = {'Q1':Q1, 'Q2':Q2, 'N':N}
        return res

    def reply_for_Z2(self, package):
        # set Q = Sb'
        users = package['users']
        Q = self.S[users].T
        Q1, Q2, N = self.f(Q, package['P1'], package['P2'])
        res = {'Q1':Q1, 'Q2':Q2, 'N':N}
        return res

    def f(self, Q, P1, P2):
        if Q.shape[0] % 2 == 1: Q = torch.cat((Q, torch.zeros((1, Q.shape[0]))), 0)
        Q = Q.to(self.device)
        # SSMM
        Q_rdm = torch.randn_like(Q)
        Q_rdme, Q_rdmo = self.split_matrix(Q_rdm, by='row')
        Q_rdm, Q_rdme, Q_rdmo = Q_rdm.to(self.device), Q_rdme.to(self.device), Q_rdmo.to(self.device)
        Q1 = Q_rdm-Q
        Q2 = Q_rdme-Q_rdmo
        N = torch.matmul(P1, (2*Q-Q_rdm)) - torch.matmul(P2, (Q2+Q_rdme))
        return Q1, Q2, N

    def split_matrix(self, M, by='row'):
        if by == 'row':
            odd_idx = [i for i in range(0, M.shape[0], 2)]
            even_idx = [i for i in range(1, M.shape[0], 2)]
            return M[even_idx], M[odd_idx]
        else:
            tmp1, tmp2 = self.split_matrix(M.T, by='row')
            return tmp1.T, tmp2.T