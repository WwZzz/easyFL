from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn

device=None
lossfunc=None
Optim = None
Model = None

def get_optimizer(name="SGD", model = None, lr = 0.1, weight_decay = 0, momentum = 0):
    if name.lower() == 'sgd':
        return Optim(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name.lower() == 'adam':
        return Optim(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
    else:
        return None

class XYDataset(Dataset):
    def __init__(self, xs=[], ys=[]):
        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

class FModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.ingraph = False

    def __add__(self, other):
        if isinstance(other, int) and other == 0 : return self
        if not isinstance(other, FModule): raise TypeError
        return _model_add(self, other)

    def __radd__(self, other):
        return _model_add(self, other)

    def __sub__(self, other):
        if isinstance(other, int) and other == 0: return self
        if not isinstance(other, FModule): raise TypeError
        return _model_sub(self, other)

    def __mul__(self, other):
        return _model_scale(self, other)

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        return self*(1.0/other)

    def __pow__(self, power, modulo=None):
        return _model_norm(self, power)

    def __neg__(self):
        return _model_scale(self, -1.0)

    def norm(self, p=2):
        return self**p

    def zeros_like(self):
        return self*0

    def dot(self, other):
        return _model_dot(self, other)

    def cos_sim(self, other):
        return _model_cossim(self, other)

    def op_with_graph(self):
        self.ingraph = True

    def op_without_graph(self):
        self.ingraph = False

    def load(self, other):
        self.op_without_graph()
        self.load_state_dict(other.state_dict())
        return

    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def zero_dict(self):
        self.op_without_graph()
        for p in self.parameters():
            p.data.zero_()

    def normalize(self):
        self.op_without_graph()
        self.load_state_dict((self/(self**2)).state_dict())

    def get_device(self):
        return next(self.parameters()).device

def normalize(w):
    return w/(w**2)

def dot(w1, w2):
    return w1.dot(w2)

def cos_sim(w1, w2):
    return w1.cos_sim(w2)

def exp(w):
    """element-wise exp"""
    return element_wise_func(w, torch.exp)

def log(w):
    """element-wise log"""
    return element_wise_func(w, torch.log)

def element_wise_func(w, func):
    if not w: return None
    res = Model().to(w.get_device())
    if w.ingraph:
        res.op_with_graph()
        wl = get_module_from_model(w)
        for wd in wl:
            rd = _modeldict_element_wise(wd._parameters, func)
            for l in wd._parameters.keys():
                wd._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_element_wise(w.state_dict(), func))
    return res

def sum(ms):
    if not ms: return None
    op_with_graph = sum([w.ingraph for w in ms]) > 0
    res = Model().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_sum(mpks)
            for l in mlr[n]._parameters.keys():
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        _modeldict_cp(res.state_dict(), _modeldict_sum([w.state_dict() for w in ms]))
    return res

def average(ms = [], p = []):
    if not ms: return None
    if not p: p = [1.0 / len(ms) for _ in range(len(ms))]
    op_with_graph = sum([w.ingraph for w in ms]) > 0
    res = Model().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_weighted_average(mpks, p)
            for l in mlr[n]._parameters.keys():
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        _modeldict_cp(res.state_dict(), _modeldict_weighted_average([mi.state_dict() for mi in ms], p))
    return res

def _model_add(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = Model().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_add(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_add(m1.state_dict(), m2.state_dict()))
    return res

def _model_sub(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = Model().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_sub(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_sub(m1.state_dict(), m2.state_dict()))
    return res

def _model_scale(m, s):
    op_with_graph = m.ingraph
    res = Model().to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        mlr = get_module_from_model(res)
        res.op_with_graph()
        for n, nr in zip(ml, mlr):
            rd = _modeldict_scale(n._parameters, s)
            for l in nr._parameters.keys():
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_scale(m.state_dict(), s))
    return res

def _model_norm(m, power=2):
    op_with_graph = m.ingraph
    res = torch.tensor(0.).to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        for n in ml:
            for l in n._parameters.keys():
                if n._parameters[l] is None: continue
                if n._parameters[l].dtype not in [torch.float, torch.float32, torch.float64]: continue
                res += torch.sum(torch.pow(n._parameters[l], power))
        return torch.pow(res, 1.0 / power).view(-1)
    else:
        return _modeldict_norm(m.state_dict(), power)

def _model_dot(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = torch.tensor(0.).to(m1.get_device())
    if op_with_graph:
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
        return res.view(-1)
    else:
        _modeldict_cp(res.state_dict(), _modeldict_dot(m1.state_dict(), m2.state_dict()))
    return res

def _model_cossim(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = torch.tensor(0.).to(m1.get_device())
    if op_with_graph:
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        l1 = torch.tensor(0.).to(m1.device)
        l2 = torch.tensor(0.).to(m1.device)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
            for l in n1._parameters.keys():
                l1 += torch.sum(torch.pow(n1._parameters[l], 2))
                l2 += torch.sum(torch.pow(n2._parameters[l], 2))
        return (res / torch.pow(l1, 0.5) * torch(l2, 0.5)).view(-1)
    else:
        _modeldict_cp(res.state_dict(), _modeldict_cossim(m1.state_dict(), m2.state_dict()))
    return res

def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res

def train(model, dataset, epochs=1, learning_rate=0.1, batch_size=128, momentum=0):
    model.train()
    if batch_size == -1:
        # full gradient descent
        batch_size = len(dataset)
    ldr_train = DataLoader(dataset, batch_size= batch_size, shuffle=True)
    optimizer = Optim(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_loss = []
    for iter in range(epochs):
        batch_loss = []
        for batch_idx, (features, labels) in enumerate(ldr_train):
            features, labels = features.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(features)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/len(labels))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return sum(epoch_loss) / len(epoch_loss)

@torch.no_grad()
def test(model, dataset):
    model.eval()
    loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=64)
    for idx, (features, labels) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss += (lossfunc(outputs, labels).item()*len(labels))
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    accuracy = float(correct) * 100.00 / len(dataset)
    loss/=len(dataset)
    return accuracy, loss

def _modeldict_cp(wd1, wd2):
    for layer in wd1.keys():
        wd1[layer].data.copy_(wd2[layer])
    return

def _modeldict_sum(wds):
    if not wds: return None
    wd_sum = {}
    for layer in wds[0].keys():
        wd_sum[layer] = torch.zeros_like(wds[0][layer])
    for wid in range(len(wds)):
        for layer in wd_sum.keys():
            if wds[0][layer] == None:
                wd_sum[layer] = None
            wd_sum[layer] = wd_sum[layer] + wds[wid][layer]
    return wd_sum

def _modeldict_weighted_average(wds, weights=[]):
    if not wds:
        return None
    wd_avg = {}
    for layer in wds[0].keys(): wd_avg[layer] = torch.zeros_like(wds[0][layer])
    if len(weights) == 0: weights = [1.0 / len(wds) for _ in range(len(wds))]
    for wid in range(len(wds)):
        for layer in wd_avg.keys():
            if wds[0][layer] == None:
                wd_avg[layer] = None
            weight = weights[wid] if "num_batches_tracked" not in layer else 1
            wd_avg[layer] = wd_avg[layer] + wds[wid][layer] * weight
    return wd_avg

def _modeldict_to_device(wd, device = device):
    res = {}
    for layer in wd.keys():
        if wd[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd[layer].to(device)
    return res

def _modeldict_to_cpu(wd):
    res = {}
    for layer in wd.keys():
        if wd[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd[layer].cpu()
    return res

def _modeldict_zeroslike(wd):
    res = {}
    for layer in wd.keys():
        if wd[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd[layer] - wd[layer]
    return res

def _modeldict_add(wd1, wd2):
    res = {}
    for layer in wd1.keys():
        if wd1[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd1[layer] + wd2[layer]
    return res

def _modeldict_scale(wd, c):
    res = {}
    for layer in wd.keys():
        if wd[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd[layer] * c
    return res

def _modeldict_sub(wd1, wd2):
    res = {}
    for layer in wd1.keys():
        if wd1[layer] is None:
            res[layer] = None
            continue
        res[layer] = wd1[layer] - wd2[layer]
    return res

def _modeldict_norm(wd, p=2):
    res = torch.tensor(0.).to(wd[list(wd)[0]].device)
    for layer in wd.keys():
        if wd[layer] is None: continue
        if wd[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
        res += torch.sum(torch.pow(wd[layer], p))
    return torch.pow(res, 1.0/p).view(-1)

def _modeldict_to_tensor1D(wd):
    res = torch.Tensor().type_as(wd[list(wd)[0]]).to(wd[list(wd)[0]].device)
    for layer in wd.keys():
        if wd[layer] is None:
            continue
        res = torch.cat((res, wd[layer].view(-1)))
    return res

def _modeldict_dot(wd1, wd2):
    res = torch.tensor(0.).to(wd1[list(wd1)[0]].device)
    for layer in wd1.keys():
        if wd1[layer] is None:
            continue
        if wd1[layer].dtype not in [torch.float, torch.float32, torch.float64]:continue
        res += (wd1[layer].view(-1).dot(wd2[layer].view(-1)))
    return res.view(-1)

def _modeldict_cossim(wd1, wd2):
    res = torch.tensor(0.).to(wd1[list(wd1)[0]].device)
    l1 = torch.tensor(0.).to(wd1[list(wd1)[0]].device)
    l2 = torch.tensor(0.).to(wd1[list(wd1)[0]].device)
    for layer in wd1.keys():
        if wd1[layer] is None:
            continue
        if wd1[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
        res += (wd1[layer].view(-1).dot(wd2[layer].view(-1)))
        l1 += torch.sum(torch.pow(wd1[layer], 2))
        l2 += torch.sum(torch.pow(wd2[layer], 2))
    return (res/(torch.pow(l1, 0.5)*torch.pow(l2, 0.5))).view(-1)

def _modeldict_element_wise(wd, func):
    res = {}
    for layer in wd.keys():
        if wd[layer] is None:
            res[layer] = None
            continue
        res[layer] = func(wd[layer])
    return res

def _modeldict_num_parameters(wd):
    res = 0
    for layer in wd.keys():
        if wd[layer] is None:
            continue
        s = 1
        for l in wd[layer].shape:
            s *= l
        res += s
    return res

def _modeldict_print(wd):
    for layer in wd.keys():
        print("{}:{}".format(layer, wd[layer]))
