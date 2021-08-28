from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn

device=None
lossfunc=None
Optim = None
Model = None

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

    def __add__(self, other):
        if isinstance(other, int) and other == 0 : return self
        if not isinstance(other, FModule): raise TypeError
        res = Model().to(device)
        res.load_state_dict(modeldict_add(self.state_dict(), other.state_dict()))
        return res

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        if not isinstance(other, FModule): raise TypeError
        res = Model().to(device)
        res.load_state_dict(modeldict_sub(self.state_dict(), other.state_dict()))
        return res

    def __mul__(self, other):
        res = Model().to(device)
        res.load_state_dict(modeldict_scale(self.state_dict(), other))
        return res

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        res = Model().to(device)
        res.load_state_dict(modeldict_scale(self.state_dict(), 1.0/other))
        return res

    def __pow__(self, power, modulo=None):
        return modeldict_norm(self.state_dict(), power)

    def __neg__(self):
        res = Model().to(device)
        res.load_state_dict(modeldict_scale(self.state_dict(), -1.0))
        return res

    def load(self, other):
        self.load_state_dict(other.state_dict())
        return

    def zero_dict(self):
        for p in self.parameters():
            p.data.zero_()

    def normalize(self):
        self.load_state_dict((self/(self**2)).state_dict())
        return

    def add(self, other):
        self.load_state_dict(modeldict_add(self.state_dict(), other.state_dict()))
        return

    def sub(self, other):
        self.load_state_dict(modeldict_sub(self.state_dict(), other.state_dict()))
        return

    def mul(self, other):
        self.load_state_dict(modeldict_scale(self.state_dict(),other))
        return

    def div(self, other):
        self.load_state_dict(modeldict_scale(self.state_dict(), 1.0/other))
        return

    def neg(self):
        self.load_state_dict(modeldict_scale(self.state_dict(), -1))
        return

    def norm(self, p=2):
        return self**p

    def zeros_like(self):
        return self-self

    def dot(self, other):
        return modeldict_dot(self.state_dict(), other.state_dict())

    def cos_sim(self, other):
        return (self/self**2).dot(other/other**2)

def sum(ws):
    if not ws: return None
    res = Model().to(device)
    res.load_state_dict(modeldict_sum([w.state_dict() for w in ws]))
    return res

def normalize(w):
    return w/(w**2)

def dot(w1, w2):
    return w1.dot(w2)

def average(ws = [], p = []):
    if not ws: return None
    if not p: p = [1.0 / len(ws) for _ in range(len(ws))]
    res = Model().to(device)
    res.load_state_dict(modeldict_weighted_average([w.state_dict() for w in ws], p))
    return res

def cos_sim(w1, w2):
    return w1.cos_sim(w2)

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

def modeldict_sum(wds):
    if not wds: return None
    wd_sum = {}
    for layer in wds[0].keys():
        wd_sum[layer] = torch.zeros_like(wds[0][layer])
    for wid in range(len(wds)):
        for layer in wd_sum.keys():
            wd_sum[layer] = wd_sum[layer] + wds[wid][layer]
    for layer in wds[0].keys():
        wd_sum[layer] = wd_sum[layer].type_as(wds[0][layer])
    return wd_sum

def modeldict_weighted_average(wds, weights=[]):
    if not wds:
        return None
    wd_avg = {}
    for layer in wds[0].keys():
        wd_avg[layer] = torch.zeros_like(wds[0][layer])
    if len(weights) == 0: weights = [1.0 / len(wds) for _ in range(len(wds))]
    for wid in range(len(wds)):
        for layer in wd_avg.keys():
            wd_avg[layer] = wd_avg[layer] + wds[wid][layer] * weights[wid]
    for layer in wds[0].keys():
        wd_avg[layer] = wd_avg[layer].type_as(wds[0][layer])
    return wd_avg

def modeldict_to_device(wd, device = device):
    res = {}
    for layer in wd.keys():
        res[layer] = wd[layer].to(device)
    return res

def modeldict_to_cpu(wd):
    res = {}
    for layer in wd.keys():
        res[layer] = wd[layer].cpu()
    return res

def modeldict_zeroslike(wd):
    res = {}
    for layer in wd.keys():
        res[layer] = wd[layer] - wd[layer]
    return res

def modeldict_scale(wdict, c):
    res = {}
    for layer in wdict.keys():
        res[layer] = wdict[layer] * c
    return res

def modeldict_sub(wd1, wd2):
    res = {}
    for layer in wd1.keys():
        res[layer] = wd1[layer] - wd2[layer]
    return res

def modeldict_norm(wd, p=2):
    res = torch.Tensor([0.]).to(wd[list(wd)[0]].device)
    for layer in wd.keys():
        if wd[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
        res += torch.pow(torch.norm(wd[layer], p),p)
    return torch.pow(res, 1.0/p)

def modeldict_to_tensor1D(wd):
    res = torch.Tensor().type_as(wd[list(wd)[0]]).to(wd[list(wd)[0]].device)
    for layer in wd.keys():
        res = torch.cat((res, wd[layer].view(-1)))
    return res

def modeldict_add(wd1, wd2):
    res = {}
    for layer in wd1.keys():
        res[layer] = wd1[layer] + wd2[layer]
    return res

def modeldict_dot(wd1, wd2):
    res = torch.Tensor([0.]).to(wd1[list(wd1)[0]].device)
    for layer in wd1.keys():
        if wd1[layer].dtype not in [torch.float, torch.float32, torch.float64]:continue
        res += (wd1[layer].view(-1).dot(wd2[layer].view(-1)))
    return res.view(-1)

def modeldict_num_parameters(wd):
    res = 0
    for layer in wd.keys():
        s = 1
        for l in wd[layer].shape:
            s *= l
        res += s
    return res

def modeldict_print(wd):
    for layer in wd.keys():
        print("{}:{}".format(layer, wd[layer]))

