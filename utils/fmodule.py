from torch.utils.data import DataLoader, Dataset
import torch

device=None
optim = None
lossfunc=None

class XYDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = torch.tensor(xs)
        self.ys = torch.tensor(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        return self.xs[item], self.ys[item]

def train(model, dataset, epochs=1, learning_rate=0.1, batch_size=128, momentum=0):
    model.train()
    if batch_size == -1:
        # full gradient descent
        batch_size = len(dataset)
    ldr_train = DataLoader(dataset, batch_size= batch_size, shuffle=True)
    optimizer = optim(model.parameters(), lr=learning_rate, momentum=momentum)
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
        log_probs = model(features)
        loss += (lossfunc(log_probs, labels).item()*len(labels))
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    accuracy = float(correct) * 100.00 / len(dataset)
    loss/=len(dataset)
    return accuracy, loss

def modeldict_weighted_average(ws, weights=[]):
    if not ws:
        return None
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    if len(weights) == 0: weights = [1.0/len(ws) for _ in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
    return w_avg

def modeldict_to_device(w, device = device):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer].to(device)
    return res

def modeldict_to_cpu(w):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer].cpu()
    return res

def modeldict_zeroslike(w):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] - w[layer]
    return res

def modeldict_scale(w, c):
    res = {}
    for layer in w.keys():
        res[layer] = w[layer] * c
    return res

def modeldict_sub(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] - w2[layer]
    return res

def modeldict_norm(w, p=2):
    return torch.norm(modeldict_to_tensor1D(w), p)

def modeldict_to_tensor1D(w):
    res = torch.Tensor().to(w[list(w)[0]].device)
    for layer in w.keys():
        res = torch.cat((res, w[layer].view(-1)))
    return res

def modeldict_add(w1, w2):
    res = {}
    for layer in w1.keys():
        res[layer] = w1[layer] + w2[layer]
    return res

def modeldict_dot(w1, w2):
    res = 0
    for layer in w1.keys():
        s = 1
        for l in w1[layer].shape:
            s *= l
        res += (w1[layer].view(1, s).mm(w2[layer].view(1, s).T))
    return res

def modeldict_num_parameters(w):
    res = 0
    for layer in w.keys():
        s = 1
        for l in w[layer].shape:
            s *= l
        res += s
    return res

def modeldict_print(w):
    for layer in w.keys():
        print("{}:{}".format(layer, w[layer]))

