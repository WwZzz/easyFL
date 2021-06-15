import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

device=None
optim = None
lossfunc=None

def train(model, dataset, epochs=1, learning_rate=0.1, batch_size=128, momentum=0, regularzation=0):
    model.train()
    if batch_size == -1:
        # full gradient descent
        batch_size = len(dataset)
    ldr_train = DataLoader(dataset, batch_size= batch_size, shuffle=True)
    optimizer = optim(model.parameters(), lr=learning_rate, momentum=momentum)
    epoch_loss = []
    for iter in range(epochs):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(images)
            loss = lossfunc(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item()/len(labels))
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    return sum(epoch_loss) / len(epoch_loss)

def test(model, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    l = len(data_loader)
    for idx, (features, labels) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        log_probs = model(features)
        test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(dataset)
    accuracy = float(correct) * 100.00 / len(data_loader.dataset)
    return accuracy, test_loss


def modeldict_weighted_average(ws, weights=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    if weights == []: weights = [1.0/len(ws) for i in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
    return w_avg

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
    res = torch.Tensor().to(device)
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
    return res.item()

def modeldict_print(w):
    for layer in w.keys():
        print("{}:{}".format(layer, w[layer]))

if __name__ == '__main__':
    w= {'a': torch.Tensor([[1, 4], [3, 4]]),  'c':torch.Tensor([1])}
    res=modeldict_norm(w)
    print(res**2)