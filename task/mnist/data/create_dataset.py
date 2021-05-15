from torchvision import datasets, transforms
import numpy as np
import json
import random
num_clients=200

def sample(dataset, num_clients, pickshark=1):
    client_datasize = int(len(dataset) / num_clients)
    all_idxs = [i for i in range(len(dataset))]
    labels = [dataset[i][1] for i in all_idxs]
    z=zip(labels,all_idxs)
    z=sorted(z)
    labels, all_idxs=zip(*z)
    sharksize = int(client_datasize / pickshark)
    idxs_shard=range(int(num_clients * pickshark))
    cidxs=[[] for i in range(num_clients)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idxs_shard, pickshark, replace=False))
        idxs_shard = list(set(idxs_shard) - rand_set)
        for rand in rand_set:
            cidxs[i].extend(all_idxs[rand * sharksize:(rand + 1) * sharksize])
    return cidxs

if __name__ == '__main__':
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('./raw_data', train=True, download=True, transform=trans_mnist)
    # dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    cidxs = sample(dataset, num_clients, 2)
    train_output = "./train/mytrain.json"
    test_output = "./test/mytest.json"
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    for client in range(num_clients):
        cname = "user{}".format(client)
        train_data['users'].append(cname)
        train_data['user_data'][cname]={'x':[], 'y':[]}
        test_data['users'].append(cname)
        test_data['user_data'][cname] = {'x': [], 'y': []}
        dids = cidxs[client]
        random.shuffle(dids)
        trainlen = int(0.8*len(dids))
        train_data['num_samples'].append(trainlen)
        for did in dids[:trainlen]:
            train_data['user_data'][cname]['x'].append(dataset[did][0].tolist())
            train_data['user_data'][cname]['y'].append(dataset[did][1])
        test_data['num_samples'].append(len(dids)-trainlen)
        for did in dids[trainlen:]:
            test_data['user_data'][cname]['x'].append(dataset[did][0].tolist())
            test_data['user_data'][cname]['y'].append(dataset[did][1])
    with open(train_output,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_output, 'w') as outfile:
        json.dump(test_data, outfile)










