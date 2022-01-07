import matplotlib.pyplot as plt
import pandas as pd
import ujson
import os


def read_data():
    with open(os.path.join('../fedtask/', task, 'data.json'), 'r') as inf:
        feddata = ujson.load(inf)
    train_datas = [feddata[name]['dtrain']['y'] for name in feddata['client_names']]
    valid_datas = [feddata[name]['dvalid']['y'] for name in feddata['client_names']]
    test_datas = feddata['dtest']['y']
    return train_datas, valid_datas, test_datas, feddata['client_names']


def draw_partition():
    dict = {}
    label = [i for i in range(num_class)]
    train_datas, valid_datas, test_datas, cnames = read_data()
    if partition_on == 'train':
        for cname, train_data in zip(cnames, train_datas):
            dict[cname] = [train_data.count(y) for y in label]
    elif partition_on == 'valid':
        for cname, train_data in zip(cnames, valid_datas):
            dict[cname] = [train_data.count(y) for y in label]
    elif partition_on == 'test':
        for cname in cnames:
            dict[cname] = [test_datas.count(y) for y in label]

    count_box = {}
    for y in label:
        temp = []
        for cname in cnames:
            temp.append(dict[cname][y])
        count_box[y] = temp

    df = pd.DataFrame(count_box)
    df[label].plot.barh(stacked=True)
    # plt.figure(figsize=(100, 100), dpi=100)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(task)
    plt.ylabel('client ' + partition_on)
    plt.xlabel('sample number')
    plt.show()


if __name__ == '__main__':
    task = 'mnist_cnum100_dist0_skew0_seed0'
    partition_on = 'train'
    # partition_on = 'valid'
    # partition_on = 'test'
    num_class = 10
    draw_partition()
