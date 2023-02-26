import matplotlib.pyplot as plt
import random
import matplotlib.colors
import collections
import numpy as np
import os

def visualize_by_class(generator, partitioner):
    all_labels = [d[-1] for d in generator.train_data]
    num_classes = len(set(all_labels))
    ax = plt.subplots()
    colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
    random.shuffle(colors)
    client_height = 1
    if hasattr(partitioner, 'dirichlet_dist'):
        client_dist = generator.partitioner.dirichlet_dist.tolist()
        data_columns = [sum(cprop) for cprop in client_dist]
        row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(generator.partitioner.num_parties)])}
        for cid, cprop in enumerate(client_dist):
            offset = 0
            y_bottom = row_map[cid] - client_height / 2.0
            y_top = row_map[cid] + client_height / 2.0
            for lbi in range(len(cprop)):
                plt.fill_between([offset, offset + cprop[lbi]], y_bottom, y_top, facecolor=colors[lbi])
                # plt.barh(cid, cprop[lbi], client_height, left=offset, color=)
                offset += cprop[lbi]
    else:
        data_columns = [len(cidx) for cidx in generator.local_datas]
        row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(generator.partitioner.num_parties)])}
        for cid, cidxs in enumerate(generator.local_datas):
            labels = [int(generator.train_data[did][-1]) for did in cidxs]
            lb_counter = collections.Counter(labels)
            offset = 0
            y_bottom = row_map[cid] - client_height / 2.0
            y_top = row_map[cid] + client_height / 2.0
            for lbi in range(num_classes):
                plt.fill_between([offset, offset + lb_counter[lbi]], y_bottom, y_top, facecolor=colors[lbi])
                offset += lb_counter[lbi]
    plt.xlim(0, max(data_columns))
    plt.ylim(-0.5, generator.partitioner.num_parties - 0.5)
    plt.ylabel('Client ID')
    plt.xlabel('Number of Samples')
    plt.show()
