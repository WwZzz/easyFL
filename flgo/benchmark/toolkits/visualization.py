import matplotlib.pyplot as plt
import random
import matplotlib.colors
import collections
import numpy as np
import os


def visualize_by_class(generator, partitioner, task_path:str):
    r"""
    Visualize the partitioned classification dataset and save the figure

    Args:
        generator (flgo.benchmark.toolkits.BasicTaskGenerator): task generator
        partitioner (flgo.benchmark.toolkits.partition.BasicPartitioner): partitioner
        task_path (str): the path storing the figure
    """
    all_labels = [d[-1] for d in generator.train_data]
    num_classes = len(set(all_labels))
    ax = plt.subplots()
    colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
    if len(colors)<num_classes: colors = list(matplotlib.colors._colors_full_map.keys())
    random.shuffle(colors)
    client_height = 1
    if hasattr(generator.partitioner, 'num_parties'):
        n = generator.partitioner.num_parties
    else:
        n = generator.partitioner.num_clients
    if hasattr(partitioner, 'dirichlet_dist'):
        client_dist = generator.partitioner.dirichlet_dist.tolist()
        data_columns = [sum(cprop) for cprop in client_dist]
        row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(n)])}
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
        row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(n)])}
        for cid, cidxs in enumerate(generator.local_datas):
            labels = [int(generator.train_data[did][-1]) for did in cidxs]
            lb_counter = collections.Counter(labels)
            offset = 0
            y_bottom = row_map[cid] - client_height / 2.0
            y_top = row_map[cid] + client_height / 2.0
            for lbi in range(num_classes):
                plt.fill_between([offset, offset + lb_counter[lbi]], y_bottom, y_top, facecolor=colors[lbi%len(colors)])
                offset += lb_counter[lbi]
    plt.xlim(0, max(data_columns))
    plt.ylim(-0.5, n- 0.5)
    plt.ylabel('Client ID')
    plt.xlabel('Number of Samples')
    plt.savefig(os.path.join(task_path, 'res.png'))
    plt.show()

def visualize_by_community(generator, partitioner, task_path:str):
    r"""
    Visualize the partitioned graph node-level dataset and save the figure

    Args:
        generator (flgo.benchmark.toolkits.BasicTaskGenerator): task generator
        partitioner (flgo.benchmark.toolkits.partition.BasicPartitioner): partitioner
        task_path (str): the path storing the figure
    """
    # from communities.
    import networkx as nx
    import community
    community = community.community_louvain.best_partition(generator.train_data)
    groups = collections.defaultdict(list)
    for ni, gi in community.items():
        groups[gi].append(ni)
    groups = list(groups.values())

    local_datas = generator.local_datas
    client_community = {}
    for cid, nodes in enumerate(local_datas):
        for node in nodes: client_community[node] = cid
    G = generator.train_data
    total_edges = len(G.edges)
    subgraphs = [nx.subgraph(G, li) for li in local_datas]
    local_edges = [len(g.edges) for g in subgraphs]
    edge_losing_rate = 1.0*(total_edges-sum(local_edges))/total_edges
    std_edge_num = np.std(local_edges)
    Gdegree = sum([G.degree[i] for i in G.nodes])/len(G.nodes)
    subgraphs_degree = [sum([g.degree[i] for i in g.nodes])/len(g.nodes) for g in subgraphs]
    std_degree = np.std(subgraphs_degree)
    mean_degree = np.mean(subgraphs_degree)
    title = "num_clients:{}\nglobal_edges:{} | edge_losing_rate:{:.4f} | std_edge_num:{:.4f}\nglobal_degree:{:.4f} | ave_degree:{:.4f} | std_degree:{:.4f}".format(len(local_datas), total_edges, edge_losing_rate, std_edge_num, Gdegree, mean_degree, std_degree)
    colors = [key for key in matplotlib.colors.XKCD_COLORS.keys()]
    pos = community_layout(G, client_community)
    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges), width=0.5, alpha=0.5)
    local_colors = ['r', 'b', 'g']
    for cid, cnodes in enumerate(local_datas):
        nx.draw_networkx_nodes(G, pos, cnodes, node_size=800,alpha=0.02,
                               node_color=local_colors[cid%len(local_colors)],
                               node_shape='o')

    for cid, cnodes in enumerate(groups):
        nx.draw_networkx_nodes(G, pos, cnodes, node_size=20,
                               node_color=colors[cid%len(colors)],
                               node_shape='>')

    plt.title(title)
    plt.savefig(os.path.join(task_path, 'res.png'))
    plt.show()
    return

def visualize_hier_by_class(generator, partitioner, task_path:str):
    r"""
    Visualize the partitioned classification dataset and save the figure

    Args:
        generator (flgo.benchmark.toolkits.BasicTaskGenerator): task generator
        partitioner (flgo.benchmark.toolkits.partition.BasicPartitioner): partitioner
        task_path (str): the path storing the figure
    """
    import torch.utils.data
    dataloader = torch.utils.data.DataLoader(generator.train_data, batch_size=50, shuffle=False)
    all_labels = []
    for _, batch in enumerate(dataloader):
        all_labels.append(batch[-1])
    all_labels = torch.cat(all_labels).tolist()
    # all_labels = [d[-1] for d in generator.train_data]
    num_classes = len(set(all_labels))
    colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
    if len(colors)<num_classes: colors = list(matplotlib.colors._colors_full_map.keys())
    random.shuffle(colors)
    client_height = 1
    local_datas = generator.local_datas
    num_edge_servers = len(local_datas)
    cols = 5
    rows = num_edge_servers//cols
    if num_edge_servers%cols!=0: rows += 1
    if rows==1: cols = num_edge_servers
    fig = plt.figure(figsize=(3*cols, 3*rows))
    for i in range(num_edge_servers):
        plt.subplot(rows, cols, i+1)
        edge_datas = local_datas[i]
        data_columns = [len(cidx) for cidx in edge_datas]
        row_map = {k: i for k, i in zip(np.argsort(data_columns), [_ for _ in range(len(edge_datas))])}
        for cid, cidxs in enumerate(edge_datas):
            labels = [int(all_labels[did]) for did in cidxs]
            lb_counter = collections.Counter(labels)
            offset = 0
            y_bottom = row_map[cid] - client_height / 2.0
            y_top = row_map[cid] + client_height / 2.0
            for lbi in range(num_classes):
                plt.fill_between([offset, offset + lb_counter[lbi]], y_bottom, y_top, facecolor=colors[lbi%len(colors)])
                offset += lb_counter[lbi]
        plt.title(f'edge_server {i}')
        plt.xlim(0, max(data_columns))
        plt.ylim(-0.5, len(edge_datas)- 0.5)
        plt.ylabel('Client ID')
        plt.xlabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(task_path, 'res.png'))
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos