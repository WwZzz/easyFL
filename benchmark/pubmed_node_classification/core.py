from torch_geometric.datasets import Planetoid

from benchmark.toolkits.graph.horizontal.node_classification import *

TaskCalculator = NodeClassificationTaskCalculator
class TaskGenerator(NodeClassificationTaskGen):
    def __init__(self, *args, **kwargs):
        super(TaskGenerator, self).__init__(benchmark='pubmed_node_classification',
                                            rawdata_path='./benchmark/RAW_DATA/PUBMED', builtin_class=Planetoid,
                                            dataset_name='PubMed', num_clients=10)

class TaskPipe(NodeClassificationTaskPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, Planetoid, transform=None)