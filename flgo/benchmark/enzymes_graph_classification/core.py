from torch_geometric.datasets import TUDataset
import flgo.benchmark
from flgo.benchmark.toolkits.graph.graph_classification import *

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'ENZYMES'),*args, **kwargs):
        super(TaskGenerator, self).__init__(benchmark='enzymes_graph_classification',rawdata_path=rawdata_path,
                                            builtin_class=TUDataset, dataset_name='Enzymes')

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, TUDataset, transform=None)