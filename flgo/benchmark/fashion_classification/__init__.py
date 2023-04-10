import flgo.benchmark.toolkits.visualization
from flgo.benchmark.fashion_classification.model import lr
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = lr
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class