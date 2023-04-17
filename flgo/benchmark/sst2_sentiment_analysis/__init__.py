import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition
from .model import XLMR_Roberta
default_model = XLMR_Roberta
default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':10}
# default_model = cnn
# visualize = flgo.benchmark.toolkits.visualization.visualize_by_class