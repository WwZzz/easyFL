import flgo.benchmark.toolkits.visualization
from .model import fcn_resnet50, unet
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = fcn_resnet50
