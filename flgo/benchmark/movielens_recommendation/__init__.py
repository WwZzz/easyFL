import flgo.benchmark.toolkits.partition
from .model import mf
default_model = mf
default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':10}
